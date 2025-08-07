# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os, zipfile, tempfile
import PyPDF2
import shutil
from groq import Groq
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import re
import logging
import datetime
import glob

# --- App and Logging Configuration ---
app = FastAPI()

# Create a logs directory if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define the log file name with a timestamp
log_file = os.path.join(log_dir, f"app_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Groq and API setup ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable not set.")
    raise ValueError("GROQ_API_KEY environment variable not set.")

groq_client = Groq(api_key=GROQ_API_KEY)

# --- ChromaDB and Embedding Model setup ---
client = chromadb.PersistentClient(path="./invoices_db")
try:
    invoice_collection = client.get_or_create_collection(name="invoice_analyses")
    logger.info("ChromaDB collection 'invoice_analyses' is ready.")
except Exception as e:
    logger.critical(f"Error initializing ChromaDB: {e}", exc_info=True)
    raise

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("SentenceTransformer model 'all-MiniLM-L6-v2' loaded.")


# --- File handling and PDF parsing functions ---
async def save_uploaded_files(policy_file: UploadFile, invoice_zip: UploadFile):
    """Saves uploaded policy and invoice files to a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    policy_path = os.path.join(temp_dir, "policy.pdf")
    invoices_dir = os.path.join(temp_dir, "invoices")

    # Save policy file
    try:
        with open(policy_path, "wb") as f:
            f.write(await policy_file.read())
        logger.info(f"Policy file saved to: {policy_path}")
    except Exception as e:
        logger.error(f"Failed to save policy file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save policy file.")

    # Extract invoices directly from the UploadFile object
    os.makedirs(invoices_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(invoice_zip.file, 'r') as zip_ref:
            zip_ref.extractall(invoices_dir)
        logger.info(f"Successfully extracted invoices to: {invoices_dir}")
    except zipfile.BadZipFile:
        logger.error("Uploaded file is not a valid zip file.")
        raise HTTPException(status_code=400, detail="The uploaded invoice file is not a valid zip file.")
    except Exception as e:
        logger.error(f"Failed to extract zip file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract zip file: {e}")
    
    return policy_path, invoices_dir

def parse_pdf(file_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        logger.info(f"Successfully parsed PDF: {file_path}")
    except Exception as e:
        logger.error(f"Error parsing PDF {file_path}: {e}")
        return ""
    return text

def analyze_invoice_with_groq(policy_text, invoice_text):
    """Analyzes a single invoice against a policy using the Groq API."""
    prompt_template = """
    You are an AI assistant for an invoice reimbursement system.
    Your task is to analyze the following reimbursement policy and invoice details.
    
    Reimbursement Policy:
    {policy}
    
    Invoice Details:
    {invoice}
    
    Based on the policy, determine the reimbursement status (Fully Reimbursed, Partially Reimbursed, Declined) and a clear, concise reason.
    Respond with the following format:
    Status: <Status>
    Reason: <Reason>
    """
    
    full_prompt = prompt_template.format(policy=policy_text, invoice=invoice_text)
    
    try:
        logger.info("Calling Groq API for invoice analysis...")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                { "role": "user", "content": full_prompt }
            ],
            model="llama3-8b-8192", 
            temperature=0.7
        )
        
        content = chat_completion.choices[0].message.content
        logger.info("Successfully received analysis from Groq API.")
        return {"status": "success", "analysis": content}
        
    except Exception as e:
        error_message = f"An error occurred with the Groq API: {e}"
        logger.error(error_message, exc_info=True)
        return {"status": "error", "message": error_message}

def create_embeddings_and_store(invoice_id, invoice_text, analysis_result, employee_name):
    """Generates embeddings and stores the data in ChromaDB."""
    status_line = next((line for line in analysis_result.split('\n') if line.startswith("Status:")), None)
    reason_line = next((line for line in analysis_result.split('\n') if line.startswith("Reason:")), None)
    
    reimbursement_status = status_line.split("Status:")[1].strip() if status_line else "Unknown"
    detailed_reason = reason_line.split("Reason:")[1].strip() if reason_line else "No reason provided."

    date_match = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', invoice_text)
    date_of_invoice = date_match.group(1) if date_match else "Unknown"

    try:
        invoice_embedding = embedding_model.encode(invoice_text).tolist()
        reason_embedding = embedding_model.encode(detailed_reason).tolist()
    except Exception as e:
        logger.error(f"Error encoding embeddings for invoice {invoice_id}: {e}")
        return False

    doc_id_base = str(uuid.uuid4())
    invoice_doc_id = f"{doc_id_base}-invoice"
    reason_doc_id = f"{doc_id_base}-reason"

    try:
        invoice_collection.add(
            embeddings=[invoice_embedding, reason_embedding],
            documents=[invoice_text, detailed_reason],
            metadatas=[
                {
                    "invoice_id": invoice_id, 
                    "reimbursement_status": reimbursement_status,
                    "employee_name": employee_name,
                    "date": date_of_invoice,
                    "type": "invoice_content"
                },
                {
                    "invoice_id": invoice_id, 
                    "reimbursement_status": reimbursement_status,
                    "employee_name": employee_name,
                    "date": date_of_invoice,
                    "type": "analysis_reason"
                }
            ],
            ids=[invoice_doc_id, reason_doc_id]
        )
        logger.info(f"Successfully stored analysis for invoice: {invoice_id}")
        return True
    except Exception as e:
        logger.error(f"Error storing data in ChromaDB for invoice {invoice_id}: {e}", exc_info=True)
        return False

# --- Part One: Main API Endpoint ---
@app.post("/analyze-invoices/")
async def analyze_invoices_endpoint(
    policy_file: UploadFile = File(...),
    invoice_zip: UploadFile = File(...),
    employee_name: str = Form(...)
):
    """
    Endpoint to analyze multiple invoices from a zip file against a policy and store results in a vector store.
    """
    temp_dir_to_clean = None
    results = []
    logger.info(f"Received new request to analyze invoices for employee: {employee_name}")
    
    try:
        policy_path, invoices_dir = await save_uploaded_files(policy_file, invoice_zip)
        temp_dir_to_clean = os.path.dirname(invoices_dir)
        
        policy_text = parse_pdf(policy_path)
        if not policy_text:
            logger.warning("Could not extract text from policy PDF.")
            raise HTTPException(status_code=400, detail="Could not extract text from policy PDF.")
        
        invoice_files = glob.glob(os.path.join(invoices_dir, '**', '*.pdf'), recursive=True)
        # [f for f in os.listdir(invoices_dir) if f.endswith(".pdf")]
        logger.info(f"Found {len(invoice_files)} invoices in the zip file.")

        if not invoice_files:
            logger.warning("No PDF files found in the invoices zip file.")
            return JSONResponse(content={"results": []})
        
        for filename in invoice_files:
            logger.info(f"Processing invoice: {filename}")
            invoice_path = os.path.join(invoices_dir, filename)
            invoice_text = parse_pdf(invoice_path)
            
            if not invoice_text:
                results.append({"invoice": filename, "employee": employee_name, "analysis": {"status": "error", "message": "Could not extract text from invoice PDF."}, "vector_store_status": "skipped"})
                continue
            
            analysis = analyze_invoice_with_groq(policy_text, invoice_text)
            
            storage_status = "skipped"
            if analysis["status"] == "success":
                invoice_id = os.path.basename(filename).split('.')[0] # Use os.path.basename to get filename from path
                storage_success = create_embeddings_and_store(
                    invoice_id, 
                    invoice_text, 
                    analysis["analysis"],
                    employee_name
                )
                storage_status = "success" if storage_success else "failure"

            results.append({
                "invoice": filename,
                "employee": employee_name,
                "analysis": analysis,
                "vector_store_status": storage_status
            })

        return JSONResponse(content={"results": results})
        
    except HTTPException:
        # Re-raise HTTPExceptions to be handled by FastAPI
        raise
    except Exception as e:
        logger.critical(f"An unexpected error occurred during invoice analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        if temp_dir_to_clean and os.path.exists(temp_dir_to_clean):
            shutil.rmtree(temp_dir_to_clean)
            logger.info(f"Cleaned up temporary directory: {temp_dir_to_clean}")

# --- Part Two: RAG Chatbot Endpoint ---
# A simple in-memory store for chat history
chat_history = {}

def parse_metadata_from_query(query: str):
    """
    Parses a user query to extract metadata filters.
    """
    filters = {}
    
    name_match = re.search(r"for\s+([A-Za-z\s]+)", query, re.IGNORECASE)
    if name_match:
        filters["employee_name"] = name_match.group(1).strip()
    
    status_match = re.search(r"(fully reimbursed|partially reimbursed|declined)", query, re.IGNORECASE)
    if status_match:
        filters["reimbursement_status"] = status_match.group(1).capitalize()
    
    date_match = re.search(r"from\s+(\d{4}-\d{2}-\d{2})", query, re.IGNORECASE)
    if date_match:
        filters["date"] = date_match.group(1)
    
    query_text = query
    for val in filters.values():
        query_text = query_text.replace(val, "").replace("from", "").replace("for", "").strip()

    return query_text, filters

def build_chromadb_where(metadata_filters):
    """
    Dynamically builds a ChromaDB 'where' clause based on the number of filters.
    """
    if not metadata_filters:
        return {} # No filter needed
    
    # If there's only one filter, return it directly without $and
    if len(metadata_filters) == 1:
        key, value = list(metadata_filters.items())[0]
        return {key: {"$eq": value}}
    
    # If there are multiple filters, use $and
    return {
        "$and": [
            {key: {"$eq": value}}
            for key, value in metadata_filters.items()
        ]
    }

def rag_query(query_text: str, metadata_filters: dict, user_id: str):
    try:
        query_embedding = embedding_model.encode(query_text).tolist()
        chromadb_where = build_chromadb_where(metadata_filters)
        
        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": 10
        }
        if chromadb_where:  # Only add 'where' if not empty
            query_args["where"] = chromadb_where

        # The error occurred on this line
        results = invoice_collection.query(**query_args)
    except Exception as e:
        logger.error(f"Error querying ChromaDB with RAG for user '{user_id}': {e}", exc_info=True)
        return "An internal error occurred while retrieving information."
    
    retrieved_docs = results['documents'][0]
    
    if not retrieved_docs:
        context = "No relevant documents were found for this query."
        logger.info(f"No documents found for RAG query from user '{user_id}'.")
    else:
        context_parts = []
        for i in range(len(retrieved_docs)):
            doc = retrieved_docs[i]
            metadata = results['metadatas'][0][i]
            context_parts.append(f"--- Document {i+1} ---\nEmployee: {metadata.get('employee_name', 'N/A')}\nInvoice ID: {metadata.get('invoice_id', 'N/A')}\nStatus: {metadata.get('reimbursement_status', 'N/A')}\nDate: {metadata.get('date', 'N/A')}\nContent:\n{doc}\n--- End Document ---\n")
        context = "\n".join(context_parts)
        logger.info(f"Found {len(retrieved_docs)} relevant documents for RAG query from user '{user_id}'.")

    previous_context = chat_history.get(user_id, "")
    
    llm_prompt = f"""
    You are an AI assistant designed to answer questions about invoice reimbursement analyses.
    Use the following retrieved information and your previous conversation history to answer the user's query.
    Your response should be clear, concise, and formatted in markdown.

    Previous Conversation History:
    {previous_context}

    Retrieved Documents:
    {context}

    User's Query:
    {query_text}

    Answer:
    """

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": llm_prompt}
            ],
            model="llama3-8b-8192", 
            temperature=0.3
        )
        response_content = chat_completion.choices[0].message.content
        chat_history[user_id] = f"{previous_context}\nUser: {query_text}\nAI: {response_content}"
        logger.info(f"Chatbot response generated for user '{user_id}'.")
        return response_content
    except Exception as e:
        logger.error(f"An error occurred with the Groq API during chat for user '{user_id}': {e}", exc_info=True)
        return "Sorry, I am unable to process your request at this time."

@app.post("/chat/")
async def chat_endpoint(
    query: str = Form(...),
    user_id: str = Form(...)
):
    """
    RAG LLM Chatbot Endpoint.
    """
    try:
        query_text, metadata_filters = parse_metadata_from_query(query)
        response = rag_query(query_text, metadata_filters, user_id)
        
        return JSONResponse(content={"response": response})
    except Exception as e:
        logger.critical(f"An unexpected error occurred in the chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)