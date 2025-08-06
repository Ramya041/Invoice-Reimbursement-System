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

app = FastAPI()

# --- File handling and PDF parsing functions ---
async def save_uploaded_files(policy_file: UploadFile, invoice_zip: UploadFile):
    """Saves uploaded policy and invoice files to a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    policy_path = os.path.join(temp_dir, "policy.pdf")
    zip_path = os.path.join(temp_dir, "invoices.zip")
    invoices_dir = os.path.join(temp_dir, "invoices")

    with open(policy_path, "wb") as f:
        f.write(await policy_file.read())
    with open(zip_path, "wb") as f:
        f.write(await invoice_zip.read())

    os.makedirs(invoices_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(invoices_dir)

    return policy_path, invoices_dir

def parse_pdf(file_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error parsing PDF {file_path}: {e}")
        return ""
    return text

# --- Groq and API setup ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

groq_client = Groq(api_key=GROQ_API_KEY)

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
        chat_completion = groq_client.chat.completions.create(
            messages=[
                { "role": "user", "content": full_prompt }
            ],
            model="llama3-8b-8192", 
            temperature=0.7
        )
        
        content = chat_completion.choices[0].message.content
        return {"status": "success", "analysis": content}
        
    except Exception as e:
        error_message = f"An error occurred with the Groq API: {e}"
        print(f"DEBUG: {error_message}")
        return {"status": "error", "message": error_message}

# --- ChromaDB and Embedding Model setup ---
client = chromadb.PersistentClient(path="./invoices_db")
try:
    invoice_collection = client.get_or_create_collection(name="invoice_analyses")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    raise

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings_and_store(invoice_id, invoice_text, analysis_result, employee_name):
    """Generates embeddings and stores the data in ChromaDB."""
    status_line = next((line for line in analysis_result.split('\n') if line.startswith("Status:")), None)
    reason_line = next((line for line in analysis_result.split('\n') if line.startswith("Reason:")), None)
    
    reimbursement_status = status_line.split("Status:")[1].strip() if status_line else "Unknown"
    detailed_reason = reason_line.split("Reason:")[1].strip() if reason_line else "No reason provided."

    date_match = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', invoice_text)
    date_of_invoice = date_match.group(1) if date_match else "Unknown"

    invoice_embedding = embedding_model.encode(invoice_text).tolist()
    reason_embedding = embedding_model.encode(detailed_reason).tolist()

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
        print(f"Successfully stored analysis for invoice: {invoice_id}")
        return True
    except Exception as e:
        print(f"Error storing data in ChromaDB: {e}")
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
    
    try:
        policy_path, invoices_dir = await save_uploaded_files(policy_file, invoice_zip)
        temp_dir_to_clean = os.path.dirname(invoices_dir)
        
        policy_text = parse_pdf(policy_path)
        
        invoice_files = [f for f in os.listdir(invoices_dir) if f.endswith(".pdf")]
        
        for filename in invoice_files:
            invoice_path = os.path.join(invoices_dir, filename)
            invoice_text = parse_pdf(invoice_path)
            
            analysis = analyze_invoice_with_groq(policy_text, invoice_text)
            
            storage_status = "skipped"
            if analysis["status"] == "success":
                invoice_id = filename.split('.')[0]
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
        
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="The uploaded invoice file is not a valid zip file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        if temp_dir_to_clean and os.path.exists(temp_dir_to_clean):
            shutil.rmtree(temp_dir_to_clean)

# --- Part Two: RAG Chatbot Endpoint ---
# A simple in-memory store for chat history
chat_history = {}

def parse_metadata_from_query(query: str):
    """
    Parses a user query to extract metadata filters.
    e.g., "invoices for John Doe", "reimbursement status declined", "invoices from 2023-01-01"
    """
    filters = {}
    
    # Employee name (simple regex, can be improved)
    name_match = re.search(r"for\s+([A-Za-z\s]+)", query, re.IGNORECASE)
    if name_match:
        filters["employee_name"] = name_match.group(1).strip()
    
    # Reimbursement status
    status_match = re.search(r"(fully reimbursed|partially reimbursed|declined)", query, re.IGNORECASE)
    if status_match:
        filters["reimbursement_status"] = status_match.group(1).capitalize()
        
    # Date (simple regex for YYYY-MM-DD format)
    date_match = re.search(r"from\s+(\d{4}-\d{2}-\d{2})", query, re.IGNORECASE)
    if date_match:
        filters["date"] = date_match.group(1)
        
    # Remove the parsed metadata from the query to get the core search text
    query_text = query
    for val in filters.values():
        query_text = query_text.replace(val, "").replace("from", "").replace("for", "").strip()

    return query_text, filters

def rag_query(query_text: str, metadata_filters: dict, user_id: str):
    """
    Performs the RAG process:
    1. Embeds the user query.
    2. Performs a similarity search on ChromaDB with metadata filters.
    3. Augments the LLM prompt with retrieved documents.
    4. Calls the LLM to generate a response.
    """
    # 1. Embed the user query
    query_embedding = embedding_model.encode(query_text).tolist()
    
    # 2. Perform similarity search with metadata filtering
    results = invoice_collection.query(
        query_embeddings=[query_embedding],
        n_results=10,  # Retrieve top 10 results
        where=metadata_filters  # Apply metadata filtering
    )
    
    # 3. Augment the LLM prompt with retrieved documents
    retrieved_docs = results['documents'][0]
    
    if not retrieved_docs:
        context = "No relevant documents were found for this query."
    else:
        # Create a formatted context string from the retrieved documents and their metadata
        context_parts = []
        for i in range(len(retrieved_docs)):
            doc = retrieved_docs[i]
            metadata = results['metadatas'][0][i]
            context_parts.append(f"--- Document {i+1} ---\nEmployee: {metadata.get('employee_name', 'N/A')}\nInvoice ID: {metadata.get('invoice_id', 'N/A')}\nStatus: {metadata.get('reimbursement_status', 'N/A')}\nDate: {metadata.get('date', 'N/A')}\nContent:\n{doc}\n--- End Document ---\n")
        
        context = "\n".join(context_parts)

    # Check for previous conversation context
    previous_context = chat_history.get(user_id, "")
    
    # 4. Call the LLM to generate a response
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
        
        # Update chat history
        chat_history[user_id] = f"{previous_context}\nUser: {query_text}\nAI: {response_content}"
        
        return response_content
    except Exception as e:
        print(f"An error occurred with the Groq API during chat: {e}")
        return "Sorry, I am unable to process your request at this time."

@app.post("/chat/")
async def chat_endpoint(query: str, user_id: str = Form(...)):
    """
    RAG LLM Chatbot Endpoint.
    Accepts a user query, performs RAG, and returns a formatted response.
    """
    try:
        query_text, metadata_filters = parse_metadata_from_query(query)
        response = rag_query(query_text, metadata_filters, user_id)
        
        return JSONResponse(content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)