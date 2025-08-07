# **Invoice Reimbursement RAG System**

#### **Project Overview**

This project is a FastAPI application that serves as an automated invoice reimbursement system. It uses a Large Language Model (LLM) from Groq to analyze invoices against a provided policy. The system is enhanced with a Retrieval-Augmented Generation (RAG) architecture, using ChromaDB as a vector store to enable a chatbot to answer questions about past invoice analyses.

The main components are:

  - **Invoice Analysis API**: An endpoint that accepts a reimbursement policy (PDF) and a zip file of invoices. It uses the LLM to determine the reimbursement status and reason for each invoice.
  - **Vector Store Integration**: The analysis results are converted into embeddings and stored in a persistent ChromaDB database, along with relevant metadata.
  - **RAG Chatbot API**: A chatbot endpoint that allows users to query the stored invoice data using natural language. It retrieves relevant documents from the vector store to provide accurate, context-aware answers.

#### **Installation Instructions**

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Ramya041/Invoice-Reimbursement-System.git
    cd Invoice-Reimbursement-System
    ```
2.  **Set up a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up your Groq API key**:
    Create a `.env` file in the project's root directory and add your key:
    ```env
    GROQ_API_KEY="your_groq_api_key_here"
    ```
5.  **Run the FastAPI application**:
    ```bash
    uvicorn app.main:app --reload
    ```
6.  **Run the Streamlit UI** (in a separate terminal):
    ```bash
    streamlit run app/ui.py
    ```

#### **Usage Guide**

  - **Invoice Analysis**: Navigate to the Streamlit UI at `http://localhost:8501`. Use the "1. Invoice Analysis" section to upload a policy PDF and a zip file of invoice PDFs. Enter an employee name and click "Analyze Invoices" to process the files and store the results.
  - **RAG Chatbot**: After analyzing invoices, use the "2. RAG Chatbot" section of the Streamlit UI. Enter a unique User ID and type your query (e.g., "What was the reason for declining the travel invoice?"). The chatbot will use the stored data to provide a response.

#### **Technical Details**

  - **Framework**: FastAPI for building the API endpoints and Streamlit for the testing UI.
  - **LLM**: Groq's `llama3-8b-8192` model is used for its high speed in analyzing policy and invoice data.
  - **Embedding Model**: The `sentence-transformers` library with the `all-MiniLM-L6-v2` model is used to create vector embeddings of text data for the vector store.
  - **Vector Store**: ChromaDB is a lightweight, persistent vector database used to store the embeddings and metadata, enabling efficient semantic search.
  - **File Handling**: The application handles PDF and ZIP file uploads, extracts text using `PyPDF2`, and manages temporary directories.

#### **Prompt Design**

  - **Analysis Prompt**: The prompt for invoice analysis is a template that provides the LLM with the reimbursement policy and invoice details. It instructs the LLM to respond with a structured format: `Status: <Status>` and `Reason: <Reason>` to facilitate programmatic parsing.
  - **Chatbot Prompt**: The chatbot's prompt is designed for a RAG system. It includes the user's query, a history of the conversation, and a context window populated with relevant documents retrieved from ChromaDB. This forces the LLM to ground its answers in the provided facts, preventing inaccuracies.

#### **Challenges & Solutions**

  - **Metadata Filtering**: A key challenge was efficiently querying the vector store based on non-text criteria (e.g., employee name or status). The `parse_metadata_from_query` function was created to extract these filters from the user's natural language query, which are then passed to ChromaDB's `where` parameter for a targeted search.
  - **File Handling**: Handling file uploads, especially ZIP files, required robust logic to ensure the files were correctly saved and extracted. The `save_uploaded_files` function was refactored to directly handle the file-like object from the `UploadFile` instance, making it more reliable.
  - **Robustness**: The application includes comprehensive logging using Python's `logging` module. This helps monitor the application's behavior and diagnose issues related to file parsing, API calls, or database operations.
