import streamlit as st
import requests
import zipfile
import os

st.title("Invoice Reimbursement RAG System")

# --- Part 1: Analyze Invoices ---
st.header("1. Invoice Analysis")
with st.form("analysis_form"):
    employee_name = st.text_input("Employee Name")
    policy_file = st.file_uploader("Upload Reimbursement Policy (PDF)", type=["pdf"])
    invoice_zip = st.file_uploader("Upload Invoices (ZIP file with PDFs)", type=["zip"])
    submit_analysis = st.form_submit_button("Analyze Invoices")

    if submit_analysis:
        if not all([employee_name, policy_file, invoice_zip]):
            st.error("Please fill in all fields and upload both files.")
        else:
            files = {
                "policy_file": (policy_file.name, policy_file.getvalue(), "application/pdf"),
                "invoice_zip": (invoice_zip.name, invoice_zip.getvalue(), "application/zip")
            }
            data = {"employee_name": employee_name}

            with st.spinner("Analyzing invoices..."):
                try:
                    response = requests.post("http://127.0.0.1:8000/analyze-invoices/", files=files, data=data)
                    if response.status_code == 200:
                        st.success("Analysis complete!")
                        st.json(response.json())
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Connection Error: Is the FastAPI server running?")

# --- Part 2: RAG Chatbot ---
st.header("2. RAG Chatbot")
user_id = st.text_input("Enter a User ID (for chat history)")
query = st.text_area("Ask a question about the invoices...")
submit_chat = st.button("Send")

if submit_chat and user_id and query:
    if not user_id or not query:
        st.error("Please enter a User ID and a query.")
    else:
        data = {"query": query, "user_id": user_id}
        with st.spinner("Thinking..."):
            try:
                response = requests.post("http://127.0.0.1:8000/chat/", data=data)
                if response.status_code == 200:
                    st.success("Response received!")
                    response_json = response.json()
                    st.markdown("---")
                    st.markdown(response_json.get("response", "No response received."))
                    st.markdown("---")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Is the FastAPI server running?")