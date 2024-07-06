import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import requests
import sys


# Add the path to load_docs.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from backend.load_docs import PDFProcessor
from backend.custom_logger import logger
from backend.custom_exceptions import CustomException



# Load environment variables
load_dotenv()

# URL of the FastAPI backend
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://backend:8000/process_query/")

# Initialize session state if it does not exist
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "query" not in st.session_state:
    st.session_state["query"] = ""

# Initialize PDFProcessor
openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
pdf_processor = PDFProcessor(openai_api_key, qdrant_url, qdrant_api_key)

# Function to handle sending the query
def send_query():
    if st.session_state.query:
        try:
            response = requests.post(
                FASTAPI_URL, json={"query": st.session_state.query}
            )
            response.raise_for_status()

            result = response.json().get("result", "No result found")

            st.session_state.chat_history.append(f"User: {st.session_state.query}")
            st.session_state.chat_history.append(f"Bot: {result}")

            st.session_state.query = ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending query: {e}")
            st.error(f"Error: {e}")
        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
            st.error(f"CustomException: {str(ce)}")
        except Exception as e:
            logger.exception("Unexpected error occurred while sending query")
            st.error("Internal server error")

# Save uploaded files
def save_uploaded_files(uploaded_files):
    files_info = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            files_info.append((tmpfile.name, uploaded_file.name))
    return files_info

# Sidebar for PDF Management
st.sidebar.title("PDF Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
)

url_input = st.sidebar.text_input("Enter PDF URL (separate multiple URLs with commas):")
if st.sidebar.button("Load PDF from URL"):
    urls = [url.strip() for url in url_input.split(",") if url.strip()]
    for url in urls:
        try:
            data = pdf_processor.load_from_url(url)
            if data:
                pdf_processor.split_and_store(data)
                result = pdf_processor.create_rag_system()
                st.sidebar.success(f"Loaded and processed URL: {url}")
                st.sidebar.write(result)
            else:
                st.sidebar.error(f"Failed to load data from URL: {url}")
        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
            st.sidebar.error(str(ce))
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            st.sidebar.error("Internal server error")

if uploaded_files:
    if st.sidebar.button("Process Uploaded PDFs"):
        files_info = save_uploaded_files(uploaded_files)
        for temp_path, original_name in files_info:
            try:
                data = pdf_processor.load_from_file(temp_path)
                if data:
                    pdf_processor.split_and_store(data)
                    result = pdf_processor.create_rag_system()
                    st.sidebar.success(f"Processed file: {original_name}")
                    st.sidebar.write(result)
                else:
                    st.sidebar.error(f"Failed to load data from file: {original_name}")
            except CustomException as ce:
                logger.error(f"CustomException: {str(ce)}")
                st.sidebar.error(str(ce))
            except Exception as e:
                logger.exception(f"Unexpected error: {str(e)}")
                st.sidebar.error("Internal server error")

# Main page for Chat Functionality
st.title("Chat with Query Processing and PDF Upload App")

# Display chat history
for message in st.session_state["chat_history"]:
    st.write(message)

# Input text box for the query
st.text_area("Your message:", key="query", height=100, on_change=send_query)
