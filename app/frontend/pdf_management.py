import os
import tempfile
import streamlit as st
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.frontend.load_docs import PDFProcessor
from custom_logger import logger
from custom_exceptions import CustomException
# Ensure the backend module is found

# Initialize PDFProcessor
openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
pdf_processor = PDFProcessor(openai_api_key, qdrant_url, qdrant_api_key)

def save_uploaded_files(uploaded_files):
    files_info = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            files_info.append((tmpfile.name, uploaded_file.name))
    return files_info

def render_pdf_management():
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
