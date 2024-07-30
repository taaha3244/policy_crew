import streamlit as st
from dotenv import load_dotenv
import os
import sys

# Directly set the project root directory
project_root = "/policy_crew"
# Ensure the project root is at the top of sys.path
sys.path.insert(0, project_root)



# Import custom modules
from app.frontend.state_management import initialize_state
from app.frontend.pdf_management import render_pdf_management
from app.frontend.chat_interface import render_chat_interface

# Load environment variables
load_dotenv()

# URL of the FastAPI backend
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://backend:8000")

# Initialize session state
initialize_state()

# Sidebar for PDF Management
render_pdf_management()

# Main page for Chat Functionality
st.title("Multi Agent Policy Extractor")

# Create tabs for different pages
tab1, tab2 = st.tabs(["Crew AI", "Langraph Agent/Graph RAG"])

# Crew AI tab
with tab1:
    render_chat_interface("query_crew_ai", f"{FASTAPI_URL}/process_query/")

# Langraph Agent/Graph RAG tab
with tab2:
    render_chat_interface("query_langraph", f"{FASTAPI_URL}/process_query_langraph/")
