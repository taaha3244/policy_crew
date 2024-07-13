import streamlit as st
from app.frontend.api_requests import send_query

def render_chat_interface(query_key, endpoint):
    # Display chat history
    for message in st.session_state["chat_history"]:
        st.write(message)
    
    # Input text box for the query
    st.text_area("Your message:", key=query_key, height=100, on_change=lambda: send_query(query_key, endpoint))
