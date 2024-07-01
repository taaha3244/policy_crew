import streamlit as st
import requests

# URL of the FastAPI backend
FASTAPI_URL = "http://localhost:8000/process_query/"

# Initialize session state if it does not exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'query' not in st.session_state:
    st.session_state['query'] = ''

# Function to handle sending the query
def send_query():
    if st.session_state.query:
        try:
            # Send the query to the FastAPI backend
            response = requests.post(FASTAPI_URL, json={"query": st.session_state.query})
            response.raise_for_status()  # Check if the request was successful
            
            # Get the result from the response
            result = response.json().get("result", "No result found")
            
            # Update chat history
            st.session_state.chat_history.append(f"User: {st.session_state.query}")
            st.session_state.chat_history.append(f"Bot: {result}")
            
            # Clear the input box
            st.session_state.query = ''
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")

# Streamlit app title
st.title("Chat with Query Processing App")

# Display chat history
for message in st.session_state['chat_history']:
    st.write(message)

# Create a large spacer to push the input box to the bottom
st.write('<div style="height: 400px;"></div>', unsafe_allow_html=True)

# Input text box for the query at the bottom
st.text_area("Your message:", key="query", height=100, on_change=send_query)

