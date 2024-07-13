import streamlit as st

def initialize_state():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "query_crew_ai" not in st.session_state:
        st.session_state["query_crew_ai"] = ""

    if "query_langraph" not in st.session_state:
        st.session_state["query_langraph"] = ""
