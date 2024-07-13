import requests
import streamlit as st
import os
import sys



from custom_logger import logger
from custom_exceptions import CustomException

def send_query(query_key, endpoint):
    if st.session_state[query_key]:
        try:
            response = requests.post(
                endpoint, json={"query": st.session_state[query_key]}
            )
            response.raise_for_status()

            result = response.json().get("result", "No result found")

            st.session_state.chat_history.append(f"User: {st.session_state[query_key]}")
            st.session_state.chat_history.append(f"Bot: {result}")

            st.session_state[query_key] = ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending query: {e}")
            st.error(f"Error: {e}")
        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
            st.error(f"CustomException: {str(ce)}")
        except Exception as e:
            logger.exception("Unexpected error occurred while sending query")
            st.error("Internal server error")
