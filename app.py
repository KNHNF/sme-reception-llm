"""
Streamlit frontend for the SME voice assistant.

Run the backend first:
    python backend.py

Then in a second terminal:
    pip install streamlit requests
    streamlit run app.py

Opens at http://localhost:8501
"""

import streamlit as st
import requests

BACKEND_URL = "http://localhost:5005/turn"

st.set_page_config(
    page_title="SME Reception Assistant",
    page_icon="📞",
    layout="centered",
)

st.title("📞 SME Reception Assistant")
st.caption("UWE Bristol MSc Data Science | Group 6 IGP Demo")
st.write("Type an appointment request below. The assistant handles booking, cancellations, and availability checks.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-demo-001"

# Example utterances sidebar
with st.sidebar:
    st.header("Try these examples")
    examples = [
        "I'd like to book a consultation for next Monday at 2pm",
        "Do you have any slots available on Thursday?",
        "I need to cancel my appointment on Wednesday at 10am",
        "Book me in for a follow-up appointment",
        "What are your opening hours?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_input = ex

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = "streamlit-demo-001"
        st.rerun()

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("action"):
            with st.expander("Structured action (JSON)"):
                st.json(msg["action"])

# Handle example button click
pending = st.session_state.pop("pending_input", None)

# Chat input
user_input = st.chat_input("e.g. I'd like to book a consultation for Monday at 2pm") or pending

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call backend
    try:
        response = requests.post(
            BACKEND_URL,
            json={"utterance": user_input, "session_id": st.session_state.session_id},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        spoken = data.get("spoken", "Sorry, I could not process that request.")
        action = data.get("action")

        with st.chat_message("assistant"):
            st.write(spoken)
            if action:
                with st.expander("Structured action (JSON)"):
                    st.json(action)

        st.session_state.messages.append({
            "role": "assistant",
            "content": spoken,
            "action": action,
        })

    except requests.exceptions.ConnectionError:
        with st.chat_message("assistant"):
            st.error("Cannot reach the backend. Make sure `python backend.py` is running on port 5005.")
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error: {e}")

    st.rerun()
