# app.py

import streamlit as st
import os
import pandas as pd
from mental_health_support_rag import get_rag_chain, load_documents, get_vector_store, CSV_DATA_PATH

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="🧠 Mental Health Chatbot",
    page_icon="💬",
    layout="centered"
)

# --- Initialize Session State ---
# This ensures that the RAG chain and other components are initialized only once
# when the Streamlit app starts, rather than on every rerun.
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'db_ready' not in st.session_state:
    st.session_state.db_ready = False

# --- Custom Styling (Optional but good for aesthetics) ---
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .chat-message {
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e6f7ff; /* Light blue for user messages */
        justify-content: flex-end;
    }
    .chat-message.bot {
        background-color: #f0f0f0; /* Light gray for bot messages */
        justify-content: flex-start;
    }
    .chat-icon {
        font-size: 24px;
        margin: 0 10px;
    }
    .user-icon {
        color: #007bff;
    }
    .bot-icon {
        color: #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Title and Description ---
st.title("🧠 Mental Health RAG Chatbot")
st.markdown(
    "Hello! I'm here to provide information and support regarding mental health topics. "
    "Please remember that I'm an AI and not a substitute for professional medical advice."
)

# --- Function to Initialize RAG Components ---
@st.cache_resource
def initialize_rag():
    """
    Initializes the document loading, vector store, and RAG chain.
    Uses st.cache_resource to ensure these heavy operations run only once.
    """
    with st.spinner("🚀 Setting up the chatbot... This might take a moment."):
        # Load documents from all sources (PDF, TXT, CSV)
        documents = load_documents()
        if not documents:
            st.error("No documents found or loaded. Please ensure you have files in the 'docs' directory and/or 'mental_health_qa.csv'.")
            return None, False

        # Create or load the ChromaDB vector store
        vector_store = get_vector_store(documents)
        if vector_store is None:
            st.error("Failed to create or load the vector store.")
            return None, False

        # Create the RAG chain
        rag_chain = get_rag_chain(vector_store)
        if rag_chain is None:
            st.error("Failed to create the RAG chain. Check your Hugging Face API token and model.")
            return None, False

        return rag_chain, True

# --- Initialize RAG on app start ---
if not st.session_state.db_ready:
    st.session_state.rag_chain, st.session_state.db_ready = initialize_rag()
    if st.session_state.db_ready:
        st.success("Chatbot is ready! Ask me anything about mental health.")
    else:
        st.error("Chatbot setup failed. Please check the console for errors and ensure all dependencies are installed and API keys are set.")

# --- Chat Interface ---
if st.session_state.db_ready:
    # Initialize chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me a question about mental health..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                # Invoke the RAG chain with the user's question
                    response = st.session_state.rag_chain.invoke({"input": prompt})

                    # Safely get full answer
                    raw_answer = response.get("answer") or response.get("output") or str(response)

                    # ✅ Extract only the last "Answer:" portion
                    if "Answer:" in raw_answer:
                        final_answer = raw_answer.split("Answer:")[-1].strip()
                    else:
                        final_answer = raw_answer.strip()

                except Exception as e:
                    final_answer = f"An error occurred while processing your request:\n\n{e}"
                    st.error(final_answer)

            # Show only the final answer
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
else:
    st.info("Chatbot is not ready. Please ensure all setup steps are complete and no errors occurred.")
