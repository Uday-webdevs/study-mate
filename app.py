"""
StudyMate - Modular AI Study Assistant
Main application file that orchestrates all components.
"""

import streamlit as st
from backend.config import config
from backend.document_processor import DocumentProcessor
from backend.rag_engine import StudyMateRAG
from frontend.ui_components import UIComponents
from frontend.chat_interface import ChatInterface


def main():
    # ---------------- Page Setup ----------------
    UIComponents.setup_page_config()
    UIComponents.apply_custom_css()

    # ---------------- Session State ----------------
    if "document_processor" not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()

    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = StudyMateRAG(vectorstore=None)

    if "chat_interface" not in st.session_state:
        st.session_state.chat_interface = ChatInterface(
            st.session_state.rag_engine
        )

    if "document_info" not in st.session_state:
        st.session_state.document_info = None

    # 🔥 NEW: async-like control flags
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    if "pending_thinking_id" not in st.session_state:
        st.session_state.pending_thinking_id = None

    # ---------------- Load Sample Data by Default ----------------
    if not st.session_state.document_info:
        result = st.session_state.document_processor.process_sample_data()
        if result["success"]:
            st.session_state.document_info = {"name": "Sample Data"}
            st.session_state.rag_engine.vectorstore = (
                st.session_state.document_processor.vectorstore
            )

    # ---------------- Config Validation ----------------
    config_validation = config.validate()

    # ---------------- Sidebar Upload ----------------
    document_info = UIComponents.render_sidebar_upload(
        st.session_state.document_processor,
        config_validation
    )

    if document_info:
        st.session_state.document_info = document_info

        # 🔥 IMPORTANT: do NOT recreate chat_interface
        st.session_state.rag_engine.vectorstore = (
            st.session_state.document_processor.vectorstore
        )

    # ---------------- Header ----------------
    UIComponents.render_header()

    # ---------------- Chat UI ----------------
    chat = st.session_state.chat_interface
    chat_history = chat.get_chat_history()

    user_query, is_suggestion = UIComponents.render_chat_interface(chat_history)

    # ======================================================
    # PHASE 1 — User submits a question (FAST)
    # ======================================================
    if user_query and st.session_state.pending_query is None:
        # 1️⃣ Add user message immediately
        chat.add_user_message(user_query)

        # 2️⃣ Add thinking placeholder
        thinking_id = chat.add_thinking_message()

        # 3️⃣ Store pending work
        st.session_state.pending_query = user_query
        st.session_state.pending_thinking_id = thinking_id
        st.session_state.is_suggestion = is_suggestion

        # 🔥 Force UI refresh NOW
        st.rerun()

    # ======================================================
    # PHASE 2 — Process RAG response (SLOW)
    # ======================================================
    if st.session_state.pending_query:
        response = chat.process_query(
            st.session_state.pending_query,
            verbose=False,
            max_words=30 if st.session_state.is_suggestion else None
        )

        chat.replace_thinking_with_response(
            st.session_state.pending_thinking_id,
            response
        )

        # Clear pending state
        st.session_state.pending_query = None
        st.session_state.pending_thinking_id = None
        st.session_state.is_suggestion = False

        # Final UI update
        st.rerun()

    # ---------------- Footer ----------------
    UIComponents.render_footer()


if __name__ == "__main__":
    main()
