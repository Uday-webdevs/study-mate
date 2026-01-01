"""
StudyMate UI Components – Modern Chat UI
"""

import streamlit as st


class UIComponents:

    # ---------------- Page Setup ----------------
    @staticmethod
    def setup_page_config():
        st.set_page_config(
            page_title="StudyMate",
            page_icon="📚",
            layout="wide"
        )

    @staticmethod
    def apply_custom_css():
        st.markdown("""
        <style>
        body {
            background-color: #f7f9fc;
        }

        .app-container {
            max-width: 1100px;
            margin: auto;
            width: 100%;
        }

        /* ===== GLOBAL BUTTON FIX ===== */
        button, 
        div[data-testid="stButton"] button {
            white-space: nowrap !important;
        }

        /* ===== STUDY BANNER ===== */
        .study-banner {
            background: linear-gradient(135deg, #1e88e5, #42a5f5);
            padding: 1rem 2rem;
            border-radius: 18px;
            color: white;
            box-shadow: 0 10px 28px rgba(30,136,229,0.35);
            margin-bottom: 2rem;
        }

        .study-banner h1 {
            margin: 0;
            font-size: 2.2rem;
            font-weight: 700;
        }

        .study-banner p {
            font-size: 1.05rem;
            opacity: 0.95;
            max-width: 700px;
        }

        /* ===== REMOVE INPUT OUTLINE ===== */
        div[data-baseweb="input"] > div:focus-within {
            outline: none !important;
            box-shadow: none !important;
            border-color: #d6dbe6 !important;
        }

        input:focus,
        input:focus-visible {
            outline: none !important;
            box-shadow: none !important;
        }

        /* ===== CHAT AREA ===== */
        .chat-wrapper {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            margin-top: 1.5rem;
        }

        /* ===== MESSAGE ROWS ===== */
        .chat-row {
            display: flex;
            align-items: flex-end;
            gap: 0.6rem;
        }

        .chat-row.user {
            justify-content: flex-end;
            margin: 1rem 0;
        }

        .chat-row.bot {
            justify-content: flex-start;
        }

        /* ===== ICONS ===== */
        .user-chat-icon {
            font-size: 1.8rem;
            line-height: 1;
        }

        .bot-icon {
            font-size: 2.5rem;
            line-height: 1;
        }

        /* ===== USER BUBBLE ===== */
        .user-bubble {
            background: linear-gradient(135deg, #1e88e5, #42a5f5);
            color: white;
            padding: 0.75rem 1rem;
            border-radius: 18px 18px 4px 18px;
            max-width: 70%;
            box-shadow: 0 6px 14px rgba(0,0,0,0.12);
            word-wrap: break-word;
        }

        /* ===== BOT BUBBLE ===== */
        .assistant-bubble {
            background: white;
            padding: 1rem;
            border-radius: 18px 18px 18px 4px;
            max-width: 70%;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border: 1px solid #eef1f6;
        }

        .assistant-content {
            margin-top: 0.25rem;
        }

        .thinking {
            font-style: italic;
            color: #888;
        }

        .footer {
            text-align: center;
            color: #999;
            font-size: 0.85rem;
        }
        </style>
        """, unsafe_allow_html=True)

    # ---------------- Header ----------------
    @staticmethod
    def render_header():
        st.markdown("""
        <div class="app-container">
            <div class="study-banner">
                <h1>📖 StudyMate</h1>
                <p>
                    Your AI-powered study companion for understanding documents effortlessly.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- Sidebar ----------------
    @staticmethod
    def render_sidebar_upload(document_processor, config_validation):
        with st.sidebar:
            st.markdown("### 📄 Study Material")

            if not config_validation["valid"]:
                st.error("Invalid configuration")
                return None

            uploaded_file = st.file_uploader(
                "Upload a PDF",
                type=["pdf"],
                help="Upload study material in PDF format"
            )

            upload_clicked = st.button("📤 Upload & Index")
            sample_clicked = st.button("📚 Load Sample Data")

            if upload_clicked and uploaded_file:
                with st.spinner("Processing document..."):
                    result = document_processor.process_uploaded_file(uploaded_file)
                if result["success"]:
                    st.success("Document indexed successfully!")
                    return {"name": uploaded_file.name}

            if sample_clicked:
                with st.spinner("Loading sample data..."):
                    result = document_processor.process_sample_data()
                if result["success"]:
                    st.success("Sample data ready!")
                    return {"name": "Sample Data"}

        return None

    # ---------------- Chat UI ----------------
    @staticmethod
    def render_chat_interface(chat_history):
        st.markdown("<div class='app-container'>", unsafe_allow_html=True)

        st.markdown("### 💬 Ask a question")

        with st.form("chat_form", clear_on_submit=True):
            user_query = st.text_input(
                "",
                placeholder="Ask something like: What is a noun?"
            )
            submit = st.form_submit_button("🚀 Send")

        if not chat_history:
            st.markdown("**💡 Try one of these:**")
            suggestions = [
                "What is a noun?",
                "Explain verbs in one sentence",
                "How do adjectives work?",
                "What are pronouns?"
            ]
            cols = st.columns(len(suggestions))
            for i, q in enumerate(suggestions):
                with cols[i]:
                    if st.button(q, key=f"suggest_{i}"):
                        return q, True

        st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)

        if chat_history:
            for msg in chat_history[::-1]:
                if msg["role"] == "user":
                    st.markdown(
                        f"""
                        <div class="chat-row user">
                            <div class="user-bubble">{msg['content']}</div>
                            <div class="user-chat-icon">🧑</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    content = msg["content"]
                    if msg.get("is_thinking"):
                        content = f"<span class='thinking'>{content}</span>"

                    st.markdown(
                        f"""
                        <div class="chat-row bot">
                            <div class="bot-icon">🤖</div>
                            <div class="assistant-bubble">
                                <div class="assistant-content">{content}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.markdown("</div></div>", unsafe_allow_html=True)

        return (user_query, False) if submit and user_query.strip() else (None, False)

    # ---------------- Footer ----------------
    @staticmethod
    def render_footer():
        st.markdown("---")
        st.markdown(
            "<div class='footer'>StudyMate v2.0 • Built with Streamlit & LangChain</div>",
            unsafe_allow_html=True
        )
