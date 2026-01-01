"""
StudyMate Chat Interface
Handles chat functionality and user interactions.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from backend.rag_engine import StudyMateRAG, RAGResponse
import uuid


class ChatInterface:
    def __init__(self, rag_engine: StudyMateRAG):
        self.rag_engine = rag_engine
        self.chat_history: List[Dict[str, Any]] = []

    # ---------------- RAG call ----------------
    def process_query(
        self,
        user_query: str,
        verbose: bool = False,
        max_words: Optional[int] = None
    ) -> RAGResponse:
        response = self.rag_engine.query(user_query, verbose=verbose)

        if max_words:
            response.answer = " ".join(response.answer.split()[:max_words])

        return response

    # ---------------- History mutation ----------------
    def add_user_message(self, content: str):
        self.chat_history.append({
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": content,
            "timestamp": self._ts()
        })

    def add_thinking_message(self) -> str:
        thinking_id = str(uuid.uuid4())
        self.chat_history.append({
            "id": thinking_id,
            "role": "assistant",
            "content": "⏳ Thinking...",
            "is_thinking": True,
            "timestamp": self._ts()
        })
        return thinking_id

    def replace_thinking_with_response(
        self,
        thinking_id: str,
        response: RAGResponse
    ):
        for msg in self.chat_history:
            if msg.get("id") == thinking_id:
                msg.update({
                    "content": response.answer,
                    "is_thinking": False,
                    "quality": response.context_quality.value,
                    "confidence": response.confidence,
                    "specificity": response.specificity_score,
                    "relevance": response.completeness_score,
                })
                break

    # ---------------- Accessors ----------------
    def get_chat_history(self) -> List[Dict[str, Any]]:
        return list(self.chat_history)

    # ---------------- Helpers ----------------
    def _ts(self) -> str:
        return datetime.now().isoformat()
