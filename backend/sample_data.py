"""
StudyMate Document Processor
Handles PDF document loading, chunking, and vector store management.
"""

import os
import tempfile
from typing import Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from backend.config import config
from sample_data import SAMPLE_DATA_CONTENT


# ============================================================
# DOCUMENT PROCESSOR
# ============================================================

class DocumentProcessor:
    """
    Handles document processing for StudyMate.
    Manages PDF loading, text chunking, and vector store operations.
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

        self.vectorstore = None
        self.documents = []
        self.document_metadata = {}

        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY
        )

    # ============================================================
    # FILE VALIDATION
    # ============================================================

    def validate_file(self, uploaded_file) -> Dict[str, Any]:
        if not uploaded_file:
            return {"valid": False, "error": "No file provided", "size_mb": 0}

        file_name = uploaded_file.name.lower()
        if not any(file_name.endswith(ext) for ext in config.ALLOWED_EXTENSIONS):
            return {
                "valid": False,
                "error": f"File type not allowed. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}",
                "size_mb": 0
            }

        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > config.MAX_FILE_SIZE_MB:
            return {
                "valid": False,
                "error": f"File too large. Maximum size: {config.MAX_FILE_SIZE_MB}MB",
                "size_mb": file_size_mb
            }

        return {"valid": True, "error": "", "size_mb": file_size_mb}

    # ============================================================
    # MAIN UPLOAD PIPELINE
    # ============================================================

    def process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        validation = self.validate_file(uploaded_file)
        if not validation["valid"]:
            return {"success": False, "error": validation["error"], "chunks": 0}

        collection_name = self._create_collection_name(uploaded_file.name)

        # ✅ SAFE CACHE CHECK
        if self._load_cached_vectorstore(collection_name):
            info = self.get_document_info()
            return {
                "success": True,
                "chunks": info.get("total_chunks", 0),
                "collection_name": collection_name,
                "cached": True
            }

        # Save temp PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            result = self.load_and_chunk_pdf(tmp_path, uploaded_file.name)
            if not result["success"]:
                return result

            vs_result = self.create_vectorstore(collection_name)
            return {
                "success": vs_result["success"],
                "chunks": result["chunks"],
                "collection_name": collection_name,
                "cached": False,
                "error": vs_result.get("error", "")
            }
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    # ============================================================
    # PDF LOADING & CHUNKING (UNCHANGED LOGIC)
    # ============================================================

    def load_and_chunk_pdf(self, pdf_path: str, original_filename: str) -> Dict[str, Any]:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        all_chunks = []
        for i, page in enumerate(pages):
            text = page.page_content.strip()
            if len(text) < 50:
                continue

            chunks = self.text_splitter.split_text(text)
            for j, chunk in enumerate(chunks):
                all_chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": original_filename,
                        "page": i + 1,
                        "chunk_id": j
                    }
                ))

        if not all_chunks:
            return {"success": False, "error": "No usable content", "chunks": 0}

        self.documents = all_chunks
        self.document_metadata = {
            "filename": original_filename,
            "total_pages": len(pages),
            "total_chunks": len(all_chunks)
        }

        return {"success": True, "chunks": len(all_chunks)}

    # ============================================================
    # VECTORSTORE CREATION
    # ============================================================

    def create_vectorstore(self, collection_name: str) -> Dict[str, Any]:
        persist_dir = os.path.join(config.VECTOR_STORE_PATH, collection_name)

        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            persist_directory=persist_dir,
            collection_name=collection_name
        )
        self.vectorstore.persist()

        return {"success": True}

    # ============================================================
    # 🔥 FIX: SAFE CACHE LOADING (CRITICAL FIX)
    # ============================================================

    def _load_cached_vectorstore(self, collection_name: str) -> bool:
        persist_dir = os.path.join(config.VECTOR_STORE_PATH, collection_name)

        if not os.path.exists(persist_dir):
            return False

        try:
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )

            # 🔥 CRITICAL: semantic validation
            docs = vectorstore.similarity_search("test", k=1)
            if not docs:
                return False

            self.vectorstore = vectorstore
            self.document_metadata = {
                "filename": collection_name.replace("_index", ""),
                "total_chunks": vectorstore._collection.count(),
                "source": "cached"
            }
            return True

        except Exception as e:
            print("❌ Cache load failed:", e)
            return False

    # ============================================================
    # SAMPLE DATA (FULLY RESTORED & FIXED)
    # ============================================================

    def process_sample_data(self) -> Dict[str, Any]:
        collection_name = "sample_data_index"

        if self._load_cached_vectorstore(collection_name):
            info = self.get_document_info()
            return {
                "success": True,
                "chunks": info.get("total_chunks", 0),
                "collection_name": collection_name
            }

        docs = [Document(
            page_content=SAMPLE_DATA_CONTENT.strip(),
            metadata={"source": "sample_data", "page": 1}
        )]

        self.documents = self.text_splitter.split_documents(docs)
        self.document_metadata = {
            "filename": "Sample Grammar Data",
            "total_chunks": len(self.documents),
            "total_pages": 1
        }

        self.create_vectorstore(collection_name)
        return {
            "success": True,
            "chunks": len(self.documents),
            "collection_name": collection_name
        }

    # ============================================================
    # HELPERS
    # ============================================================

    def get_document_info(self) -> Dict[str, Any]:
        return self.document_metadata if self.document_metadata else {"loaded": False}

    def clear_documents(self):
        self.documents = []
        self.document_metadata = {}
        self.vectorstore = None

    def _create_collection_name(self, filename: str) -> str:
        name = Path(filename).stem
        name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        return f"{name}_index"
