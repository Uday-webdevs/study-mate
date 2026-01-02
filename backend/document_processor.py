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

from backend.config import config

# ==========================================================
# Sample data for testing - English Grammar Concepts
# ==========================================================
SAMPLE_DATA_CONTENT = """
English Grammar Fundamentals: Parts of Speech and Basic Concepts

English grammar is the set of rules that govern how words are used in the English language. Understanding these basic concepts is essential for effective communication and writing.

NOUNS
A noun is a word that represents a person, place, thing, or idea. Nouns can be classified into several types:

1. Common Nouns: General names for people, places, or things (e.g., book, city, teacher)
2. Proper Nouns: Specific names that are capitalized (e.g., London, Shakespeare, Microsoft)
3. Abstract Nouns: Names of ideas, qualities, or states (e.g., love, happiness, freedom)
4. Collective Nouns: Names for groups of people or things (e.g., team, family, herd)

VERBS
A verb is a word that expresses an action, occurrence, or state of being. Verbs are the most important part of a sentence and show what the subject is doing.

1. Action Verbs: Express physical or mental actions (e.g., run, think, eat, sleep)
2. Linking Verbs: Connect the subject to a noun or adjective (e.g., is, are, was, were, become)
3. Helping Verbs: Help the main verb express action or state (e.g., have, has, had, will, would)

PRONOUNS
A pronoun is a word that takes the place of a noun to avoid repetition. Pronouns must agree with their antecedents in number, gender, and person.

1. Personal Pronouns: Refer to specific persons or things
   - First person: I, me, we, us
   - Second person: you
   - Third person: he, him, she, her, it, they, them

2. Possessive Pronouns: Show ownership (e.g., my, your, his, her, its, our, their)
3. Demonstrative Pronouns: Point to specific things (e.g., this, that, these, those)
4. Relative Pronouns: Introduce relative clauses (e.g., who, whom, whose, which, that)

ADJECTIVES
An adjective is a word that describes or modifies a noun or pronoun. Adjectives provide more information about the qualities or characteristics of nouns.

1. Descriptive Adjectives: Describe qualities (e.g., beautiful, tall, intelligent, red)
2. Quantitative Adjectives: Indicate quantity (e.g., some, many, few, several)
3. Demonstrative Adjectives: Point out specific nouns (e.g., this, that, these, those)
4. Possessive Adjectives: Show ownership (e.g., my, your, his, her, its, our, their)

ADVERBS
An adverb is a word that modifies a verb, adjective, or another adverb. Adverbs often answer questions like how, when, where, why, or to what extent.

1. Adverbs of Manner: Tell how something is done (e.g., quickly, slowly, carefully)
2. Adverbs of Time: Tell when something happens (e.g., now, then, soon, yesterday)
3. Adverbs of Place: Tell where something happens (e.g., here, there, everywhere)
4. Adverbs of Degree: Tell to what extent (e.g., very, quite, almost, too)

PREPOSITIONS
A preposition is a word that shows the relationship between a noun or pronoun and other words in a sentence. Prepositions often indicate location, time, or direction.

Common prepositions: in, on, at, to, from, with, by, for, about, under, over, between, among, through, during, before, after, since, until.

CONJUNCTIONS
A conjunction is a word that connects words, phrases, or clauses. Conjunctions help create complex sentences and show relationships between ideas.

1. Coordinating Conjunctions: Connect equal parts (e.g., and, but, or, nor, for, so, yet)
2. Subordinating Conjunctions: Connect dependent clauses (e.g., although, because, since, unless, while, when)
3. Correlative Conjunctions: Work in pairs (e.g., either...or, neither...nor, both...and)

INTERJECTIONS
An interjection is a word or phrase that expresses strong emotion or surprise. Interjections are often followed by exclamation marks.

Common interjections: oh, wow, hey, ouch, ah, bravo, congratulations, good grief.

ARTICLES
Articles are special adjectives that introduce nouns. There are three articles in English:

1. The Definite Article: "the" - refers to specific nouns
2. Indefinite Articles: "a" and "an" - refer to non-specific nouns

SENTENCE STRUCTURE
A complete sentence must contain at least one independent clause with a subject and a verb. Sentences can be classified as:

1. Simple Sentences: One independent clause (e.g., "I study English.")
2. Compound Sentences: Two or more independent clauses joined by conjunctions (e.g., "I study English, and I practice every day.")
3. Complex Sentences: One independent clause and one or more dependent clauses (e.g., "Although I study English, I still make mistakes.")
4. Compound-Complex Sentences: Multiple independent and dependent clauses

PUNCTUATION
Proper punctuation helps clarify meaning and makes writing easier to read:

1. Period (.): Ends a sentence
2. Comma (,): Separates items in a list or clauses
3. Question Mark (?): Ends a question
4. Exclamation Mark (!): Shows strong emotion
5. Colon (:): Introduces a list or explanation
6. Semicolon (;): Connects related independent clauses
7. Apostrophe ('): Shows possession or contractions
8. Quotation Marks (" "): Enclose direct speech or quotations
9. Parentheses ( ): Contain additional information
10. Hyphen (-): Joins compound words or shows word breaks

Understanding these fundamental grammar concepts will help you communicate more effectively in English. Practice regularly and pay attention to how these parts of speech work together in sentences.
"""

class DocumentProcessor:
    """
    Handles document processing for StudyMate.
    Manages PDF loading, text chunking, and vector store operations.
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            # ❗ FIX: removed "" separator (critical)
            separators=["\n\n", "\n", ". ", " "],
            length_function=len
        )

        self.vectorstore = None
        self.documents = []
        self.document_metadata = {}

    # ======================================================
    # VALIDATION
    # ======================================================
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

    # ======================================================
    # UPLOADED FILE PROCESSING
    # ======================================================
    def process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        validation = self.validate_file(uploaded_file)
        if not validation["valid"]:
            return {"success": False, "chunks": 0, "error": validation["error"], "collection_name": ""}

        collection_name = self._create_collection_name(uploaded_file.name)

        if self.is_collection_cached(collection_name):
            if self.load_existing_vectorstore(collection_name):
                info = self.get_document_info()
                return {
                    "success": True,
                    "chunks": info.get("total_chunks", 0),
                    "collection_name": collection_name,
                    "pages": info.get("total_pages", 0),
                    "file_size_mb": validation["size_mb"],
                    "cached": True,
                    "error": ""
                }

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            result = self.load_and_chunk_pdf(tmp_path, uploaded_file.name)
            if not result["success"]:
                return result

            vectorstore_result = self.create_vectorstore(collection_name)
            if not vectorstore_result["success"]:
                return vectorstore_result

            return {
                "success": True,
                "chunks": result["chunks"],
                "pages": result["pages"],
                "collection_name": collection_name,
                "file_size_mb": validation["size_mb"],
                "cached": False,
                "error": ""
            }

        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ======================================================
    # 🔥 CORE FIX: PDF LOADING & CHUNKING
    # ======================================================
    def load_and_chunk_pdf(self, pdf_path: str, original_filename: str) -> Dict[str, Any]:
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            if not pages:
                return {"success": False, "chunks": 0, "pages": 0, "error": "No content found in PDF"}

            cleaned_docs = []
            for page in pages:
                text = " ".join(page.page_content.split())
                if len(text) >= 50:
                    cleaned_docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": original_filename,
                                "page": page.metadata.get("page", 1)
                            }
                        )
                    )

            if not cleaned_docs:
                return {"success": False, "chunks": 0, "pages": 0, "error": "No usable text extracted"}

            # ✅ CORRECT METHOD
            chunks = self.text_splitter.split_documents(cleaned_docs)

            self.documents = []
            for i, doc in enumerate(chunks):
                doc.metadata.update({
                    "chunk_index": i,
                    "total_pages": len(pages)
                })
                self.documents.append(doc)

            self.document_metadata = {
                "filename": original_filename,
                "total_pages": len(pages),
                "pages_with_content": len(cleaned_docs),
                "total_chunks": len(self.documents),
                "avg_chunk_length": sum(len(d.page_content) for d in self.documents) / len(self.documents)
            }

            return {"success": True, "chunks": len(self.documents), "pages": len(cleaned_docs), "error": ""}

        except Exception as e:
            return {"success": False, "chunks": 0, "pages": 0, "error": str(e)}

    # ======================================================
    # VECTOR STORE
    # ======================================================
    def create_vectorstore(self, collection_name: str) -> Dict[str, Any]:
        if not self.documents:
            return {"success": False, "error": "No documents to index"}

        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY
        )

        collection_dir = os.path.join(config.VECTOR_STORE_PATH, collection_name)

        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=embeddings,
            persist_directory=collection_dir,
            collection_name=collection_name
        )

        self.vectorstore.persist()
        return {"success": True, "error": ""}

    # ======================================================
    # SAMPLE DATA (UNCHANGED LOGIC)
    # ======================================================
    def process_sample_data(self) -> Dict[str, Any]:
        collection_name = "sample_data_index"

        if self._is_sample_data_embedded(collection_name):
            if self.load_existing_vectorstore(collection_name):
                info = self.get_document_info()
                return {
                    "success": True,
                    "chunks": info.get("total_chunks", 0),
                    "collection_name": collection_name,
                    "pages": 1,
                    "error": "already loaded"
                }

        doc = Document(
            page_content=SAMPLE_DATA_CONTENT.strip(),
            metadata={"source": "sample_english_grammar.txt", "page": 1}
        )

        self.documents = self.text_splitter.split_documents([doc])
        self.document_metadata = {
            "filename": "English Grammar Fundamentals",
            "total_chunks": len(self.documents),
            "total_pages": 1,
            "source": "static_sample"
        }

        return self.create_vectorstore(collection_name) | {
            "success": True,
            "chunks": len(self.documents),
            "collection_name": collection_name,
            "pages": 1
        }

    # ======================================================
    # HELPERS
    # ======================================================
    def _is_sample_data_embedded(self, collection_name: str) -> bool:
        return os.path.exists(os.path.join(config.VECTOR_STORE_PATH, collection_name))

    def load_existing_vectorstore(self, collection_name: str) -> bool:
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY
        )

        try:
            self.vectorstore = Chroma(
                persist_directory=os.path.join(config.VECTOR_STORE_PATH, collection_name),
                embedding_function=embeddings,
                collection_name=collection_name
            )
            return bool(self.vectorstore.similarity_search("test", k=1))
        except Exception:
            return False

    def is_collection_cached(self, collection_name: str) -> bool:
        return self._is_sample_data_embedded(collection_name)

    def _create_collection_name(self, filename: str) -> str:
        name = Path(filename).stem
        name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        return f"{name}_index"

    def get_document_info(self) -> Dict[str, Any]:
        if not self.document_metadata:
            return {"loaded": False}
        return {"loaded": True, **self.document_metadata}

    def clear_documents(self):
        self.documents = []
        self.document_metadata = {}
        self.vectorstore = None
