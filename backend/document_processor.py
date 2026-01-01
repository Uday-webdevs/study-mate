"""
StudyMate Document Processor
Handles PDF document loading, chunking, and vector store management.
"""

import os
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from backend.config import config

# Sample data for testing - English Grammar Concepts
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
        """Initialize document processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

        self.vectorstore = None
        self.documents = []
        self.document_metadata = {}

    def validate_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Validate uploaded file.
        Returns: {"valid": bool, "error": str, "size_mb": float}
        """
        if not uploaded_file:
            return {"valid": False, "error": "No file provided", "size_mb": 0}

        # Check file extension
        file_name = uploaded_file.name.lower()
        if not any(file_name.endswith(ext) for ext in config.ALLOWED_EXTENSIONS):
            return {
                "valid": False,
                "error": f"File type not allowed. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}",
                "size_mb": 0
            }

        # Check file size
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > config.MAX_FILE_SIZE_MB:
            return {
                "valid": False,
                "error": f"File too large. Maximum size: {config.MAX_FILE_SIZE_MB}MB",
                "size_mb": file_size_mb
            }

        return {"valid": True, "error": "", "size_mb": file_size_mb}

    def process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Process uploaded file and create vector store.
        Checks for existing cache first.
        Returns: {"success": bool, "chunks": int, "error": str, "collection_name": str}
        """
        try:
            # Validate file
            validation = self.validate_file(uploaded_file)
            if not validation["valid"]:
                return {
                    "success": False,
                    "chunks": 0,
                    "error": validation["error"],
                    "collection_name": ""
                }

            collection_name = self._create_collection_name(uploaded_file.name)

            # Check if already cached
            if self.is_collection_cached(collection_name):
                # Load existing vectorstore
                if self.load_existing_vectorstore(collection_name):
                    # Get document info from metadata
                    doc_info = self.get_document_info()
                    return {
                        "success": True,
                        "chunks": doc_info.get("total_chunks", 0),
                        "error": "",
                        "collection_name": collection_name,
                        "pages": doc_info.get("total_pages", 0),
                        "file_size_mb": validation["size_mb"],
                        "cached": True
                    }

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
                print(f"📥 Saved uploaded file to temporary path: {tmp_path}")
            try:
                # Process the PDF
                result = self.load_and_chunk_pdf(tmp_path, uploaded_file.name)

                if result["success"]:
                    # Create vector store
                    vectorstore_result = self.create_vectorstore(collection_name)

                    if vectorstore_result["success"]:
                        return {
                            "success": True,
                            "chunks": result["chunks"],
                            "error": "",
                            "collection_name": collection_name,
                            "pages": result["pages"],
                            "file_size_mb": validation["size_mb"],
                            "cached": False
                        }
                    else:
                        return {
                            "success": False,
                            "chunks": 0,
                            "error": vectorstore_result["error"],
                            "collection_name": ""
                        }
                else:
                    return result

            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        except Exception as e:
            return {
                "success": False,
                "chunks": 0,
                "error": f"Processing failed: {str(e)}",
                "collection_name": ""
            }

    def load_and_chunk_pdf(self, pdf_path: str, original_filename: str) -> Dict[str, Any]:
        """
        Load PDF and split into chunks.
        Returns: {"success": bool, "chunks": int, "pages": int, "error": str}
        """
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            if not pages:
                return {
                    "success": False,
                    "chunks": 0,
                    "pages": 0,
                    "error": "No content found in PDF"
                }

            print(f"📄 Loaded {len(pages)} pages from {original_filename}")

            # Process pages and filter out empty ones
            all_chunks = []
            pages_with_content = 0

            for i, page in enumerate(pages):
                try:
                    page_content = page.page_content.strip()

                    # Skip empty or very short pages
                    if not page_content or len(page_content) < 50:
                        continue

                    pages_with_content += 1

                    # Split text with error handling
                    try:
                        chunks = self.text_splitter.split_text(page_content)
                    except Exception as split_error:
                        print(f"⚠️ Error splitting page {i+1}: {split_error}")
                        # Fallback: treat whole page as one chunk
                        chunks = [page_content]

                    # Filter out empty chunks and create documents
                    for j, chunk in enumerate(chunks):
                        if chunk and chunk.strip():
                            doc = Document(
                                page_content=chunk.strip(),
                                metadata={
                                    "source": original_filename,
                                    "page": i + 1,
                                    "chunk_id": j,
                                    "total_pages": len(pages),
                                    "chunk_index": len(all_chunks)
                                }
                            )
                            all_chunks.append(doc)

                except Exception as page_error:
                    print(f"⚠️ Error processing page {i+1}: {page_error}")
                    continue

            if not all_chunks:
                return {
                    "success": False,
                    "chunks": 0,
                    "pages": pages_with_content,
                    "error": "No usable text content extracted from PDF"
                }

            # Store documents
            self.documents = all_chunks
            self.document_metadata = {
                "filename": original_filename,
                "total_pages": len(pages),
                "pages_with_content": pages_with_content,
                "total_chunks": len(all_chunks),
                "avg_chunk_length": sum(len(doc.page_content) for doc in all_chunks) / len(all_chunks)
            }

            print(f"✅ Created {len(all_chunks)} chunks from {pages_with_content} content pages")

            return {
                "success": True,
                "chunks": len(all_chunks),
                "pages": pages_with_content,
                "error": ""
            }

        except Exception as e:
            return {
                "success": False,
                "chunks": 0,
                "pages": 0,
                "error": f"PDF loading failed: {str(e)}"
            }

    def create_vectorstore(self, collection_name: str) -> Dict[str, Any]:
        """
        Create ChromaDB vectorstore from documents.
        Returns: {"success": bool, "error": str}
        """
        if not self.documents:
            return {"success": False, "error": "No documents to index"}

        try:
            from langchain_openai import OpenAIEmbeddings
            from langchain_chroma import Chroma

            # Create embeddings
            embeddings = OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL,
                api_key=config.OPENAI_API_KEY
            )

            # Create unique directory for this collection
            collection_dir = os.path.join(config.VECTOR_STORE_PATH, collection_name)

            # Create vectorstore with persist_directory
            self.vectorstore = Chroma.from_documents(
                documents=self.documents,
                embedding=embeddings,
                persist_directory=collection_dir,
                collection_name=collection_name
            )

            # Persist the vectorstore
            self.vectorstore.persist()

            print(f"✅ Vector store created: {collection_name} in {collection_dir} with {len(self.documents)} documents")

            return {"success": True, "error": ""}

        except Exception as e:
            return {"success": False, "error": f"Vector store creation failed: {str(e)}"}

    def load_existing_vectorstore(self, collection_name: str) -> bool:
        """
        Load existing vectorstore.
        Returns: True if loaded successfully
        """
        try:
            from langchain_openai import OpenAIEmbeddings
            from langchain_chroma import Chroma

            embeddings = OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL,
                api_key=config.OPENAI_API_KEY
            )

            # Use collection-specific directory
            collection_dir = os.path.join(config.VECTOR_STORE_PATH, collection_name)

            self.vectorstore = Chroma(
                persist_directory=collection_dir,
                embedding_function=embeddings,
                collection_name=collection_name
            )

            # Test if collection exists and has documents
            try:
                docs = self.vectorstore.similarity_search("test", k=1)
                if docs:
                    # Get collection count and set metadata
                    collection_count = self.vectorstore._collection.count()

                    # Set basic metadata for cached data
                    self.document_metadata = {
                        "filename": collection_name.replace("_index", ""),
                        "total_chunks": collection_count,
                        "total_pages": 1 if collection_name == "sample_data_index" else 0,  # We don't know pages for cached files
                        "source": "cached"
                    }
                return True
            except:
                return False

        except Exception as e:
            print(f"Error loading vectorstore: {e}")
            return False

    def _create_collection_name(self, filename: str) -> str:
        """Create a unique collection name from filename."""
        # Remove extension and sanitize
        name = Path(filename).stem
        # Replace spaces and special chars with underscores
        name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        return f"{name}_index"

    def _create_collection_name(self, filename: str) -> str:
        """Create a unique collection name from filename."""
        # Remove extension and sanitize
        name = Path(filename).stem
        # Replace spaces and special chars with underscores
        name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        return f"{name}_index"

    def create_collection_name(self, filename: str) -> str:
        """Create a unique collection name from filename. Public method."""
        return self._create_collection_name(filename)

    def get_document_info(self) -> Dict[str, Any]:
        """Get information about the loaded document."""
        if not self.document_metadata:
            return {"loaded": False}

        return {
            "loaded": True,
            **self.document_metadata
        }

    def process_file_from_path(self, file_path: str, display_name: str) -> Dict[str, Any]:
        """
        Process a file from a file path (for sample data).
        Returns: {"success": bool, "chunks": int, "error": str, "collection_name": str, "pages": int}
        """
        try:
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "chunks": 0,
                    "error": f"File not found: {file_path}",
                    "collection_name": "",
                    "pages": 0
                }

            # Check file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > config.MAX_FILE_SIZE_MB:
                return {
                    "success": False,
                    "chunks": 0,
                    "error": f"File too large. Maximum size: {config.MAX_FILE_SIZE_MB}MB",
                    "collection_name": "",
                    "pages": 0
                }

            # Process the PDF
            result = self.load_and_chunk_pdf(file_path, display_name)

            if result["success"]:
                # Create vector store
                collection_name = self._create_collection_name(display_name)
                vectorstore_result = self.create_vectorstore(collection_name)

                if vectorstore_result["success"]:
                    return {
                        "success": True,
                        "chunks": result["chunks"],
                        "error": "",
                        "collection_name": collection_name,
                        "pages": result["pages"]
                    }
                else:
                    return {
                        "success": False,
                        "chunks": 0,
                        "error": vectorstore_result["error"],
                        "collection_name": "",
                        "pages": 0
                    }
            else:
                return result

        except Exception as e:
            return {
                "success": False,
                "chunks": 0,
                "error": f"Processing failed: {str(e)}",
                "collection_name": "",
                "pages": 0
            }

    def process_sample_data(self) -> Dict[str, Any]:
        """
        Process the static sample data about English grammar.
        Checks if already embedded to save costs.
        Returns: {"success": bool, "chunks": int, "error": str, "collection_name": str, "pages": int}
        """
        collection_name = "sample_data_index"

        try:
            # Check if sample data is already embedded
            if self._is_sample_data_embedded(collection_name):
                # Load existing vectorstore
                if self.load_existing_vectorstore(collection_name):
                    # Get document info from metadata
                    doc_info = self.get_document_info()
                    return {
                        "success": True,
                        "chunks": doc_info.get("total_chunks", 0),
                        "error": "already loaded",
                        "collection_name": collection_name,
                        "pages": 1  # Sample data is treated as 1 page
                    }

            # Process the sample text content
            documents = [Document(
                page_content=SAMPLE_DATA_CONTENT.strip(),
                metadata={"source": "sample_english_grammar.txt", "page": 1}
            )]

            # Split into chunks
            self.documents = self.text_splitter.split_documents(documents)
            self.document_metadata = {
                "filename": "English Grammar Fundamentals",
                "chunks": len(self.documents),
                "pages": 1,
                "file_size_mb": len(SAMPLE_DATA_CONTENT) / (1024 * 1024),
                "source": "static_sample"
            }

            # Create vectorstore
            vectorstore_result = self.create_vectorstore(collection_name)

            if vectorstore_result["success"]:
                return {
                    "success": True,
                    "chunks": len(self.documents),
                    "error": "",
                    "collection_name": collection_name,
                    "pages": 1
                }
            else:
                return {
                    "success": False,
                    "chunks": 0,
                    "error": vectorstore_result["error"],
                    "collection_name": "",
                    "pages": 0
                }

        except Exception as e:
            return {
                "success": False,
                "chunks": 0,
                "error": f"Sample data processing failed: {str(e)}",
                "collection_name": "",
                "pages": 0
            }

    def _is_sample_data_embedded(self, collection_name: str) -> bool:
        """
        Check if sample data is already embedded in the vectorstore.
        """
        try:
            from langchain_openai import OpenAIEmbeddings
            from langchain_chroma import Chroma

            embeddings = OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL,
                api_key=config.OPENAI_API_KEY
            )

            # Check collection-specific directory
            collection_dir = os.path.join(config.CHROMA_PATH, collection_name)
            if not os.path.exists(collection_dir):
                return False

            # Try to load the collection
            try:
                vectorstore = Chroma(
                    persist_directory=collection_dir,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )
                # Test if collection has documents
                docs = vectorstore.similarity_search("test", k=1)
                return len(docs) > 0
            except:
                return False

        except:
            return False

    def is_collection_cached(self, collection_name: str) -> bool:
        """
        Check if a collection is already cached in the vectorstore.
        Public method for UI access.
        """
        return self._is_sample_data_embedded(collection_name)

    def clear_documents(self):
        """Clear loaded documents and vectorstore."""
        self.documents = []
        self.document_metadata = {}
        self.vectorstore = None