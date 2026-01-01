"""
StudyMate RAG Engine
Main RAG system with multi-level fallback, corrective RAG, and comprehensive security.
Based on school chatbot implementation with enhancements for general educational use.
"""

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend.config import config
from backend.guardrails import StudyMateGuardrails, GuardrailResult, SafetyLevel, ContentCategory

class QualityLevel(Enum):
    """Context quality levels for Corrective RAG."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"

class RetrievalLevel(Enum):
    """Fallback retrieval levels."""
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    TERTIARY = "TERTIARY"
    QUATERNARY = "QUATERNARY"
    FALLBACK = "FALLBACK"

@dataclass
class ContextEvaluation:
    """Evaluation result for Corrective RAG."""
    relevance_score: float
    completeness_score: float
    clarity_score: float
    quality_level: QualityLevel
    needs_correction: bool
    reasoning: str

@dataclass
class RAGResponse:
    """Final response from the RAG system."""
    answer: str
    context_quality: QualityLevel
    retrieval_level: RetrievalLevel
    sources: List[str]
    was_corrected: bool
    guardrail_passed: bool
    confidence: str
    processing_time: float
    safety_check: GuardrailResult
    completeness_score: float = 0.0
    specificity_score: float = 0.0

class StudyMateRAG:
    """
    Advanced RAG system for StudyMate with comprehensive security and multi-level fallback.
    Features:
    - Guardrails for input/output safety
    - Corrective RAG (context evaluation and query refinement)
    - Multi-level retrieval fallback (5 levels)
    - Student-friendly responses
    """

    def __init__(self, vectorstore=None):
        """Initialize the RAG system."""
        # Initialize OpenAI models
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,  # Slightly higher for more natural responses
            api_key=config.OPENAI_API_KEY
        )

        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY
        )

        # Vector store
        self.vectorstore = vectorstore

        # Initialize guardrails
        self.guardrails = StudyMateGuardrails()

        # System prompt for educational responses
        self.system_prompt = """You are StudyMate, a friendly and knowledgeable AI study assistant designed specifically for students.

Your role:
- Help students understand and learn from their study materials
- Provide clear, accurate explanations based only on the provided context
- Encourage critical thinking and deeper understanding
- Be patient, supportive, and encouraging
- Use age-appropriate language suitable for students
- Stay focused on educational content and academic success

Guidelines:
- Always base your answers on the provided textbook/study material context
- If the context doesn't fully answer the question, acknowledge this and suggest what additional information might help
- Use simple, clear language that students can easily understand
- Be encouraging and positive in your responses
- If something is unclear in the material, suggest students ask their teacher or look for additional resources
- Promote academic integrity and honest learning

Remember: You're helping students learn and grow academically. Be their supportive study buddy! 📚🤝"""

    def query(self, user_query: str, verbose: bool = False) -> RAGResponse:
        """
        Main query method with comprehensive security and multi-level fallback.

        Args:
            user_query: Student's question
            verbose: Print progress information

        Returns:
            RAGResponse with answer and metadata
        """
        import time
        start_time = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"📚 Student Question: {user_query}")
            print('='*60)

        # =====================================================
        # STEP 1: INPUT GUARDRAILS
        # =====================================================
        if verbose:
            print("\n🛡️ Step 1: Input Safety Check...")

        input_check = self.guardrails.validate_input(user_query)

        if not input_check.is_safe:
            if verbose:
                print(f"   🚫 BLOCKED: {input_check.blocked_category.value}")

            return RAGResponse(
                answer=input_check.reason,
                context_quality=QualityLevel.POOR,
                retrieval_level=RetrievalLevel.FALLBACK,
                sources=[],
                was_corrected=False,
                guardrail_passed=False,
                confidence="N/A",
                processing_time=time.time() - start_time,
                safety_check=input_check
            )

        if verbose:
            print("   ✅ Input is safe!")

        safe_query = input_check.sanitized_text

        # =====================================================
        # STEP 2: PRIMARY RETRIEVAL
        # =====================================================
        if verbose:
            print("\n📥 Step 2: Primary Retrieval...")

        docs, context = self._retrieve_primary(safe_query)
        retrieval_level = RetrievalLevel.PRIMARY

        if not docs:
            if verbose:
                print("   ❌ No documents found!")
            return RAGResponse(
                answer="I couldn't find any relevant information in your uploaded document. Please make sure you've uploaded a study material and try rephrasing your question.",
                context_quality=QualityLevel.POOR,
                retrieval_level=RetrievalLevel.FALLBACK,
                sources=[],
                was_corrected=False,
                guardrail_passed=True,
                confidence="Low",
                processing_time=time.time() - start_time,
                safety_check=input_check,
                completeness_score=0.0,
                specificity_score=0.0
            )

        if verbose:
            print(f"   ✅ Found {len(docs)} relevant sections")

        # =====================================================
        # STEP 3: CORRECTIVE RAG - Context Quality Evaluation
        # =====================================================
        if verbose:
            print("\n🔍 Step 3: Context Quality Evaluation (CORRECTIVE RAG)...")

        evaluation = self._evaluate_context(safe_query, context)
        was_corrected = False

        if verbose:
            print(f"   📊 Relevance: {evaluation.relevance_score:.2f}")
            print(f"   📊 Completeness: {evaluation.completeness_score:.2f}")
            print(f"   📊 Clarity: {evaluation.clarity_score:.2f}")
            print(f"   📊 Quality Level: {evaluation.quality_level.value}")

        # =====================================================
        # STEP 4: FALLBACK MECHANISM (if needed)
        # =====================================================
        if evaluation.needs_correction:
            if verbose:
                print("\n🔄 Step 4: Applying Multi-Level Fallback...")

            # Try SECONDARY retrieval (keyword expansion)
            if verbose:
                print("   📥 Trying SECONDARY retrieval (keyword expansion)...")
            docs, context = self._retrieve_secondary(safe_query)
            evaluation = self._evaluate_context(safe_query, context)
            retrieval_level = RetrievalLevel.SECONDARY

            if evaluation.needs_correction:
                # Try TERTIARY retrieval (semantic expansion)
                if verbose:
                    print("   📥 Trying TERTIARY retrieval (semantic expansion)...")

                # Also try query refinement (Corrective RAG)
                refined_query = self._refine_query(safe_query, evaluation)
                if verbose:
                    print(f"   🔧 Refined query: {refined_query}")

                docs, context = self._retrieve_tertiary(refined_query)
                evaluation = self._evaluate_context(refined_query, context)
                retrieval_level = RetrievalLevel.TERTIARY
                was_corrected = True

                if evaluation.needs_correction:
                    # Try QUATERNARY retrieval (cross-domain expansion)
                    if verbose:
                        print("   📥 Trying QUATERNARY retrieval (cross-domain expansion)...")

                    docs, context = self._retrieve_quaternary(refined_query)
                    evaluation = self._evaluate_context(refined_query, context)
                    retrieval_level = RetrievalLevel.QUATERNARY

            if verbose and not evaluation.needs_correction:
                print(f"   ✅ Final Quality: {evaluation.quality_level.value}")

        # =====================================================
        # STEP 5: GENERATE RESPONSE
        # =====================================================
        if verbose:
            print("\n💬 Step 5: Generating Student-Friendly Response...")

        generation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Based on the textbook/study material content below, answer the student's question.

TEXTBOOK CONTENT:
{context}

STUDENT QUESTION: {safe_query}

Important: Base your answer ONLY on the provided textbook content. If the content doesn't fully answer the question, acknowledge this and suggest the student consult additional resources or their teacher.

YOUR ANSWER:""")
        ])

        response = self.llm.invoke(generation_prompt.format_messages())

        answer = response.content.strip()

        # =====================================================
        # STEP 6: OUTPUT GUARDRAILS
        # =====================================================
        if verbose:
            print("\n🛡️ Step 6: Output Safety Check...")

        output_check = self.guardrails.validate_output(answer)

        if not output_check.is_safe:
            if verbose:
                print(f"   🚫 Output blocked: {output_check.blocked_category.value}")
            answer = "I'm sorry, but I need to provide a different response. Let's focus on your studies! 📚"
        else:
            if verbose:
                print("   ✅ Output is safe!")

        # Get sources
        sources = list(set([
            f"Page {doc.metadata.get('page', '?')}"
            for doc in docs if doc.metadata.get('page')
        ]))

        # Determine confidence
        if evaluation.quality_level == QualityLevel.EXCELLENT:
            confidence = "High 🌟"
        elif evaluation.quality_level == QualityLevel.GOOD:
            confidence = "Good 👍"
        elif evaluation.quality_level == QualityLevel.FAIR:
            confidence = "Medium 🤔"
        else:
            confidence = "Low ❓"

        processing_time = time.time() - start_time

        return RAGResponse(
            answer=answer,
            context_quality=evaluation.quality_level,
            retrieval_level=retrieval_level,
            sources=sources,
            was_corrected=was_corrected,
            guardrail_passed=output_check.is_safe,
            confidence=confidence,
            processing_time=processing_time,
            safety_check=input_check,
            completeness_score=evaluation.completeness_score * 100,  # Convert to percentage
            specificity_score=evaluation.relevance_score * 100  # Use relevance as specificity
        )

    def _retrieve_primary(self, query: str, k: int = None) -> Tuple[List[Document], str]:
        """Level 1: Primary vector similarity search."""
        if k is None:
            k = config.TOP_K

        if not self.vectorstore:
            return [], ""

        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            if not docs:
                return [], ""
            context = "\n\n".join([doc.page_content for doc in docs if doc.page_content])
            return docs, context
        except Exception as e:
            print(f"   ⚠️ Primary retrieval error: {e}")
            return [], ""

    def _retrieve_secondary(self, query: str, k: int = None) -> Tuple[List[Document], str]:
        """Level 2: Keyword expansion search."""
        if k is None:
            k = config.TOP_K + 2

        if not self.vectorstore:
            return [], ""

        try:
            # Expand query with educational keywords
            expanded_query = f"{query} definition explanation example concept theory principle"
            docs = self.vectorstore.similarity_search(expanded_query, k=k)
            if not docs:
                return [], ""
            context = "\n\n".join([doc.page_content for doc in docs if doc.page_content])
            return docs, context
        except Exception as e:
            print(f"   ⚠️ Secondary retrieval error: {e}")
            return [], ""

    def _retrieve_tertiary(self, query: str, k: int = None) -> Tuple[List[Document], str]:
        """Level 3: Semantic expansion with LLM."""
        if k is None:
            k = config.TOP_K + 4

        if not self.vectorstore:
            return [], ""

        try:
            # Use LLM to expand query semantically
            expand_prompt = f"""Expand this student question with related educational concepts and synonyms.

Question: {query}

Add related terms, concepts, and educational keywords that would help find relevant textbook content.

Expanded query:"""

            response = self.llm.invoke(expand_prompt)
            expanded_query = response.content.strip()

            docs = self.vectorstore.similarity_search(expanded_query, k=k)
            if not docs:
                return [], ""
            context = "\n\n".join([doc.page_content for doc in docs if doc.page_content])
            return docs, context
        except Exception as e:
            print(f"   ⚠️ Tertiary retrieval error: {e}")
            return [], ""

    def _retrieve_quaternary(self, query: str, k: int = None) -> Tuple[List[Document], str]:
        """Level 4: Cross-domain concept search."""
        if k is None:
            k = config.TOP_K + 6

        if not self.vectorstore:
            return [], ""

        try:
            # Find analogous concepts in related academic fields
            cross_prompt = f"""Find analogous or related concepts in other academic subjects that could help explain this topic.

Question: {query}

Think of related concepts from other subjects (math, science, history, literature, etc.) that might have similar principles or explanations.

Related concepts:"""

            response = self.llm.invoke(cross_prompt)
            cross_query = response.content.strip()

            # Combine original and cross-domain queries
            combined_query = f"{query} {cross_query}"
            docs = self.vectorstore.similarity_search(combined_query, k=k)
            if not docs:
                return [], ""
            context = "\n\n".join([doc.page_content for doc in docs if doc.page_content])
            return docs, context
        except Exception as e:
            print(f"   ⚠️ Quaternary retrieval error: {e}")
            return [], ""

    def _evaluate_context(self, query: str, context: str) -> ContextEvaluation:
        """
        Evaluate the quality of retrieved context (Corrective RAG).
        """
        # Handle empty context
        if not context or not context.strip():
            return ContextEvaluation(
                relevance_score=0.0,
                completeness_score=0.0,
                clarity_score=0.0,
                quality_level=QualityLevel.POOR,
                needs_correction=True,
                reasoning="No context retrieved"
            )

        evaluation_prompt = PromptTemplate(
            template="""Evaluate how well this retrieved context answers the student's question.

STUDENT QUESTION: {query}

RETRIEVED CONTEXT:
{context}

Evaluate on 3 criteria (score 0.0 to 1.0):
1. RELEVANCE: How directly related is the context to the question?
2. COMPLETENESS: Does the context provide enough information to fully answer?
3. CLARITY: Is the context clear and understandable for students?

Respond in JSON:
{{
    "relevance_score": <0.0-1.0>,
    "completeness_score": <0.0-1.0>,
    "clarity_score": <0.0-1.0>,
    "reasoning": "<brief explanation>"
}}

JSON:""",
            input_variables=["query", "context"]
        )

        try:
            response = self.llm.invoke(evaluation_prompt.format(
                query=query,
                context=context[:2000]  # Limit context length
            ))

            response_text = response.content.strip()

            # Handle JSON in code blocks
            if "```" in response_text:
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:].strip()

            scores = json.loads(response_text)
        except Exception as e:
            print(f"   ⚠️ Evaluation error: {e}")
            scores = {
                "relevance_score": 0.5,
                "completeness_score": 0.5,
                "clarity_score": 0.5,
                "reasoning": "Evaluation parsing failed"
            }

        # Calculate average and determine quality level
        avg_score = (
            scores.get("relevance_score", 0.5) +
            scores.get("completeness_score", 0.5) +
            scores.get("clarity_score", 0.5)
        ) / 3

        if avg_score >= 0.8:
            quality_level = QualityLevel.EXCELLENT
            needs_correction = False
        elif avg_score >= 0.6:
            quality_level = QualityLevel.GOOD
            needs_correction = False
        elif avg_score >= 0.4:
            quality_level = QualityLevel.FAIR
            needs_correction = True
        else:
            quality_level = QualityLevel.POOR
            needs_correction = True

        return ContextEvaluation(
            relevance_score=scores.get("relevance_score", 0.5),
            completeness_score=scores.get("completeness_score", 0.5),
            clarity_score=scores.get("clarity_score", 0.5),
            quality_level=quality_level,
            needs_correction=needs_correction,
            reasoning=scores.get("reasoning", "")
        )

    def _refine_query(self, original_query: str, evaluation: ContextEvaluation) -> str:
        """
        Refine query when context quality is poor (Corrective RAG).
        """
        refine_prompt = PromptTemplate(
            template="""The student's question didn't find good answers in the textbook. Improve the search query.

ORIGINAL QUESTION: {query}

PROBLEM: {reasoning}

Create a better search query that:
- Uses specific keywords from the subject matter
- Is more focused and precise
- Would likely find better matching content in educational materials

Return ONLY the improved query (no explanation):

IMPROVED QUERY:""",
            input_variables=["query", "reasoning"]
        )

        try:
            response = self.llm.invoke(refine_prompt.format(
                query=original_query,
                reasoning=evaluation.reasoning
            ))
            return response.content.strip()
        except Exception as e:
            print(f"   ⚠️ Query refinement error: {e}")
            return original_query

    def get_guardrail_metrics(self) -> Dict:
        """Get guardrail safety metrics."""
        return self.guardrails.get_metrics()

    def get_safety_report(self) -> str:
        """Generate comprehensive safety report."""
        return self.guardrails.get_safety_report()