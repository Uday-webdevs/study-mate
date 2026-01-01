"""
StudyMate Guardrails System
Comprehensive security and safety checks for user protection, knowledge protection, and app misuse prevention.
Based on school chatbot guardrails with enhancements for general educational use.
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from langchain_core.prompts import PromptTemplate
from backend.config import config

class SafetyLevel(Enum):
    """Safety assessment levels."""
    SAFE = "SAFE"
    WARNING = "WARNING"
    DANGEROUS = "DANGEROUS"
    BLOCKED = "BLOCKED"

class ContentCategory(Enum):
    """Content categories for classification."""
    EDUCATIONAL = "educational"
    INAPPROPRIATE = "inappropriate"
    HARMFUL = "harmful"
    MISLEADING = "misleading"
    CHEATING = "cheating"
    PERSONAL_INFO = "personal_info"
    OFF_TOPIC = "off_topic"

@dataclass
class GuardrailResult:
    """Result of guardrail validation."""
    is_safe: bool
    safety_level: SafetyLevel
    blocked_category: Optional[ContentCategory]
    reason: str
    sanitized_text: str
    confidence_score: float
    processing_time: float

@dataclass
class SafetyMetrics:
    """Safety metrics tracking."""
    total_queries: int = 0
    safe_queries: int = 0
    blocked_queries: int = 0
    warning_queries: int = 0
    category_counts: Dict[str, int] = None
    recent_blocks: List[Dict] = None

    def __post_init__(self):
        if self.category_counts is None:
            self.category_counts = {}
        if self.recent_blocks is None:
            self.recent_blocks = []

class StudyMateGuardrails:
    """
    Comprehensive guardrails system for StudyMate chatbot.
    Protects users, knowledge base, and prevents app misuse.
    """

    def __init__(self):
        """Initialize guardrails with patterns and prompts."""
        self.metrics = SafetyMetrics()

        # Initialize patterns for rule-based filtering
        self._init_patterns()

        # Initialize LLM-based validation (lazy loaded)
        self.llm = None
        self.embeddings = None

        # Content policies
        self.max_query_length = config.MAX_QUERY_LENGTH
        self.max_response_length = config.MAX_RESPONSE_LENGTH

    def _init_patterns(self):
        """Initialize regex patterns for content filtering."""
        # Inappropriate content patterns
        self.inappropriate_patterns = [
            r'\b(sex|sexual|porn|xxx|nsfw)\b',
            r'\b(drug|drugs|cocaine|heroin|marijuana|weed)\b',
            r'\b(violence|violent|kill|murder|rape|assault)\b',
            r'\b(hate|hateful|racist|racism|nazi|kkk)\b',
            r'\b(cheat|cheating|exam.*hack|test.*hack)\b',
            r'\b(hack|hacking|exploit|malware|virus)\b',
            r'\b(suicide|self.*harm|cutting)\b',
            r'\b(alcohol|beer|wine|liquor|get.*drunk)\b',
            r'\b(gambling|casino|bet|lottery)\b',
            r'\b(weapon|gun|knife|bomb|explosive)\b'
        ]

        # Personal information patterns
        self.personal_info_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
            r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{1,5}\s\w+\s\w+\b',  # Address patterns
            r'\b(password|login|username|credential)\b'
        ]

        # Educational keywords (positive indicators)
        self.educational_keywords = [
            'study', 'learn', 'understand', 'explain', 'chapter', 'lesson',
            'homework', 'assignment', 'concept', 'theory', 'practice',
            'example', 'question', 'answer', 'review', 'summary'
        ]

    def _rule_based_check(self, text: str) -> Tuple[bool, Optional[ContentCategory], str]:
        """
        Rule-based content checking using regex patterns.
        Returns: (is_safe, category, reason)
        """
        text_lower = text.lower()

        # Check inappropriate content
        for pattern in self.inappropriate_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False, ContentCategory.INAPPROPRIATE, "Contains inappropriate content"

        # Check personal information
        for pattern in self.personal_info_patterns:
            if re.search(pattern, text):
                return False, ContentCategory.PERSONAL_INFO, "Contains personal information"

        # Check cheating attempts
        cheating_indicators = ['how to cheat', 'cheat on', 'hack exam', 'steal answers']
        for indicator in cheating_indicators:
            if indicator in text_lower:
                return False, ContentCategory.CHEATING, "Academic dishonesty attempt detected"

        # Check off-topic content
        educational_score = sum(1 for keyword in self.educational_keywords if keyword in text_lower)
        if len(text.split()) > 10 and educational_score < 2:
            # Allow some non-educational queries but flag suspicious ones
            suspicious_words = ['party', 'game', 'movie', 'music', 'sport', 'celebrity']
            suspicious_score = sum(1 for word in suspicious_words if word in text_lower)
            if suspicious_score > educational_score:
                return False, ContentCategory.OFF_TOPIC, "Query appears off-topic for educational assistant"

        return True, None, ""

    def _llm_based_check(self, text: str, check_type: str) -> Dict[str, Any]:
        """
        LLM-based content validation for complex cases.
        """
        if not self.llm:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)

        if check_type == "input":
            prompt = PromptTemplate(
                template="""Analyze this user query for safety and appropriateness in an educational context.

Query: {query}

Evaluate:
1. Is this query safe for an educational AI assistant?
2. Does it promote academic integrity?
3. Is it appropriate for students?
4. Could it lead to harmful or inappropriate responses?

Respond in JSON:
{{
    "is_safe": true/false,
    "category": "educational/inappropriate/harmful/misleading/cheating/personal_info/off_topic",
    "reason": "brief explanation",
    "confidence": 0.0-1.0
}}

JSON:""",
                input_variables=["query"]
            )
        else:  # output check
            prompt = PromptTemplate(
                template="""Review this AI response for safety and appropriateness.

Response: {response}

Check if the response:
- Contains harmful information
- Promotes inappropriate behavior
- Includes personal information
- Violates educational guidelines

Respond in JSON:
{{
    "is_safe": true/false,
    "category": "educational/inappropriate/harmful/misleading/cheating/personal_info/off_topic",
    "reason": "brief explanation",
    "confidence": 0.0-1.0
}}

JSON:""",
                input_variables=["response"]
            )

        try:
            response = self.llm.invoke(prompt.format(**{check_type: text}))
            result = json.loads(response.content.strip())

            # Handle JSON in code blocks
            if isinstance(result, str) and "```" in result:
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
                result = json.loads(result)

            return result
        except Exception as e:
            print(f"LLM validation error: {e}")
            return {
                "is_safe": True,  # Default to safe on error
                "category": "educational",
                "reason": "Validation error, defaulting to safe",
                "confidence": 0.5
            }

    def validate_input(self, query: str) -> GuardrailResult:
        """
        Comprehensive input validation.
        """
        start_time = datetime.now()

        # Update metrics
        self.metrics.total_queries += 1

        # Basic validation
        if not query or not query.strip():
            return GuardrailResult(
                is_safe=False,
                safety_level=SafetyLevel.BLOCKED,
                blocked_category=ContentCategory.OFF_TOPIC,
                reason="Empty query",
                sanitized_text="",
                confidence_score=1.0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

        if len(query) > self.max_query_length:
            return GuardrailResult(
                is_safe=False,
                safety_level=SafetyLevel.BLOCKED,
                blocked_category=ContentCategory.OFF_TOPIC,
                reason=f"Query too long (max {self.max_query_length} characters)",
                sanitized_text=query[:self.max_query_length],
                confidence_score=1.0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

        # Rule-based check
        rule_safe, rule_category, rule_reason = self._rule_based_check(query)

        if not rule_safe:
            self.metrics.blocked_queries += 1
            self.metrics.category_counts[rule_category.value] = \
                self.metrics.category_counts.get(rule_category.value, 0) + 1

            self.metrics.recent_blocks.append({
                "timestamp": datetime.now().isoformat(),
                "query": query[:100] + "..." if len(query) > 100 else query,
                "category": rule_category.value,
                "reason": rule_reason
            })

            return GuardrailResult(
                is_safe=False,
                safety_level=SafetyLevel.BLOCKED,
                blocked_category=rule_category,
                reason=rule_reason,
                sanitized_text=self._sanitize_text(query),
                confidence_score=0.9,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

        # LLM-based check for complex cases
        if config.ENABLE_GUARDRAILS:
            llm_result = self._llm_based_check(query, "input")

            if not llm_result.get("is_safe", True):
                category = ContentCategory(llm_result.get("category", "inappropriate"))
                self.metrics.blocked_queries += 1
                self.metrics.category_counts[category.value] = \
                    self.metrics.category_counts.get(category.value, 0) + 1

                return GuardrailResult(
                    is_safe=False,
                    safety_level=SafetyLevel.BLOCKED,
                    blocked_category=category,
                    reason=llm_result.get("reason", "LLM validation failed"),
                    sanitized_text=self._sanitize_text(query),
                    confidence_score=llm_result.get("confidence", 0.8),
                    processing_time=(datetime.now() - start_time).total_seconds()
                )

        # Query is safe
        self.metrics.safe_queries += 1

        return GuardrailResult(
            is_safe=True,
            safety_level=SafetyLevel.SAFE,
            blocked_category=None,
            reason="Query passed all safety checks",
            sanitized_text=self._sanitize_text(query),
            confidence_score=0.95,
            processing_time=(datetime.now() - start_time).total_seconds()
        )

    def validate_output(self, response: str) -> GuardrailResult:
        """
        Comprehensive output validation.
        """
        start_time = datetime.now()

        # Basic validation
        if len(response) > self.max_response_length:
            sanitized = response[:self.max_response_length] + "..."
            return GuardrailResult(
                is_safe=False,
                safety_level=SafetyLevel.WARNING,
                blocked_category=ContentCategory.OFF_TOPIC,
                reason=f"Response too long (truncated to {self.max_response_length} characters)",
                sanitized_text=sanitized,
                confidence_score=0.8,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

        # Rule-based check
        rule_safe, rule_category, rule_reason = self._rule_based_check(response)

        if not rule_safe:
            return GuardrailResult(
                is_safe=False,
                safety_level=SafetyLevel.BLOCKED,
                blocked_category=rule_category,
                reason=f"Response contains {rule_reason}",
                sanitized_text=self._sanitize_response(response),
                confidence_score=0.9,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

        # LLM-based check
        if config.ENABLE_GUARDRAILS:
            llm_result = self._llm_based_check(response, "output")

            if not llm_result.get("is_safe", True):
                return GuardrailResult(
                    is_safe=False,
                    safety_level=SafetyLevel.BLOCKED,
                    blocked_category=ContentCategory(llm_result.get("category", "inappropriate")),
                    reason=llm_result.get("reason", "Response validation failed"),
                    sanitized_text=self._sanitize_response(response),
                    confidence_score=llm_result.get("confidence", 0.8),
                    processing_time=(datetime.now() - start_time).total_seconds()
                )

        return GuardrailResult(
            is_safe=True,
            safety_level=SafetyLevel.SAFE,
            blocked_category=None,
            reason="Response passed all safety checks",
            sanitized_text=response,
            confidence_score=0.95,
            processing_time=(datetime.now() - start_time).total_seconds()
        )

    def _sanitize_text(self, text: str) -> str:
        """Sanitize input text by removing or masking sensitive content."""
        # Basic sanitization - remove excessive whitespace, normalize
        sanitized = " ".join(text.split())
        return sanitized.strip()

    def _sanitize_response(self, response: str) -> str:
        """Sanitize output response."""
        # For responses, we might want to replace sensitive content with placeholders
        # or provide a safe alternative response
        return "I'm sorry, but I can't provide that information. Let's focus on your studies! 📚"

    def get_metrics(self) -> Dict[str, Any]:
        """Get safety metrics."""
        total = self.metrics.total_queries
        return {
            "total_queries": total,
            "safe_queries": self.metrics.safe_queries,
            "blocked_queries": self.metrics.blocked_queries,
            "warning_queries": self.metrics.warning_queries,
            "safe_percentage": (self.metrics.safe_queries / total * 100) if total > 0 else 0,
            "blocked_percentage": (self.metrics.blocked_queries / total * 100) if total > 0 else 0,
            "category_breakdown": self.metrics.category_counts,
            "recent_blocks": self.metrics.recent_blocks[-10:]  # Last 10 blocks
        }

    def get_safety_report(self) -> str:
        """Generate a comprehensive safety report."""
        metrics = self.get_metrics()

        report = f"""
🛡️ StudyMate Safety Report
{'='*40}

📊 Overall Statistics:
• Total Queries: {metrics['total_queries']}
• Safe Queries: {metrics['safe_queries']} ({metrics['safe_percentage']:.1f}%)
• Blocked Queries: {metrics['blocked_queries']} ({metrics['blocked_percentage']:.1f}%)
• Warning Queries: {metrics['warning_queries']}

📋 Category Breakdown:
"""

        for category, count in metrics['category_breakdown'].items():
            report += f"• {category.title()}: {count}\n"

        if metrics['recent_blocks']:
            report += "\n🚫 Recent Blocks:\n"
            for block in metrics['recent_blocks'][-5:]:  # Last 5
                report += f"• {block['timestamp'][:19]}: {block['category']} - {block['query']}\n"

        return report.strip()

    def reset_metrics(self):
        """Reset safety metrics."""
        self.metrics = SafetyMetrics()