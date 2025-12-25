from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ValidationService:
    def __init__(self):
        # In a real implementation, this might load rules from a configuration
        self.documentation_keywords = [
            "documentation", "api", "guide", "tutorial", "how to", "reference",
            "example", "code", "usage", "configuration", "setup", "install",
            "feature", "function", "method", "class", "parameter", "config"
        ]

        self.non_documentation_keywords = [
            "weather", "joke", "news", "sports", "entertainment", "personal",
            "opinion", "gossip", "advertisement", "spam"
        ]

    async def validate_query(self, query: str) -> Dict:
        """Validate if a query is related to documentation content"""
        query_lower = query.lower()

        # Check for documentation-related keywords
        doc_matches = [keyword for keyword in self.documentation_keywords if keyword in query_lower]

        # Check for non-documentation keywords
        non_doc_matches = [keyword for keyword in self.non_documentation_keywords if keyword in query_lower]

        # Calculate relevance score
        doc_score = len(doc_matches)
        non_doc_score = len(non_doc_matches)

        is_valid = doc_score > non_doc_score and doc_score > 0

        return {
            "is_validated": is_valid,
            "validation_reason": f"Query contains {doc_score} documentation-related keywords" if is_valid else f"Query contains {non_doc_score} non-documentation keywords",
            "documentation_matches": doc_matches,
            "non_documentation_matches": non_doc_matches,
            "relevance_score": doc_score / (doc_score + non_doc_score + 1)  # +1 to avoid division by zero
        }

validation_service = ValidationService()