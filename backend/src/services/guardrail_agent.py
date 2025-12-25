from typing import Dict
from .validation import validation_service
import logging

logger = logging.getLogger(__name__)

class GuardrailAgentService:
    def __init__(self):
        self.validation_service = validation_service

    async def validate_query(self, query: str) -> Dict:
        """
        Validate if a query is related to documentation content
        """
        validation_result = await self.validation_service.validate_query(query)

        return {
            "is_validated": validation_result["is_validated"],
            "validation_reason": validation_result["validation_reason"],
            "relevance_score": validation_result["relevance_score"],
            "can_process": validation_result["is_validated"]
        }

    async def check_content_relevance(self, query: str, content_snippet: str) -> bool:
        """
        Check if the content snippet is relevant to the query
        """
        # Combine query and content to check relevance
        combined_text = f"{query} {content_snippet}"
        validation_result = await self.validation_service.validate_query(combined_text)

        return validation_result["is_validated"]

    async def filter_irrelevant_queries(self, queries: list) -> list:
        """
        Filter out queries that are not related to documentation
        """
        filtered_queries = []
        for query in queries:
            validation_result = await self.validate_query(query)
            if validation_result["can_process"]:
                filtered_queries.append(query)

        return filtered_queries

guardrail_agent_service = GuardrailAgentService()