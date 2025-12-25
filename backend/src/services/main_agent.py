from typing import Dict, List
from .agent import agent_service
from .vector_search import vector_search_service
from .guardrail_agent import guardrail_agent_service
import logging

logger = logging.getLogger(__name__)

class MainAgentService:
    def __init__(self):
        self.agent_service = agent_service
        self.vector_search_service = vector_search_service
        self.guardrail_agent_service = guardrail_agent_service

    async def process_query(self, query: str, context: List[Dict] = None) -> str:
        """
        Process a query through the main agent system
        """
        # Validate the query through the guardrail agent first
        validation_result = await self.guardrail_agent_service.validate_query(query)

        if not validation_result["can_process"]:
            return f"I can only help with documentation-related questions. {validation_result['validation_reason']}"

        # Search for relevant content in the documentation
        search_results = await self.vector_search_service.search_by_text(query, top_k=3)

        # Process the query with the AI agent using the search results as context
        response = await self.agent_service.process_documentation_query(query, search_results)

        return response

    async def generate_documentation_response(self, query: str, doc_context: List[Dict] = None) -> Dict:
        """
        Generate a comprehensive response with documentation context
        """
        # Validate the query
        validation_result = await self.guardrail_agent_service.validate_query(query)

        if not validation_result["can_process"]:
            return {
                "query": query,
                "response": f"I can only help with documentation-related questions. {validation_result['validation_reason']}",
                "sources": [],
                "is_validated": False
            }

        # Search for relevant documentation
        search_results = await self.vector_search_service.search_by_text(query, top_k=5)

        # Generate response using the agent
        response = await self.agent_service.process_documentation_query(query, search_results)

        # Extract source information
        sources = [result["id"] for result in search_results]

        return {
            "query": query,
            "response": response,
            "sources": sources,
            "is_validated": True,
            "search_results_count": len(search_results)
        }

    async def answer_from_context(self, query: str, context: str) -> str:
        """
        Answer a query based on provided context
        """
        # Validate the query
        validation_result = await self.guardrail_agent_service.validate_query(query)

        if not validation_result["can_process"]:
            return f"I can only help with documentation-related questions. {validation_result['validation_reason']}"

        # Use the agent to generate a response based on the context
        prompt = f"Based on the following documentation context, answer the user's question:\n\nContext: {context}\n\nQuestion: {query}"

        # For this implementation, we'll use the agent's general processing method
        # In a real implementation, we might have a specific method for context-based answers
        return await self.agent_service.process_documentation_query(query, [{"content_preview": context}])

main_agent_service = MainAgentService()