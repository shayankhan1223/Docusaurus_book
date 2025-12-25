from typing import Dict, Any
from .guardrail_agent import guardrail_agent_service
from .main_agent import main_agent_service
from .vector_search import vector_search_service
import logging

logger = logging.getLogger(__name__)

class QueryRouterService:
    """
    Service to route queries to appropriate handlers based on content and context
    """
    def __init__(self):
        self.guardrail_agent = guardrail_agent_service
        self.main_agent = main_agent_service
        self.vector_search = vector_search_service

    async def route_query(self, query: str, query_type: str = "general") -> Dict[str, Any]:
        """
        Route a query to the appropriate handler based on its nature
        """
        # Validate the query first
        validation_result = await self.guardrail_agent.validate_query(query)

        if not validation_result["can_process"]:
            return {
                "response": f"I can only help with documentation-related questions. {validation_result['validation_reason']}",
                "query_type": "unprocessed",
                "sources": [],
                "is_validated": False
            }

        # Determine the appropriate handler based on query type
        if query_type == "text_selection":
            return await self._handle_text_selection_query(query)
        elif query_type == "search":
            return await self._handle_search_query(query)
        else:
            return await self._handle_general_query(query)

    async def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """
        Handle a general documentation query
        """
        try:
            result = await self.main_agent.generate_documentation_response(query)
            return result
        except Exception as e:
            logger.error(f"Error handling general query: {str(e)}")
            return {
                "query": query,
                "response": "I encountered an error while processing your query. Please try again.",
                "sources": [],
                "is_validated": True
            }

    async def _handle_text_selection_query(self, query: str) -> Dict[str, Any]:
        """
        Handle a query that came from text selection
        """
        try:
            # For text selection, we might want to provide explanation or related info
            search_results = await self.vector_search.search_by_text(query, top_k=2)

            if search_results:
                # Use the main agent to generate a response based on search results
                response = await self.main_agent.answer_from_context(query, search_results[0]["content"])
            else:
                # If no related content found, try to explain the selected text
                from .agent import agent_service
                response = await agent_service.generate_explanation(query)

            return {
                "query": query,
                "response": response,
                "sources": [result["id"] for result in search_results],
                "is_validated": True
            }
        except Exception as e:
            logger.error(f"Error handling text selection query: {str(e)}")
            return {
                "query": query,
                "response": "I encountered an error while processing your text selection. Please try again.",
                "sources": [],
                "is_validated": True
            }

    async def _handle_search_query(self, query: str) -> Dict[str, Any]:
        """
        Handle a search-specific query
        """
        try:
            # Perform semantic search in the documentation
            search_results = await self.vector_search.search_by_text(query, top_k=5)

            if search_results:
                # Format search results
                formatted_results = [
                    {
                        "id": result["id"],
                        "content": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                        "similarity": result["similarity"]
                    }
                    for result in search_results
                ]

                response = f"I found {len(search_results)} relevant documentation items:\n\n"
                for i, result in enumerate(formatted_results, 1):
                    response += f"{i}. {result['content']}\n\n"
            else:
                response = "I couldn't find any relevant documentation for your query."

            return {
                "query": query,
                "response": response,
                "sources": [result["id"] for result in search_results],
                "is_validated": True
            }
        except Exception as e:
            logger.error(f"Error handling search query: {str(e)}")
            return {
                "query": query,
                "response": "I encountered an error while searching the documentation. Please try again.",
                "sources": [],
                "is_validated": True
            }

    async def classify_query(self, query: str) -> str:
        """
        Classify the query type to determine routing
        """
        query_lower = query.lower()

        # Simple classification based on keywords
        text_selection_indicators = ["explain", "what does this mean", "define", "meaning of", "help me understand"]
        search_indicators = ["find", "search", "show me", "documentation for", "how to", "where is", "locate"]

        for indicator in text_selection_indicators:
            if indicator in query_lower:
                return "text_selection"

        for indicator in search_indicators:
            if indicator in query_lower:
                return "search"

        return "general"

    async def process_query_with_classification(self, query: str) -> Dict[str, Any]:
        """
        Process a query by first classifying it, then routing appropriately
        """
        query_type = await self.classify_query(query)
        return await self.route_query(query, query_type)

query_router_service = QueryRouterService()