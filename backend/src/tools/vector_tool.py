from typing import Dict, List
from .vector_search import vector_search_service
import logging

logger = logging.getLogger(__name__)

class VectorDatabaseTool:
    """
    Tool for retrieving information from the vector database
    """
    def __init__(self):
        self.vector_search_service = vector_search_service

    async def search_documentation(self, query: str, top_k: int = 5, language: str = "en") -> List[Dict]:
        """
        Search documentation in the vector database
        """
        try:
            results = await self.vector_search_service.search_by_text(query, top_k, language)
            return results
        except Exception as e:
            logger.error(f"Error searching vector database: {str(e)}")
            return []

    async def get_document_by_id(self, doc_id: str) -> Dict:
        """
        Retrieve a specific document by ID
        """
        # In a real implementation, this would fetch from the vector DB
        # For now, return a placeholder
        return {
            "id": doc_id,
            "content": f"Content for document {doc_id}",
            "metadata": {"source": "vector_db", "retrieved_at": "timestamp"}
        }

    async def find_similar_documents(self, content: str, top_k: int = 3) -> List[Dict]:
        """
        Find documents similar to the provided content
        """
        try:
            results = await self.vector_search_service.search_by_text(content, top_k)
            return results
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            return []

    async def retrieve_content_chunks(self, query: str, max_chunks: int = 5) -> List[str]:
        """
        Retrieve content chunks relevant to the query
        """
        try:
            results = await self.search_documentation(query, top_k=max_chunks)
            return [result.get("content", "") for result in results]
        except Exception as e:
            logger.error(f"Error retrieving content chunks: {str(e)}")
            return []

    async def get_relevant_context(self, query: str) -> str:
        """
        Get relevant context for a query as a single string
        """
        try:
            chunks = await self.retrieve_content_chunks(query, max_chunks=3)
            return "\n\n".join(chunks)
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return ""

vector_tool = VectorDatabaseTool()