from fastapi import APIRouter, HTTPException
from typing import Dict, List
import uuid
from datetime import datetime
from ..services.vector_search import vector_search_service
from ..services.query_router import query_router_service

router = APIRouter()

@router.post("/search")
async def semantic_search(request: Dict):
    """
    Perform semantic search in the documentation
    """
    try:
        query = request.get("query", "")
        top_k = request.get("top_k", 5)
        language = request.get("language", "en")

        if not query.strip():
            raise HTTPException(status_code=400, detail="Query is required")

        # Perform semantic search
        results = await vector_search_service.search_by_text(query, top_k, language)

        return {
            "query": query,
            "results": results,
            "total_results": len(results),
            "query_embedding": None,  # In a real implementation, you might return the embedding
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing semantic search: {str(e)}")

@router.post("/search-with-routing")
async def search_with_routing(request: Dict):
    """
    Perform search with intelligent query routing
    """
    try:
        query = request.get("query", "")
        top_k = request.get("top_k", 5)
        language = request.get("language", "en")

        if not query.strip():
            raise HTTPException(status_code=400, detail="Query is required")

        # Route the query to appropriate handler
        result = await query_router_service.process_query_with_classification(query)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error routing search query: {str(e)}")

@router.get("/search/{doc_id}")
async def get_document_by_id(doc_id: str):
    """
    Get a specific document by ID
    """
    try:
        # In a real implementation, this would fetch from the vector DB
        # For now, return a placeholder
        return {
            "id": doc_id,
            "content": f"Content for document {doc_id}",
            "metadata": {"source": "vector_db", "retrieved_at": datetime.now().isoformat()}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")