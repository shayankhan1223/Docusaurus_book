from fastapi import APIRouter, HTTPException
from typing import Dict
import uuid
from datetime import datetime
from ..models.query import QueryRequest, QueryResponse
from ..services.agent import agent_service
from ..services.validation import validation_service
from ..services.vector_search import vector_search_service
from ..utils.data_retention import data_retention_policy

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query through the AI system"""
    try:
        # Validate the query
        validation_result = await validation_service.validate_query(request.query)

        if not validation_result["is_validated"]:
            return QueryResponse(
                id=str(uuid.uuid4()),
                query=request.query,
                response="I can only help with documentation-related questions. Your query doesn't seem to be related to the documentation.",
                is_validated=False,
                validation_reason=validation_result["validation_reason"],
                sources=[],
                timestamp=datetime.now()
            )

        # Search for relevant content in the vector database
        search_results = await vector_search_service.search_by_text(
            request.query,
            top_k=3,
            language=request.language
        )

        # Process the query with the agent using the search results as context
        response_text = await agent_service.process_documentation_query(
            request.query,
            search_results,
            language=request.language
        )

        # Enforce privacy policy on any temporary data
        # (In a real implementation, we would have temporary data to process)
        processed_query = data_retention_policy.enforce_privacy_policy({
            "query": request.query,
            "session_id": request.session_id,
            "timestamp": datetime.now().isoformat()
        })

        return QueryResponse(
            id=str(uuid.uuid4()),
            query=request.query,
            response=response_text,
            is_validated=True,
            sources=[result["id"] for result in search_results],
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.post("/explain-text")
async def explain_text(text: Dict[str, str]):
    """Generate an explanation for selected text"""
    try:
        selected_text = text.get("text", "")
        language = text.get("language", "en")  # Default to English if no language specified
        if not selected_text.strip():
            raise HTTPException(status_code=400, detail="Text is required")

        explanation = await agent_service.generate_explanation(selected_text, language=language)

        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")