from fastapi import APIRouter, HTTPException
from typing import Dict
import uuid
from datetime import datetime
from ..services.agent import agent_service
from ..services.validation import validation_service
from ..services.vector_search import vector_search_service
from ..utils.data_retention import data_retention_policy

router = APIRouter()

@router.post("/text-selection")
async def process_text_selection(text_data: Dict[str, str]):
    """Process a user's selected text to provide explanation"""
    try:
        selected_text = text_data.get("text", "")
        context = text_data.get("context", "")
        session_id = text_data.get("session_id", str(uuid.uuid4()))

        if not selected_text.strip():
            raise HTTPException(status_code=400, detail="Text is required")

        # For text selection, we'll generate an explanation directly
        explanation = await agent_service.generate_explanation(selected_text)

        # Validate that the explanation is appropriate
        validation_result = await validation_service.validate_query(explanation[:100])  # Check first 100 chars

        # Enforce privacy policy
        processed_data = data_retention_policy.enforce_privacy_policy({
            "selected_text": selected_text,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })

        return {
            "id": str(uuid.uuid4()),
            "original_text": selected_text,
            "explanation": explanation,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "is_validated": validation_result["is_validated"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text selection: {str(e)}")

@router.post("/text-selection-with-search")
async def process_text_selection_with_search(text_data: Dict[str, str]):
    """Process selected text with additional search for context"""
    try:
        selected_text = text_data.get("text", "")
        context = text_data.get("context", "")
        session_id = text_data.get("session_id", str(uuid.uuid4()))

        if not selected_text.strip():
            raise HTTPException(status_code=400, detail="Text is required")

        # First, generate an explanation for the selected text
        explanation = await agent_service.generate_explanation(selected_text)

        # Then search for related content in documentation
        search_results = await vector_search_service.search_by_text(
            selected_text,
            top_k=2
        )

        related_content = [result["content"][:200] + "..." for result in search_results]

        # Enforce privacy policy
        processed_data = data_retention_policy.enforce_privacy_policy({
            "selected_text": selected_text,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })

        return {
            "id": str(uuid.uuid4()),
            "original_text": selected_text,
            "explanation": explanation,
            "context": context,
            "related_content": related_content,
            "timestamp": datetime.now().isoformat(),
            "source_docs": [result["id"] for result in search_results]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text selection with search: {str(e)}")