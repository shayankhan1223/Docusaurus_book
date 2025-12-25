from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    language: Optional[str] = "en"

class QueryResponse(BaseModel):
    id: str
    query: str
    response: str
    is_validated: bool
    validation_reason: Optional[str] = None
    sources: List[str]
    timestamp: datetime