from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from .api.chatbot import router as chatbot_router
from .api.text_query import router as text_query_router
from .api.semantic_search import router as semantic_search_router

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Docusaurus Documentation AI API",
    description="API for AI-powered documentation search and Q&A",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chatbot_router, prefix="/api/chatbot", tags=["chatbot"])
app.include_router(text_query_router, prefix="/api/text", tags=["text-query"])
app.include_router(semantic_search_router, prefix="/api/search", tags=["semantic-search"])

@app.get("/")
def read_root():
    return {"message": "Docusaurus Documentation AI API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)