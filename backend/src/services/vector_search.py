from typing import List, Dict
from cohere import Client
from dotenv import load_dotenv
import os
import logging
import uuid
from ..db.vector_db import vector_db

load_dotenv()

logger = logging.getLogger(__name__)

class VectorSearchService:
    def __init__(self):
        self.cohere_client = Client(api_key=os.getenv("COHERE_API_KEY"))
        self.vector_db = vector_db
        # Initialize the collection if it doesn't exist
        self.vector_db.initialize_collection()

    async def search_by_text(self, query: str, top_k: int = 5, language: str = "en") -> List[Dict]:
        """Search for relevant content using semantic search"""
        try:
            # Determine the appropriate embedding model based on language
            model_name = "embed-english-v3.0"  # Default model for English and other languages
            input_type = "search_query"

            # If the language is Urdu, use the multilingual model
            if language.lower() in ["ur", "urdu"]:
                # For Urdu, use Cohere's multilingual embedding model
                model_name = "embed-multilingual-v2.0"  # Use multilingual model for Urdu
                input_type = "search_query"

            # Generate embedding for the query using Cohere
            response = self.cohere_client.embed(
                texts=[query],
                model=model_name,
                input_type=input_type or "search_query"  # Ensure input_type is provided
            )
            query_embedding = response.embeddings[0]

            # Search in the vector database
            results = self.vector_db.search(query_embedding, limit=top_k)

            # Format results
            formatted_results = []
            for result in results:
                # If the requested language is Urdu, translate the content if needed
                content = result.payload.get("content", "")
                metadata = result.payload.get("metadata", {})

                if language.lower() in ["ur", "urdu"]:
                    # In a real implementation, we would have Urdu content in the DB
                    # For now, we'll use the translation service to translate if needed
                    from .translation import translation_service
                    # We'll add the translation in the agent service later when generating responses
                    pass

                formatted_results.append({
                    "id": result.id,
                    "content": content,
                    "metadata": metadata,
                    "similarity": result.score
                })

            return formatted_results
        except Exception as e:
            logger.error(f"Error performing vector search: {str(e)}")
            # Return empty results in case of error
            return []

    async def embed_and_store_content(self, content_id: str, text: str, metadata: Dict = None, language: str = "en"):
        """Generate embedding for content and store in vector database"""
        if metadata is None:
            metadata = {}

        try:
            # Determine the appropriate embedding model based on language
            model_name = "embed-english-v3.0"  # Default model for English and other languages
            input_type = "search_document"  # Use search_document for document indexing

            if language.lower() in ["ur", "urdu"]:
                model_name = "embed-multilingual-v2.0"  # Use multilingual model for Urdu
                input_type = "search_document"  # Use search_document for document indexing

            # Generate embedding using Cohere
            response = self.cohere_client.embed(
                texts=[text],
                model=model_name,
                input_type=input_type or "document"  # Ensure input_type is provided
            )
            embedding = response.embeddings[0]

            # Store in vector database with language information
            payload = {
                "content": text,
                "metadata": metadata,
                "language": language  # Store language information
            }
            self.vector_db.add_document(content_id, embedding, payload)
        except Exception as e:
            logger.error(f"Error embedding and storing content: {str(e)}")
            raise

    async def add_document(self, document: Dict):
        """Add a single document to the vector database"""
        content_id = document.get("id", str(uuid.uuid4()))
        text = document.get("content", document.get("text", ""))
        metadata = document.get("metadata", {})
        language = document.get("language", "en")

        await self.embed_and_store_content(content_id, text, metadata, language)
        return {"status": "success", "id": content_id}

    async def batch_embed_and_store(self, contents: List[Dict]):
        """Batch embed and store multiple contents"""
        for content in contents:
            language = content.get("language", "en")  # Default to English if no language specified
            await self.embed_and_store_content(
                content["id"],
                content["text"],
                content.get("metadata", {}),
                language
            )

vector_search_service = VectorSearchService()