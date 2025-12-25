import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()

class VectorDB:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")

        # Check if this is a Qdrant Cloud instance
        if "https://" in qdrant_url and ".qdrant.io" in qdrant_url:
            # For Qdrant Cloud, we need to extract the host and use the https parameter
            from urllib.parse import urlparse
            parsed_url = urlparse(qdrant_url)
            host = parsed_url.netloc
            # For Qdrant Cloud, we use the host parameter with https=True
            self.client = QdrantClient(
                host=host,
                api_key=api_key,
                https=True  # Explicitly enable HTTPS for cloud instances
            )
        else:
            # For local instance or other configurations
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=api_key
            )
        self.collection_name = "documentation_content"

    def initialize_collection(self):
        """Initialize the collection for documentation content"""
        try:
            # Try to get the collection to see if it exists
            self.client.get_collection(self.collection_name)
            # If we get here, the collection exists, so we're done
        except Exception as e:
            # Collection doesn't exist, create it
            # Check if the error is because the collection doesn't exist or due to validation issues
            error_msg = str(e).lower()
            if "doesn't exist" in error_msg or "not found" in error_msg or "404" in str(e):
                try:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)  # Cohere v3 model produces 1024-dim vectors
                    )
                except Exception as create_error:
                    # If creation fails due to already existing, that's fine
                    if "already exists" not in str(create_error).lower() and "409" not in str(create_error):
                        raise create_error

    def search(self, query_vector: list, limit: int = 5):
        """Search for similar content in the vector database"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return results

    def add_document(self, id: str, vector: list, payload: dict):
        """Add a document to the vector database"""
        import uuid
        # Ensure the ID is a valid UUID
        try:
            # Try to parse the ID as UUID, if it fails, generate a new one
            uuid.UUID(id)
            point_id = id
        except ValueError:
            # If it's not a valid UUID, generate a new one but keep the original as a field in payload
            point_id = str(uuid.uuid4())
            # Store the original ID in the payload for reference
            payload['original_id'] = id

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

# Global instance
vector_db = VectorDB()