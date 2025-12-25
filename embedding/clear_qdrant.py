import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

class QdrantManager:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")

        # Check if this is a Qdrant Cloud instance
        if "https://" in qdrant_url and ".qdrant.io" in qdrant_url:
            # For Qdrant Cloud, extract the host from the URL
            from urllib.parse import urlparse
            parsed_url = urlparse(qdrant_url)
            host = parsed_url.netloc
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

    def clear_collection(self):
        """Clear all points from the collection"""
        try:
            # Get all points in the collection
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000  # Adjust based on your needs
            )

            # Extract IDs of all points
            point_ids = [point.id for point in scroll_result[0]]

            if point_ids:
                # Delete all points
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=point_ids
                    )
                )
                print(f"Deleted {len(point_ids)} points from collection '{self.collection_name}'")
            else:
                print(f"Collection '{self.collection_name}' is already empty")

        except Exception as e:
            print(f"Error clearing collection: {e}")
            # If collection doesn't exist, that's fine
            if "doesn't exist" in str(e).lower() or "not found" in str(e).lower():
                print(f"Collection '{self.collection_name}' doesn't exist, will be created later")

    def recreate_collection(self):
        """Recreate the collection with the proper vector configuration"""
        try:
            # Delete the collection if it exists
            self.client.delete_collection(self.collection_name)
            print(f"Deleted collection '{self.collection_name}'")
        except:
            print(f"Collection '{self.collection_name}' didn't exist, creating new one")

        # Create collection with appropriate vector size for Cohere embeddings
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=1024,  # Cohere embeddings are typically 1024-dimensional
                distance=models.Distance.COSINE
            )
        )
        print(f"Created collection '{self.collection_name}' with vector size 1024")


def main():
    print("Clearing Qdrant collection...")
    manager = QdrantManager()
    manager.recreate_collection()
    print("Qdrant collection has been cleared and recreated.")


if __name__ == "__main__":
    main()