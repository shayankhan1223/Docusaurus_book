import os
import uuid
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentationEmbedder:
    """
    Service to embed documentation content and store it in Qdrant vector database
    """
    def __init__(self):
        # Initialize Cohere client
        self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

        # Initialize Qdrant client
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")

        # Check if this is a Qdrant Cloud instance
        if "https://" in qdrant_url and ".qdrant.io" in qdrant_url:
            # For Qdrant Cloud, extract the host from the URL
            from urllib.parse import urlparse
            parsed_url = urlparse(qdrant_url)
            host = parsed_url.netloc
            self.qdrant_client = QdrantClient(
                host=host,
                api_key=api_key,
                https=True  # Explicitly enable HTTPS for cloud instances
            )
        else:
            # For local instance or other configurations
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=api_key
            )

        # Collection name for documentation - must match backend
        self.collection_name = "documentation_content"

    def initialize_collection(self):
        """
        Initialize the Qdrant collection for documentation
        """
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if not collection_exists:
                # Create collection with appropriate vector size for Cohere embeddings
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1024,  # Cohere embeddings are typically 1024-dimensional
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise

    def embed_documentation(self, documents):
        """
        Embed documentation content and store in Qdrant

        Args:
            documents: List of dictionaries with 'id', 'content', 'title', 'url', 'language'
        """
        try:
            # Extract content for embedding
            texts = [doc['content'] for doc in documents]

            # Generate embeddings using Cohere
            response = self.cohere_client.embed(
                texts=texts,
                model="embed-english-v3.0",  # Using the English model
                input_type="search_document"  # Specify this is for search documents
            )

            embeddings = response.embeddings

            # Prepare points for Qdrant
            points = []
            for i, doc in enumerate(documents):
                # Convert string ID to UUID format for Qdrant
                point_id = str(uuid.uuid4()) if doc['id'] == '1' or doc['id'] == '2' or doc['id'] == '3' else doc['id']
                # If the ID is a simple string number, convert to UUID
                if doc['id'].isdigit():
                    point_id = str(uuid.uuid4())
                else:
                    try:
                        # Check if it's already a valid UUID
                        uuid.UUID(doc['id'])
                        point_id = doc['id']
                    except ValueError:
                        # If not a UUID, generate a new one
                        point_id = str(uuid.uuid4())

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embeddings[i],
                        payload={
                            "content": doc['content'],
                            "title": doc.get('title', ''),
                            "url": doc.get('url', ''),
                            "language": doc.get('language', 'en'),
                            "created_at": doc.get('created_at', ''),
                            "updated_at": doc.get('updated_at', '')
                        }
                    )
                )

            # Upload points to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Successfully embedded and stored {len(documents)} documents")

        except Exception as e:
            logger.error(f"Error embedding documentation: {str(e)}")
            raise

    def process_sample_docs(self):
        """
        Process sample documentation for testing purposes
        """
        sample_docs = [
            {
                "id": "1",
                "content": "Getting started with our documentation platform. This guide will help you understand the basics.",
                "title": "Getting Started Guide",
                "url": "/docs/getting-started",
                "language": "en",
                "created_at": "2025-01-01",
                "updated_at": "2025-01-01"
            },
            {
                "id": "2",
                "content": "Advanced features of our documentation system. Learn about AI-powered search and intelligent assistance.",
                "title": "Advanced Features",
                "url": "/docs/advanced-features",
                "language": "en",
                "created_at": "2025-01-01",
                "updated_at": "2025-01-01"
            },
            {
                "id": "3",
                "content": "Welcome to our documentation. یہ ہماری دستاویزات کا خیر مقدم ہے۔ یہ آپ کو بنیادی چیزوں کو سمجھنے میں مدد کرے گا۔",
                "title": "Documentation Welcome",
                "url": "/docs/welcome",
                "language": "ur",
                "created_at": "2025-01-01",
                "updated_at": "2025-01-01"
            }
        ]

        self.embed_documentation(sample_docs)
        logger.info("Sample documentation processed successfully")

def main():
    """
    Main function to run the embedding process
    """
    logger.info("Starting documentation embedding process...")

    embedder = DocumentationEmbedder()

    # Initialize the collection
    embedder.initialize_collection()

    # Process sample documentation
    embedder.process_sample_docs()

    logger.info("Documentation embedding process completed successfully!")

if __name__ == "__main__":
    main()