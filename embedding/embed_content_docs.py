import os
import uuid
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import logging
from pathlib import Path
import re
from urllib.parse import urlparse


def extract_content_docs(content_dir: str) -> list:
    """
    Extract content from the content directory

    Args:
        content_dir: Path to the content directory

    Returns:
        List of dictionaries containing document information
    """
    documents = []

    content_path = Path(content_dir)

    for file_path in content_path.glob("*.md"):
        # Read the markdown file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract title from the first heading or use filename
        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1)
        else:
            # Use filename without extension as title
            title = file_path.stem.replace('_', ' ').replace('-', ' ').title()

        # Remove any existing frontmatter (content between --- markers at the beginning)
        content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)

        # Clean up the content (remove extra whitespace, etc.)
        content = content.strip()

        # Create document object
        doc = {
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, str(file_path.name))),  # Consistent UUID based on filename
            "content": content,
            "title": title,
            "url": f"/docs/content/{file_path.stem}",  # URL for the content
            "language": "en",  # Default to English
            "created_at": str(file_path.stat().st_ctime),
            "updated_at": str(file_path.stat().st_mtime)
        }

        documents.append(doc)

    return documents


class ContentDocumentationEmbedder:
    """
    Service to embed content documentation and store it in Qdrant vector database
    """
    def __init__(self):
        load_dotenv()

        # Initialize Cohere client
        self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

        # Initialize Qdrant client with proper connection for cloud instances
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")

        # Check if this is a Qdrant Cloud instance
        if "https://" in qdrant_url and ".qdrant.io" in qdrant_url:
            # For Qdrant Cloud, extract the host from the URL
            parsed_url = urlparse(qdrant_url)
            host = parsed_url.netloc
            # For Qdrant Cloud, use the host parameter with https=True
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
            self.qdrant_client.get_collection(self.collection_name)
            print(f"Collection {self.collection_name} already exists")
        except:
            # Collection doesn't exist, create it
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1024,  # Cohere embeddings are typically 1024-dimensional
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")

    def embed_documentation(self, documents):
        """
        Embed content documentation and store in Qdrant

        Args:
            documents: List of dictionaries with 'id', 'content', 'title', 'url', 'language'
        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        try:
            if not documents:
                logger.info("No documents to embed")
                return

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
                # Use the ID from the document (which is already a proper UUID)
                point_id = doc['id']

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

            logger.info(f"Successfully embedded and stored {len(documents)} content documents")

        except Exception as e:
            logger.error(f"Error embedding documentation: {str(e)}")
            raise


def main():
    """
    Main function to extract content documentation and embed it into Qdrant
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting content documentation embedding process...")

    # Extract documentation content from the content directory
    content_dir = "../content"  # Relative to the embedding directory
    logger.info(f"Extracting content documentation from {content_dir}")

    documents = extract_content_docs(content_dir)
    logger.info(f"Extracted {len(documents)} content files")

    if not documents:
        logger.warning("No content files found to embed")
        return

    # Initialize the content documentation embedder
    logger.info("Initializing content documentation embedder...")
    embedder = ContentDocumentationEmbedder()

    # Initialize the collection
    logger.info("Initializing Qdrant collection...")
    embedder.initialize_collection()

    # Embed and store all content documentation
    logger.info("Starting embedding process...")
    embedder.embed_documentation(documents)

    logger.info("Content documentation embedding process completed successfully!")


if __name__ == "__main__":
    main()