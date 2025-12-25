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


def main():
    """
    Main function to extract content documentation and embed it into Qdrant
    """
    # Load environment like the backend does
    load_dotenv()

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

    # Initialize Qdrant client using the same approach as backend VectorDB
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")

    # Check if this is a Qdrant Cloud instance
    if "https://" in qdrant_url and ".qdrant.io" in qdrant_url:
        # For Qdrant Cloud, extract the host from the URL
        parsed_url = urlparse(qdrant_url)
        host = parsed_url.netloc
        qdrant_client = QdrantClient(
            host=host,
            api_key=api_key,
            https=True  # Explicitly enable HTTPS for cloud instances
        )
    else:
        # For local instance or other configurations
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=api_key
        )

    collection_name = "documentation_content"

    # Initialize the collection
    logger.info("Initializing Qdrant collection...")
    try:
        # Try to get the collection to see if it exists
        qdrant_client.get_collection(collection_name)
        logger.info(f"Collection {collection_name} already exists")
    except Exception as e:
        # Collection doesn't exist, create it
        logger.info(f"Collection {collection_name} doesn't exist, creating it...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)  # Cohere v3 model produces 1024-dim vectors
        )
        logger.info(f"Collection {collection_name} created successfully")

    # Initialize Cohere client
    cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

    # Embed and store all content documentation
    logger.info("Starting embedding process...")

    if not documents:
        logger.info("No documents to embed")
        return

    # Extract content for embedding
    texts = [doc['content'] for doc in documents]

    # Generate embeddings using Cohere
    logger.info(f"Generating embeddings for {len(texts)} documents...")
    response = cohere_client.embed(
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
    logger.info(f"Uploading {len(points)} documents to Qdrant...")
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

    logger.info(f"Successfully embedded and stored {len(documents)} content documents in Qdrant!")

    # Test the connection by doing a simple search
    if documents:
        logger.info("Testing search functionality...")
        test_query = documents[0]['content'][:100]  # Use first 100 chars of first document
        query_response = cohere_client.embed(
            texts=[test_query],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        query_embedding = query_response.embeddings[0]

        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=1
        )

        logger.info(f"Search test successful. Found {len(search_results)} results.")


if __name__ == "__main__":
    main()