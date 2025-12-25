import os
import sys
import uuid
import cohere
from qdrant_client.http import models
from dotenv import load_dotenv
import logging
from pathlib import Path
import re
from src.db.vector_db import vector_db  # Use the working VectorDB from backend


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
    Main function to extract content documentation and embed it into Qdrant using backend's connection
    """
    # Load environment like the backend does
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting content documentation embedding process using backend connection...")

    # Extract documentation content from the content directory (relative to project root)
    content_dir = "../content"  # Go up from backend directory to project root, then to content
    logger.info(f"Extracting content documentation from {content_dir}")

    documents = extract_content_docs(content_dir)
    logger.info(f"Extracted {len(documents)} content files")

    if not documents:
        logger.warning("No content files found to embed")
        return

    # Use the VectorDB instance from the backend which we know works
    logger.info("Using backend's VectorDB connection...")

    # Check if collection exists (handling validation errors)
    collection_exists = False
    try:
        vector_db.client.get_collection(vector_db.collection_name)
        logger.info(f"Collection {vector_db.collection_name} already exists")
        collection_exists = True
    except Exception as e:
        error_msg = str(e).lower()
        if "doesn't exist" in error_msg or "not found" in error_msg or "404" in str(e):
            logger.info(f"Collection {vector_db.collection_name} doesn't exist, creating it...")
            vector_db.client.create_collection(
                collection_name=vector_db.collection_name,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)  # Cohere v3 model produces 1024-dim vectors
            )
            logger.info(f"Collection {vector_db.collection_name} created successfully")
        else:
            # If it's a validation error or other issue, assume collection exists
            logger.info(f"Collection {vector_db.collection_name} exists (access issue: {type(e).__name__})")
            collection_exists = True

    # Clear the collection if it exists
    if collection_exists:
        logger.info("Clearing existing collection to start fresh...")
        try:
            # Get all point IDs in the collection
            scroll_result, _ = vector_db.client.scroll(
                collection_name=vector_db.collection_name,
                limit=10000  # Adjust based on your needs
            )

            # Extract IDs of all points
            point_ids = [point.id for point in scroll_result]

            if point_ids:
                # Delete all points
                vector_db.client.delete(
                    collection_name=vector_db.collection_name,
                    points_selector=models.PointIdsList(
                        points=point_ids
                    )
                )
                logger.info(f"Deleted {len(point_ids)} existing points from collection '{vector_db.collection_name}'")
            else:
                logger.info(f"Collection '{vector_db.collection_name}' is already empty")
        except Exception as scroll_error:
            logger.warning(f"Could not clear collection: {scroll_error}")

    # Initialize Cohere client using the same approach as the backend
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

    # Prepare points for Qdrant using the same method as VectorDB
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

    # Upload points to Qdrant using the same method as VectorDB
    logger.info(f"Uploading {len(points)} documents to Qdrant...")
    vector_db.client.upsert(
        collection_name=vector_db.collection_name,
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

        search_results = vector_db.client.search(
            collection_name=vector_db.collection_name,
            query_vector=query_embedding,
            limit=1
        )

        logger.info(f"Search test successful. Found {len(search_results)} results.")
        logger.info("Content embedding process completed successfully!")


if __name__ == "__main__":
    main()