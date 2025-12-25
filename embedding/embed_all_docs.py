import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)

from extract_docs import extract_docs_content
from embed_docs import DocumentationEmbedder


def main():
    """
    Main function to extract ALL Docusaurus documentation and embed it into Qdrant
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting ALL Docusaurus documentation embedding process...")

    # Extract documentation content from frontend/docs (this will include all subdirectories)
    docs_dir = "../frontend/docs"  # Relative to the embedding directory
    logger.info(f"Extracting documentation from {docs_dir}")

    documents = extract_docs_content(docs_dir)
    logger.info(f"Extracted {len(documents)} documentation files total")

    if not documents:
        logger.warning("No documentation files found to embed")
        return

    # Initialize the documentation embedder
    logger.info("Initializing documentation embedder...")
    embedder = DocumentationEmbedder()

    # Initialize the collection
    logger.info("Initializing Qdrant collection...")
    embedder.initialize_collection()

    # Embed and store all documentation
    logger.info("Starting embedding process for all documentation...")
    embedder.embed_documentation(documents)

    logger.info("ALL Docusaurus documentation embedding process completed successfully!")


if __name__ == "__main__":
    main()