import os
import re
from pathlib import Path
import uuid
from typing import List, Dict


def extract_docs_content(docs_dir: str) -> List[Dict]:
    """
    Extract content from Docusaurus documentation files

    Args:
        docs_dir: Path to the docs directory

    Returns:
        List of dictionaries containing document information
    """
    documents = []

    docs_path = Path(docs_dir)

    # Walk through all subdirectories in docs
    for root, dirs, files in os.walk(docs_path):
        for file in files:
            if file.endswith('.md'):
                file_path = Path(root) / file
                relative_path = file_path.relative_to(docs_path)

                # Read the markdown file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract title from the first heading
                title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
                title = title_match.group(1) if title_match else file_path.stem

                # Remove frontmatter if present (content between --- markers)
                content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)

                # Clean up the content (remove extra whitespace, etc.)
                content = content.strip()

                # Create document object
                doc = {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, str(relative_path))),
                    "content": content,
                    "title": title,
                    "url": f"/docs/{relative_path.parent}/{relative_path.stem}" if relative_path.parent != Path('.') else f"/docs/{relative_path.stem}",
                    "language": "en",  # Default to English
                    "created_at": str(file_path.stat().st_ctime),
                    "updated_at": str(file_path.stat().st_mtime)
                }

                documents.append(doc)

    return documents


def main():
    """Main function to extract documentation content"""
    docs_dir = "../frontend/docs"  # Relative to the embedding directory

    print("Extracting documentation content...")
    documents = extract_docs_content(docs_dir)

    print(f"Found {len(documents)} documentation files")

    # Print a sample of the extracted content
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents as samples
        print(f"\nDocument {i+1}:")
        print(f"  ID: {doc['id']}")
        print(f"  Title: {doc['title']}")
        print(f"  URL: {doc['url']}")
        print(f"  Content preview: {doc['content'][:200]}...")

    return documents


if __name__ == "__main__":
    docs = main()