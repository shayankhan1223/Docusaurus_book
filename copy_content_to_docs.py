import os
import shutil
from pathlib import Path
import re


def add_frontmatter(content, filename):
    """
    Add frontmatter to markdown content
    """
    # Extract title from the first heading or use filename
    title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
    if title_match:
        title = title_match.group(1)
    else:
        # Use filename without extension as title
        title = Path(filename).stem.replace('_', ' ').replace('-', ' ').title()

    # Create frontmatter
    frontmatter = f"""---
sidebar_label: {title}
sidebar_position: {hash(filename) % 1000}  # Simple position based on filename hash
title: {title}
---

"""

    # Remove any existing frontmatter (content between --- markers at the beginning)
    content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)

    # Combine frontmatter and content
    return frontmatter + content


def copy_content_to_docs(content_dir, docs_dir):
    """
    Copy content from content directory to docs directory with frontmatter
    """
    content_path = Path(content_dir)
    docs_path = Path(docs_dir)

    # Create docs directory if it doesn't exist
    docs_path.mkdir(parents=True, exist_ok=True)

    copied_files = []

    for file_path in content_path.glob("*.md"):
        # Read the original content
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Add frontmatter
        content_with_frontmatter = add_frontmatter(original_content, file_path.name)

        # Create new file path in docs directory
        new_file_path = docs_path / file_path.name

        # Write the content with frontmatter to the new location
        with open(new_file_path, 'w', encoding='utf-8') as f:
            f.write(content_with_frontmatter)

        print(f"Copied {file_path.name} to {new_file_path} with frontmatter")
        copied_files.append(new_file_path)

    return copied_files


def main():
    content_dir = "content"  # Relative to current directory
    docs_dir = "frontend/docs/content"  # Create a content subdirectory in docs

    print(f"Copying content from {content_dir} to {docs_dir} with frontmatter...")

    copied_files = copy_content_to_docs(content_dir, docs_dir)

    print(f"\nSuccessfully copied {len(copied_files)} files:")
    for file in copied_files:
        print(f"  - {file}")

    print(f"\nFiles have been copied to {docs_dir} with appropriate frontmatter.")


if __name__ == "__main__":
    main()