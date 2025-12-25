import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Environment variables loaded:")
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")
print(f"COHERE_API_KEY: {os.getenv('COHERE_API_KEY')}")
print(f"QDRANT_URL: {os.getenv('QDRANT_URL')}")
print(f"QDRANT_API_KEY: {os.getenv('QDRANT_API_KEY')}")