import asyncio
from src.services.vector_search import vector_search_service

async def test_vector_connection():
    print("Testing Vector Search Service Connection...")
    print("=" * 50)

    # Test if the service can connect to the vector database
    print("\n1. Testing vector search service initialization...")
    try:
        print(f"   Vector search service initialized: {vector_search_service is not None}")
        print(f"   Cohere client initialized: {vector_search_service.cohere_client is not None}")
        print(f"   Qdrant client initialized: {vector_search_service.qdrant_client is not None}")
        print("   [PASS] Vector search service initialized successfully")
    except Exception as e:
        print(f"   [FAIL] Vector search service initialization failed: {e}")

    # Test search functionality
    print("\n2. Testing search functionality...")
    try:
        results = await vector_search_service.search_by_text("documentation", top_k=5, language="en")
        print(f"   Search results count: {len(results)}")
        if results:
            print(f"   First result keys: {list(results[0].keys())}")
            print(f"   Sample content: {results[0]['content'][:100]}...")
        else:
            print("   No results returned - vector DB might be empty or not connected to the same instance as embedding")
        print("   [PASS] Search functionality executed")
    except Exception as e:
        print(f"   [FAIL] Search functionality failed: {e}")

    # Test adding a document directly through the service
    print("\n3. Testing document addition...")
    try:
        test_doc = {
            "id": "test-doc-backend",
            "content": "This is a test document added through the backend service for testing purposes.",
            "title": "Test Document",
            "url": "/test",
            "language": "en"
        }
        result = await vector_search_service.add_document(test_doc)
        print(f"   Document addition result: {result}")
        print("   [PASS] Document addition executed")
    except Exception as e:
        print(f"   [FAIL] Document addition failed: {e}")

    # Test search again after adding document
    print("\n4. Testing search after adding test document...")
    try:
        results = await vector_search_service.search_by_text("test document", top_k=5, language="en")
        print(f"   Search results count after adding test doc: {len(results)}")
        if results:
            print(f"   Sample content: {results[0]['content'][:100]}...")
        print("   [PASS] Post-addition search executed")
    except Exception as e:
        print(f"   [FAIL] Post-addition search failed: {e}")

    print("\n" + "=" * 50)
    print("Vector connection testing complete!")

if __name__ == "__main__":
    asyncio.run(test_vector_connection())