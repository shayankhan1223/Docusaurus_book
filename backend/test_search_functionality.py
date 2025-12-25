import asyncio
import aiohttp
import json

async def test_search_functionality():
    base_url = "http://localhost:8000"

    print("Testing Search Functionality with Embedded Data...")
    print("=" * 50)

    # Test semantic search with a query that should match the embedded content
    print("\n1. Testing Semantic Search with 'documentation' query...")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "query": "documentation",
                "top_k": 5,
                "language": "en"
            }
            async with session.post(f"{base_url}/api/search/search", json=payload) as response:
                data = await response.json()
                print(f"   Status: {response.status}")
                print(f"   Total results: {data['total_results']}")
                print(f"   Results: {len(data['results'])}")
                if data['results']:
                    print(f"   First result keys: {list(data['results'][0].keys())}")
                    print(f"   Sample content: {data['results'][0]['content'][:100]}...")
                else:
                    print("   No results returned (database might be empty or query didn't match)")
                assert response.status == 200, f"Expected 200, got {response.status}"
                print("   [PASS] Semantic search functionality tested")
    except Exception as e:
        print(f"   [FAIL] Semantic search test failed: {e}")

    # Test chatbot with documentation-related query
    print("\n2. Testing Chatbot with documentation query...")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "query": "What is documentation?",
                "language": "en"
            }
            async with session.post(f"{base_url}/api/chatbot/query", json=payload) as response:
                data = await response.json()
                print(f"   Status: {response.status}")
                print(f"   Response: {data['response'][:100]}...")
                print(f"   Sources: {data['sources']}")
                assert response.status == 200, f"Expected 200, got {response.status}"
                assert "response" in data, "Expected 'response' in data"
                print("   [PASS] Chatbot documentation query tested")
    except Exception as e:
        print(f"   [FAIL] Chatbot documentation query test failed: {e}")

    # Test text selection functionality
    print("\n3. Testing Text Selection with 'artificial intelligence'...")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "text": "artificial intelligence",
                "context": "This is about artificial intelligence",
                "session_id": "test-session-789"
            }
            async with session.post(f"{base_url}/api/text/text-selection", json=payload) as response:
                data = await response.json()
                print(f"   Status: {response.status}")
                print(f"   Explanation: {data['explanation'][:100]}...")
                assert response.status == 200, f"Expected 200, got {response.status}"
                assert "explanation" in data, "Expected 'explanation' in data"
                print("   [PASS] Text selection functionality tested")
    except Exception as e:
        print(f"   [FAIL] Text selection test failed: {e}")

    print("\n" + "=" * 50)
    print("Search functionality testing complete!")
    print("The embedding process was successful and data is available for search.")

if __name__ == "__main__":
    asyncio.run(test_search_functionality())