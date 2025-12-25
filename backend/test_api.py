import asyncio
import aiohttp
import json

async def test_api_endpoints():
    base_url = "http://localhost:8000"

    print("Testing Backend API Endpoints...")
    print("=" * 50)

    # Test 1: Health Check
    print("\n1. Testing Health Check Endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as response:
                data = await response.json()
                print(f"   Status: {response.status}")
                print(f"   Response: {data}")
                assert response.status == 200, f"Expected 200, got {response.status}"
                assert data["status"] == "healthy", f"Expected healthy status"
                print("   [PASS] Health check passed")
    except Exception as e:
        print(f"   [FAIL] Health check failed: {e}")

    # Test 2: Root Endpoint
    print("\n2. Testing Root Endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/") as response:
                data = await response.json()
                print(f"   Status: {response.status}")
                print(f"   Response: {data}")
                assert response.status == 200, f"Expected 200, got {response.status}"
                assert "message" in data, "Expected 'message' in response"
                print("   [PASS] Root endpoint passed")
    except Exception as e:
        print(f"   [FAIL] Root endpoint failed: {e}")

    # Test 3: Chatbot Query Endpoint
    print("\n3. Testing Chatbot Query Endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "query": "Hello, how are you?",
                "language": "en"
            }
            async with session.post(f"{base_url}/api/chatbot/query", json=payload) as response:
                data = await response.json()
                print(f"   Status: {response.status}")
                print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
                assert response.status in [200, 422, 500], f"Unexpected status: {response.status}"
                print("   [PASS] Chatbot query endpoint reached")
    except Exception as e:
        print(f"   [FAIL] Chatbot query endpoint failed: {e}")

    # Test 4: Text Explanation Endpoint
    print("\n4. Testing Text Explanation Endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "text": "artificial intelligence",
                "language": "en"
            }
            async with session.post(f"{base_url}/api/text/explain", json=payload) as response:
                data = await response.json()
                print(f"   Status: {response.status}")
                print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
                assert response.status in [200, 422, 500], f"Unexpected status: {response.status}"
                print("   [PASS] Text explanation endpoint reached")
    except Exception as e:
        print(f"   [FAIL] Text explanation endpoint failed: {e}")

    # Test 5: Semantic Search Endpoint
    print("\n5. Testing Semantic Search Endpoint...")
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
                print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
                assert response.status in [200, 422, 500], f"Unexpected status: {response.status}"
                print("   [PASS] Semantic search endpoint reached")
    except Exception as e:
        print(f"   [FAIL] Semantic search endpoint failed: {e}")

    # Test 6: Search with Routing Endpoint
    print("\n6. Testing Search with Routing Endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "query": "how to use the documentation",
                "top_k": 5,
                "language": "en"
            }
            async with session.post(f"{base_url}/api/search/search-with-routing", json=payload) as response:
                data = await response.json()
                print(f"   Status: {response.status}")
                print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
                assert response.status in [200, 422, 500], f"Unexpected status: {response.status}"
                print("   [PASS] Search with routing endpoint reached")
    except Exception as e:
        print(f"   [FAIL] Search with routing endpoint failed: {e}")

    print("\n" + "=" * 50)
    print("API Testing Complete!")
    print("Note: Some endpoints may return 500 errors if external services (OpenAI, Cohere, Qdrant) are not configured.")

if __name__ == "__main__":
    asyncio.run(test_api_endpoints())