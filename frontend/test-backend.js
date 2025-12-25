const testBackendIntegration = async () => {
  console.log('Testing backend integration...');

  try {
    // Test health check
    const healthResponse = await fetch('http://localhost:8000/health');
    const healthData = await healthResponse.json();
    console.log('Health check:', healthData);

    // Test chatbot endpoint
    const chatResponse = await fetch('http://localhost:8000/api/chatbot/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: 'Hello',
        language: 'en'
      }),
    });
    const chatData = await chatResponse.json();
    console.log('Chatbot response:', chatData);

    // Test semantic search endpoint
    const searchResponse = await fetch('http://localhost:8000/api/search/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: 'test',
        top_k: 5,
        language: 'en'
      }),
    });
    const searchData = await searchResponse.json();
    console.log('Search response:', searchData);

    console.log('All backend endpoints are accessible!');
  } catch (error) {
    console.error('Error testing backend integration:', error);
  }
};

// Run the test
testBackendIntegration();