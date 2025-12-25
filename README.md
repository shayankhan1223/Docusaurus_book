# Docusaurus Documentation with OpenAI Integration

This project implements a Docusaurus-based documentation site with AI-powered features using OpenAI Agents SDK, Cohere embeddings, and Qdrant vector database.

## Features

- **Professional Landing Page**: Beautiful, tech-related landing page with a comfortable, professional feel
- **Intuitive Navigation**: Left side shows documentation topics, right side has theme button, language options, and GitHub link
- **AI-Powered Chatbot**: Floating ChatKit UI chatbot that answers documentation questions using OpenAI Agents
- **Text Selection Assistance**: Select any text on the page to get AI-powered explanations
- **Semantic Search**: Intelligent search functionality with context understanding
- **Multilingual Support**: Available in English and Urdu as specified
- **Anonymous Access**: No authentication required for documentation access

## Architecture

### Frontend
- **Docusaurus**: For documentation site generation
- **React Components**: Custom components for chatbot and text selection
- **ChatKit UI**: Floating chat interface for documentation assistance
- **Text Selection Handler**: Context menu for selected text explanations

### Backend
- **FastAPI**: Backend API server
- **OpenAI Agents SDK**: For AI processing and responses
- **Cohere**: For generating document embeddings
- **Qdrant Vector Database**: For semantic search and document storage
- **Guardrail Agent**: Validates queries are documentation-related
- **Main Agent**: Processes documentation queries with AI

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/chatbot/query` - Chatbot query processing
- `POST /api/text/explain` - Text selection explanation
- `POST /api/search/search` - Semantic search functionality
- `POST /api/search/search-with-routing` - Search with intelligent routing

## Setup

### Prerequisites
- Node.js (v18+)
- Python (v3.8+)
- Access to OpenAI API
- Access to Cohere API
- Qdrant Vector Database instance

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn src.main:app --reload
```

## Configuration

Environment variables needed in `.env` file:
- `OPENAI_API_KEY` - Your OpenAI API key
- `COHERE_API_KEY` - Your Cohere API key
- `QDRANT_URL` - URL for your Qdrant instance
- `QDRANT_API_KEY` - API key for Qdrant (if required)

## Embedding Service

The embedding service processes documentation content and stores it in the vector database:
- Located in the `embedding/` directory
- Uses Cohere for generating embeddings
- Sends embedded data to Qdrant Vector Database
- Runs once to initialize the documentation database

## Privacy and Security

- Anonymous access without authentication
- Minimal data storage (only necessary for functionality)
- No personal data collection
- Rate limiting to prevent abuse
- Query validation to ensure documentation relevance

## Development

The project follows a multi-service architecture:
1. **Frontend Service**: Docusaurus documentation site
2. **Backend Service**: FastAPI with OpenAI integration
3. **Embedding Service**: One-time content processing

## Testing

APIs can be tested using the provided endpoints. The frontend includes:
- Chatbot functionality
- Text selection assistance
- Semantic search capabilities
- Multilingual support

## Deployment

The frontend can be built and deployed as static files:
```bash
npm run build
npm run serve
```

The backend service should be deployed separately with appropriate environment variables.