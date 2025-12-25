# Quickstart: Docusaurus Documentation with OpenAI Integration

## Prerequisites
- Node.js 18+
- Python 3.11+
- Docker (for Qdrant vector database)
- OpenAI API key
- Cohere API key

## Setup Instructions

### 1. Clone and Initialize Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Set up Environment Variables
Create `.env` files in appropriate directories:

**Backend (.env):**
```
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key
```

### 3. Start Vector Database
```bash
# Using Docker
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Or using Docker Compose
docker-compose up -d qdrant
```

### 4. Install Dependencies

**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend/docusaurus
npm install
```

**Embedding Service:**
```bash
cd embedding
pip install -r requirements.txt
```

### 5. Run Initial Embedding Process
```bash
cd embedding
python src/embedder.py
```

### 6. Start Services

**Backend:**
```bash
cd backend
python -m src.main
```

**Frontend:**
```bash
cd frontend/docusaurus
npm start
```

## API Endpoints

### Backend API
- `POST /api/agents/query` - Submit query to AI agent
- `GET /api/docs/search` - Search documentation
- `POST /api/vector/search` - Semantic search in vector database

### Frontend
- `http://localhost:3000` - Documentation site
- Floating chatbot available on all pages
- Text selection Q&A available on documentation pages

## Testing
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend/docusaurus
npm test

# E2E tests
cd frontend/docusaurus
npx playwright test
```

## Configuration
- Documentation content is in `frontend/docusaurus/docs/`
- Translation files in `frontend/docusaurus/i18n/`
- API settings in `.env` files
- Docusaurus configuration in `frontend/docusaurus/docusaurus.config.js`