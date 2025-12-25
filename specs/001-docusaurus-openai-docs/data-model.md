# Data Model: Docusaurus Documentation with OpenAI Integration

## Documentation Content Entity
- **Fields**:
  - id: string (unique identifier)
  - title: string (title of the documentation page)
  - content: string (full text content of the documentation)
  - language: string (language code, e.g., "en", "ur")
  - category: string (category/section of documentation)
  - tags: array<string> (relevant tags for search)
  - embedding: array<number> (vector representation for semantic search)
  - created_at: datetime (timestamp of creation)
  - updated_at: datetime (timestamp of last update)
- **Relationships**: None
- **Validation**: Content must not be empty, language must be in supported list (en, ur)
- **State transitions**: None

## User Query Entity
- **Fields**:
  - id: string (unique identifier)
  - query_text: string (the user's question/query)
  - language: string (detected language of query)
  - session_id: string (session identifier for context)
  - timestamp: datetime (when query was submitted)
  - response: string (AI-generated response)
  - is_validated: boolean (whether query passed validation)
  - validation_reason: string (reason if query was rejected)
- **Relationships**: None
- **Validation**: Query text must not be empty, length should be reasonable
- **State transitions**: None (queries are immutable once processed)

## Knowledge Base Entry Entity
- **Fields**:
  - id: string (unique identifier)
  - content_id: string (reference to documentation content)
  - embedding: array<number> (vector representation for semantic search)
  - metadata: object (additional metadata for search)
  - created_at: datetime (timestamp of creation)
- **Relationships**:
  - References Documentation Content (content_id → Documentation Content.id)
- **Validation**: Embedding must be properly formatted vector
- **State transitions**: None

## User Interaction Data Entity
- **Fields**:
  - id: string (unique identifier)
  - session_id: string (session identifier)
  - action_type: string (type of interaction: query, page_view, etc.)
  - page_url: string (URL of the page where interaction occurred)
  - timestamp: datetime (when interaction occurred)
- **Relationships**: None
- **Validation**: Minimal data collected as per privacy requirements
- **State transitions**: None

## API Contract Requirements

### Documentation Content API
- GET /api/docs/{id} - Retrieve specific documentation content
- GET /api/docs/search - Search documentation with semantic search
- POST /api/docs/query - Submit a query to the AI system

### AI Agent API
- POST /api/agents/query - Process user query through validation and response generation
- POST /api/agents/validate - Validate if query is documentation-related

### Vector Database API
- POST /api/vector/search - Perform semantic search in vector database
- POST /api/vector/add - Add new content to vector database (for initial indexing)