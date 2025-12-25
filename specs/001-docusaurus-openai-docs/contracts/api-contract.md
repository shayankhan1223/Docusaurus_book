# API Contract: Docusaurus Documentation with OpenAI Integration

## Query API
```
POST /api/agents/query
```

### Request
```json
{
  "query": "string - The user's question/query",
  "session_id": "string - Session identifier",
  "language": "string - Language code (optional, defaults to 'en')"
}
```

### Response
```json
{
  "id": "string - Response identifier",
  "query": "string - The original query",
  "response": "string - AI-generated response",
  "is_validated": "boolean - Whether query passed validation",
  "validation_reason": "string - Reason if query was rejected",
  "sources": "array - List of documentation sources used",
  "timestamp": "datetime - Response timestamp"
}
```

### Error Responses
- `400 Bad Request`: Invalid request format
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Processing error

## Search API
```
GET /api/docs/search
```

### Query Parameters
- `q`: Search query string
- `lang`: Language code (optional, defaults to 'en')
- `limit`: Number of results (optional, defaults to 10)

### Response
```json
{
  "results": [
    {
      "id": "string - Documentation content ID",
      "title": "string - Title of the documentation",
      "content_preview": "string - Preview of the content",
      "url": "string - URL to the documentation page",
      "relevance_score": "number - Relevance score (0-1)"
    }
  ],
  "total": "number - Total number of results",
  "query": "string - The original search query"
}
```

## Vector Search API
```
POST /api/vector/search
```

### Request
```json
{
  "query": "string - Search query",
  "top_k": "number - Number of results to return (optional, defaults to 5)",
  "language": "string - Language code (optional, defaults to 'en')"
}
```

### Response
```json
{
  "results": [
    {
      "id": "string - Content ID",
      "content": "string - Content text",
      "metadata": "object - Additional metadata",
      "similarity": "number - Similarity score (0-1)"
    }
  ],
  "query_embedding": "array - Vector representation of the query"
}
```

## Documentation Content API
```
GET /api/docs/{id}
```

### Response
```json
{
  "id": "string - Content ID",
  "title": "string - Title of the documentation",
  "content": "string - Full content",
  "language": "string - Language code",
  "category": "string - Category/section",
  "tags": "array - Relevant tags",
  "created_at": "datetime - Creation timestamp",
  "updated_at": "datetime - Last update timestamp"
}
```