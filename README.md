# Semantic Cache Chat API (FastAPI/LanceDB/FastEmbed)

A Python-based semantic cache implementation for chat interactions using OpenAI's GPT model and LanceDB for efficient response retrieval. This version uses FastEmbed for generating embeddings. Extremely simplistic implementation for testing and demonstration purposes.

> **⚠️ IMPORTANT CAVEAT**: This code is intended for testing and demonstration purposes only. It is NOT suitable for production use. The implementation is deliberately simplified to illustrate core concepts and should not be deployed in production environments without substantial hardening, security improvements, and performance optimizations.

## Table of Contents
- [Core Features](#core-features)
- [Technical Architecture](#technical-architecture)
- [Getting Started](#getting-started)
- [Usage Guide](#usage-guide)
- [Monitoring and Observability](#monitoring-and-observability)
- [Development Guide](#development-guide)
- [References](#references)
- [License](#license)

## Core Features

- Semantic caching with LanceDB vector database
- FastEmbed for efficient embedding generation (384-dimension vectors)
- FastAPI-based REST API with OpenAPI documentation
- Streamlit-based interactive UI
- Batch query support for efficient processing
- Configurable cache TTL (Time To Live)
- Request rate limiting and CORS support
- Comprehensive error handling and validation
- Performance monitoring and metrics
- Stress testing capabilities

### Cache Features
- Automatic cache expiration with timestamp-based TTL
- Configurable similarity threshold
- Efficient batch processing
- Persistent storage with UUID-based entries
- Automatic cache cleanup
- Vector-based similarity search (384-dimension FastEmbed vectors)

### API Features
- RESTful endpoints for chat interactions
- Batch processing support
- Cache management endpoints
- Health checks and metrics
- Model configuration endpoints
- Rate limiting on all endpoints
- Comprehensive CORS support
- Input validation and error handling

### UI Features
- Interactive chat interface
- Real-time cache analysis
- Cache metrics visualization
- Model configuration
- Cache management controls

## Technical Architecture

### LanceDB Integration
- **Serverless Architecture**: Runs directly in application process
- **Vector Operations**: Built on Apache Arrow and Lance format
- **ACID Compliance**: Full transaction support
- **Vector Search**: Fast ANN search using IVF-PQ indexing
- **Hybrid Search**: Combines vector similarity with metadata filtering
- **Storage Format**: UUID-based entries with timestamps for TTL
- **Vector Dimension**: 384-dimension vectors from FastEmbed
- **TTL Implementation**: Timestamp-based expiration checking

### FastEmbed Integration
- **Local Embedding Generation**: 4x faster than comparable solutions
- **Memory Efficient**: Optimized model loading and quantization
- **Batch Processing**: Parallel text processing
- **Multi-Language Support**: 100+ languages supported
- **Cost Effective**: No external API calls needed
- **Vector Dimension**: Generates 384-dimension embeddings
- **Performance**: Optimized for both single and batch operations

### FastAPI Backend
- **High Performance**: On par with NodeJS and Go
- **Modern Features**: Built on Python type hints
- **Auto Documentation**: Interactive OpenAPI/Swagger UI
- **Validation**: Automatic request/response validation using Pydantic models
- **Standards-Based**: OpenAPI and JSON Schema compliant
- **Rate Limiting**: Configurable per-endpoint rate limiting
- **CORS Support**: Configurable Cross-Origin Resource Sharing
- **Error Handling**: Comprehensive error handling with detailed responses

### Streamlit Frontend
- **Interactive UI**: Real-time chat interface with immediate response display
- **Cache Analysis**: Visual representation of cache contents and patterns
- **Metrics Dashboard**: Real-time performance metrics visualization
- **Cache Management**: Interactive controls for cache operations
- **Model Configuration**: Easy model selection and configuration
- **Visual Feedback**: Clear indication of cache hits with custom styling
- **Environment Management**: Simplified OpenAI API key configuration

## Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- Required packages (see requirements.txt)
- Docker and Docker Compose (for containerized deployment)

### Installation

1. Clone and setup virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate (Unix/macOS)
source venv/bin/activate
# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. Configure environment:
```bash
# Copy environment template
cp env.example .env

# Edit .env with your settings:
OPENAI_API_KEY="sk-<your-openai-api-key>"
GPT_MODEL="gpt-3.5-turbo"
SIMILARITY_THRESHOLD=0.95
LLMFASTEMBEDCACHE_COLLECTION_NAME="semantic_cache_fastembed"
TOKENIZERS_PARALLELISM=false
CACHE_TTL=3600  # Time to live in seconds (default 1 hour)

# CORS Configuration
CORS_ALLOW_ORIGINS="http://localhost:3000,http://localhost:8000"  # Comma-separated list of allowed origins
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS="GET,POST,PUT,DELETE"
CORS_ALLOW_HEADERS="*"

# Rate Limiting Configuration
RATE_LIMIT="100/minute"
```

### Docker Deployment

#### API Service
To run the semantic cache API as a standalone service:

1. Build the container:
```bash
docker compose build
```

2. Start the service:
```bash
# Run in foreground
docker compose up

# Or run in background
docker compose up -d
```

The API service will be available at `http://localhost:8000`.

#### Streamlit UI
To run the Streamlit UI with the API service:

1. Navigate to the streamlit directory:
```bash
cd streamlit
```

2. Build and run the containers:
```bash
docker compose up --build
```

This will start both the API service and the Streamlit UI. The UI will be available at:
- API Service: http://localhost:8000
  - http://localhost:8000/docs (Swagger UI)
  - http://localhost:8000/redoc (ReDoc)
- Streamlit UI: http://localhost:8501
- OTEL Collector: http://localhost:4317

To stop all services:
```bash
docker compose down
```

#### Combined UI and API Container
To run both the API server and Streamlit UI in a single container:

1. Build the combined container:
```bash
docker build -t semantic-ui-api-image -f Dockerfile-UI-API .
```

2. Run the container:
```bash
docker run -p 8000:8000 -p 8501:8501 -p 4317:4317 semantic-ui-api-image
```

This will start both services in a single container:
- API Server: http://localhost:8000
  - http://localhost:8000/docs (Swagger UI)
  - http://localhost:8000/redoc (ReDoc)
- Streamlit UI: http://localhost:8501
- OTEL Collector: http://localhost:4317

The combined container provides a more streamlined deployment option when you want to run both services on the same host. It uses a single Python environment and startup script to manage both services efficiently.

## Development Usage Guide

### Running the Application

#### API Server
Start the API server:
```bash
uvicorn api:app --reload
```
Access API documentation at:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

#### Streamlit UI
1. Navigate to the streamlit directory:
```bash
cd streamlit
```

2. Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

The Streamlit UI will be available at http://localhost:8501

Features available in the UI:
- Interactive chat interface
- Cache analysis visualization
- Performance metrics display
- Cache management controls
- Model configuration
- Visual indicators for cache hits

### Chat Client Commands
- Type messages to chat with AI
- `clear cache`: Clear semantic cache
- `metrics`: View cache performance metrics
- `batch`: Enter batch mode for multiple queries
- `analyze`: Analyze cache contents
- `stats`: View cache statistics
- `quit`: Exit client

### API Endpoints

#### Health Check
- `GET /health`: Check API health status
  - Response: `{"status": "healthy"}`
  - Rate Limit: 100 requests per minute

#### Chat Operations
- `POST /query`: Process a single query through the semantic cache
  - Request body: `{"query": "your question here"}`
  - Response: 
    ```json
    {
      "response": "AI response",
      "cache_hit": true/false,
      "similarity": 0.95,
      "latency": 0.123
    }
    ```
  - Rate Limit: 100 requests per minute
  - Validation: Query must not be empty

- `POST /batch`: Process multiple queries in a single request
  - Request body: `{"queries": ["question1", "question2", ...]}`
  - Response: Array of query results
  - Rate Limit: 100 requests per minute
  - Validation: Each query must not be empty

#### Cache Management
- `GET /cache/contents`: Get all entries from the cache
  - Response: Array of cache entries with query, response, and timestamp
  - Rate Limit: 100 requests per minute

- `GET /cache/analyze`: Analyze cache contents using OpenAI
  - Response: Analysis of cached queries and patterns
  - Rate Limit: 100 requests per minute

- `GET /cache/stats`: Get current cache statistics
  - Response: Cache performance metrics
  - Rate Limit: 100 requests per minute

- `POST /cache/clear`: Clear the cache
  - Response: `{"status": "success", "message": "Cache cleared"}`
  - Rate Limit: 100 requests per minute

- `GET /cache/ttl`: Get current cache TTL
  - Response: `{"ttl": 3600}`
  - Rate Limit: 100 requests per minute

- `PUT /cache/ttl`: Update cache TTL
  - Request body: `{"ttl": 7200}`
  - Response: `{"ttl": 7200}`
  - Rate Limit: 100 requests per minute
  - Validation: TTL must be positive integer

#### Model Management
- `GET /model`: Get current OpenAI model
  - Response: `{"model": "gpt-3.5-turbo"}`
  - Rate Limit: 100 requests per minute

- `PUT /model`: Update OpenAI model
  - Request body: `{"model": "gpt-4"}`
  - Response: `{"status": "success", "model": "gpt-4"}`
  - Rate Limit: 100 requests per minute
  - Validation: Model must be valid OpenAI model

### Error Responses
All endpoints return standardized error responses:
```json
{
  "detail": "Error message describing what went wrong"
}
```

Common HTTP status codes:
- 400: Bad Request (validation errors)
- 429: Too Many Requests (rate limit exceeded)
- 500: Internal Server Error

### Batch Processing
Optimize performance by processing multiple queries:
```python
POST /batch
{
  "queries": [
    "What is Python?",
    "How does garbage collection work?",
    "What are decorators?"
  ]
}
```

## Monitoring and Observability

### Logging System
- **Log Files**:
  - `cache_metrics.log`: Cache operations
  - `api_metrics.log`: API requests
  - `stress_test.log`: Test results
- **Log Levels**: INFO, WARNING, ERROR, DEBUG

### Performance Metrics
- Cache hit/miss rates
- Response latencies
- Request volumes
- Error rates
- Resource utilization

### OpenTelemetry Integration
Currently implemented for cache service tracing with plans for expansion:
- Distributed tracing for cache operations
- Span attributes for query details
- Error tracking and reporting
- Performance metrics collection

Setup OpenTelemetry:
```bash
# Install packages
pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi

# Configure in .env
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=semantic-cache-api

# Start collector
docker run -p 4317:4317 otel/opentelemetry-collector
```

## Development Guide

### Testing
The project uses Pytest for comprehensive unit testing across all major components. Tests are located in the `tests/` directory and follow standard Pytest conventions.

#### Test Coverage
- **API Tests** (`test_api.py`):
  - Health check endpoint
  - Chat completion endpoints
  - Input validation
  - Error handling
  - Metrics endpoint functionality

- **Cache Service Tests** (`test_cache_service.py`):
  - Cache initialization
  - Search functionality
  - Cache addition/clearing
  - TTL management
  - Metrics collection
  - Error handling for edge cases

- **Embedding Service Tests** (`test_embedding_service.py`):
  - Single text embedding generation
  - Batch embedding processing
  - Error handling for invalid inputs
  - Vector dimensionality verification
  - Batch processing edge cases

Run tests using:
```bash
pytest
```

For verbose output with test names:
```bash
pytest -v
```

For coverage report:
```bash
pytest --cov=.
```

Run stress tests to evaluate performance:
```bash
python stress_test.py [--url URL] [--requests N] [--concurrency M]
```

Test results are saved to `stress_test_results.json` with:
- Success/failure rates
- Response time statistics
- Cache hit rates
- Error distribution
- Requests per second

This will generate `stress_test_visualization.png` containing four plots:
- Request success/failure distribution
- Error type distribution
- Response time metrics
- Performance rates (success rate and cache hit rate)

### Error Handling
The system handles:
- API connection issues
- OpenAI API errors
- Cache operation failures
- Environment configuration errors
- Rate limiting violations
- Input validation errors

### Virtual Environment Tips
- Activate environment in each new terminal
- Use `pip list` to check packages
- Update requirements: `pip freeze > requirements.txt`

## References
- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [FastEmbed Documentation](https://qdrant.github.io/fastembed/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Pytest Documentation](https://docs.pytest.org/)

## License
See the LICENSE file for details.
