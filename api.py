"""
FastAPI service for the semantic cache.
"""

import logging.config
from typing import Dict, List
import os

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, field_validator, ConfigDict

from config.settings import (
    LOGGING_CONFIG, 
    CACHE_SIMILARITY_THRESHOLD, 
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    CORS_ALLOW_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_HEADERS,
    RATE_LIMIT
)
from services.cache_service import CacheService
from services.embedding_service import EmbeddingService
from services.openai_service import OpenAIService
from models.cache_metrics import CacheMetrics

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize FastAPI app and services
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS.split(","),
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS.split(","),
    allow_headers=CORS_ALLOW_HEADERS.split(",") if CORS_ALLOW_HEADERS != "*" else ["*"]
)

# Configure rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

metrics = CacheMetrics()
embedding_service = EmbeddingService(metrics)
openai_service = OpenAIService(OPENAI_API_KEY, OPENAI_CHAT_MODEL, metrics)

# Cache service configuration
DB_PATH = "cache.db"
COLLECTION_NAME = "semantic_cache"
CACHE_TTL = 3600  # 1 hour in seconds

cache_service = CacheService(
    db_path=DB_PATH,
    collection_name=COLLECTION_NAME,
    similarity_threshold=CACHE_SIMILARITY_THRESHOLD,
    cache_ttl=CACHE_TTL,
    metrics=metrics,
    embedding_service=embedding_service,
    openai_service=openai_service
)

class Query(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    query: str

    @field_validator('query')
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v:
            raise ValueError('Query must not be empty')
        return v

class BatchQuery(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    queries: List[str]

class ModelUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    model: str

class TTLUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    ttl: int

    @field_validator('ttl')
    @classmethod
    def ttl_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError('TTL must be a positive integer')
        return v

@app.get("/health")
@limiter.limit(RATE_LIMIT)
async def health_check(request: Request):
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/model")
@limiter.limit(RATE_LIMIT)
async def get_model(request: Request):
    """Get the current OpenAI model being used."""
    try:
        model = openai_service.get_current_model()
        return {"model": model}
    except Exception as e:
        logger.error(f"Error getting current model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/model")
@limiter.limit(RATE_LIMIT)
async def update_model(model_update: ModelUpdate, request: Request):
    """Update the OpenAI model to use for completions."""
    try:
        success = openai_service.set_model(model_update.model)
        if success:
            return {"status": "success", "model": model_update.model}
        else:
            raise HTTPException(status_code=400, detail="Failed to update model")
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
@limiter.limit(RATE_LIMIT)
async def query(query: Query, request: Request) -> Dict:
    """
    Process a query through the semantic cache.
    Returns the cached result if a similar query exists, otherwise processes the new query.
    """
    try:
        if not query.query.strip():
            raise HTTPException(status_code=400, detail="Query must not be empty")
            
        logger.info(f"Received query: {query.query}")
        result = cache_service.get(query.query)
        logger.info(f"Query processed successfully. Cache hit: {result['cache_hit']}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
@limiter.limit(RATE_LIMIT)
async def batch_query(batch: BatchQuery, request: Request) -> List[Dict]:
    """
    Process multiple queries through the semantic cache in a single request.
    Returns a list of results, one for each query in the batch.
    """
    try:
        logger.info(f"Received batch request with {len(batch.queries)} queries")
        results = []
        for query in batch.queries:
            try:
                result = cache_service.get(query)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query '{query}': {str(e)}")
                results.append({"error": str(e), "query": query})
        
        logger.info("Batch processing completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error processing batch request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/contents")
@limiter.limit(RATE_LIMIT)
async def get_cache_contents(request: Request):
    """Get all entries from the cache."""
    try:
        logger.info("Retrieving cache contents")
        contents = cache_service.get_contents()
        logger.info(f"Retrieved {len(contents)} cache entries")
        return contents
    except Exception as e:
        logger.error(f"Error retrieving cache contents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/analyze")
@limiter.limit(RATE_LIMIT)
async def analyze_cache(request: Request):
    """Analyze the cache contents using OpenAI."""
    try:
        logger.info("Analyzing cache contents")
        contents = cache_service.get_contents()
        analysis = openai_service.analyze_cache_contents(contents)
        
        if not analysis["success"]:
            # If it's specifically the "no entries" case, return 404
            if analysis.get("error") == "No entries found in cache":
                logger.info("No entries found in cache")
                return analysis
            # For other errors, raise HTTP exception
            raise HTTPException(status_code=500, detail=analysis["error"])
            
        logger.info("Cache analysis completed successfully")
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing cache contents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/stats")
@limiter.limit(RATE_LIMIT)
async def get_cache_stats(request: Request) -> Dict:
    """Get current cache statistics."""
    try:
        logger.info("Retrieving cache stats")
        stats = cache_service.get_metrics()
        logger.info("Cache stats retrieved successfully")
        return stats
    except Exception as e:
        logger.error(f"Error retrieving cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/clear")
@limiter.limit(RATE_LIMIT)
async def clear_cache(request: Request) -> Dict:
    """Clear the cache."""
    try:
        logger.info("Clearing cache")
        cache_service.clear_cache()
        logger.info("Cache cleared successfully")
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/ttl")
@limiter.limit(RATE_LIMIT)
async def get_cache_ttl(request: Request) -> Dict:
    """Get the current cache TTL in seconds."""
    try:
        logger.info("Retrieving cache TTL")
        ttl = cache_service.get_ttl()
        logger.info(f"Cache TTL retrieved successfully: {ttl} seconds")
        return {"ttl": ttl}
    except Exception as e:
        logger.error(f"Error retrieving cache TTL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/cache/ttl")
@limiter.limit(RATE_LIMIT)
async def update_cache_ttl(ttl_update: TTLUpdate, request: Request) -> Dict:
    """Update the cache TTL value in seconds."""
    try:
        logger.info(f"Updating cache TTL to {ttl_update.ttl} seconds")
        new_ttl = cache_service.set_ttl(ttl_update.ttl)
        logger.info("Cache TTL updated successfully")
        return {"ttl": new_ttl}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating cache TTL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
