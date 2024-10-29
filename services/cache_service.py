import uuid
import time
from typing import List, Optional, Tuple, Dict, Any
import lancedb
from models.exceptions import DatabaseError
from models.cache_metrics import CacheMetrics
from services.embedding_service import EmbeddingService
from services.openai_service import OpenAIService
import logging
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class CacheService:
    def __init__(self, 
                 db_path: str, 
                 collection_name: str, 
                 similarity_threshold: float,
                 cache_ttl: int,
                 metrics: CacheMetrics,
                 embedding_service: EmbeddingService,
                 openai_service: OpenAIService):
        self.similarity_threshold = similarity_threshold
        self.cache_ttl = cache_ttl
        self.metrics = metrics
        self.embedding_service = embedding_service
        self.openai_service = openai_service
        self.db_path = db_path
        self.collection_name = collection_name
        
        try:
            self.db = lancedb.connect(db_path)
            self.initialize_table()
        except Exception as e:
            logger.error(f"Database Error: Failed to initialize LanceDB: {str(e)}")
            raise DatabaseError(f"Failed to initialize database: {str(e)}")

    def get(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the cache. Returns cached result if similar query exists,
        otherwise processes the new query.

        Args:
            query: The input query string

        Returns:
            Dictionary containing response and cache information
        """
        start_time = time.time()
        try:
            # Search cache for similar queries
            cached_response, similarity = self.search_cache(query)
            
            if cached_response:
                self.metrics.record_request(cache_hit=True)
                return {
                    "response": cached_response,
                    "cache_hit": True,
                    "similarity": similarity,
                    "latency": time.time() - start_time
                }
            
            # If no cache hit, get response from OpenAI
            query_vector = self.embedding_service.get_embedding(query)
            response = self.openai_service.get_completion(query)
            
            # Add to cache
            self.add_to_cache(query, response, query_vector)
            
            self.metrics.record_request(cache_hit=False)
            return {
                "response": response,
                "cache_hit": False,
                "latency": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def initialize_table(self):
        """Initialize or reset the LanceDB table"""
        try:
            self.table = self.db.create_table(
                self.collection_name,
                data=[{
                    "id": str(uuid.uuid4()),
                    "vector": [0.0] * 384,  # FastEmbed dimension
                    "query": "",
                    "response": "",
                    "timestamp": time.time()
                }],
                mode="overwrite"
            )
        except Exception as e:
            raise DatabaseError(f"Failed to initialize table: {str(e)}")

    def search_cache(self, query: str, top_k: int = 5) -> Tuple[Optional[str], Optional[float]]:
        with tracer.start_as_current_span("search_cache") as span:
            span.set_attribute("query_length", len(query))
            span.set_attribute("top_k", top_k)
            start_time = time.time()
            try:
                query_vector = self.embedding_service.get_embedding(query)
                current_time = time.time()
                results = self.table.search(query_vector).where(f"timestamp >= {current_time - self.cache_ttl}").limit(top_k).to_list()
                latency = time.time() - start_time
                span.set_attribute("latency", latency)
                logger.debug(f"Cache search completed in {latency:.3f}s")
                
                if results and results[0]["_distance"] <= 1 - self.similarity_threshold:
                    span.set_attribute("cache_hit", True)
                    return results[0]["response"], 1 - results[0]["_distance"]
                span.set_attribute("cache_hit", False)
                return None, None
            except Exception as e:
                span.record_exception(e)
                raise DatabaseError(f"Cache search failed: {str(e)}")

    def add_to_cache(self, query: str, response: str, embedding: List[float]) -> None:
        """Add a query-response pair and its embedding to the cache"""
        start_time = time.time()
        try:
            self.table.add([{
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "query": query,
                "response": response,
                "timestamp": time.time()
            }])
            logger.debug(f"Cache entry added in {time.time() - start_time:.3f}s")
        except Exception as e:
            raise DatabaseError(f"Failed to add entry to cache: {str(e)}")

    def clear_cache(self) -> None:
        """Clear the entire cache by recreating the table"""
        try:
            self.initialize_table()
            logger.info("Cache cleared successfully")
        except Exception as e:
            raise DatabaseError(f"Failed to clear cache: {str(e)}")

    def get_metrics(self) -> Dict:
        """Get current cache metrics"""
        try:
            return self.metrics.get_metrics()
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            raise

    def get_contents(self) -> List[Dict[str, Any]]:
        """Get all entries from the cache"""
        try:
            contents = []
            for record in self.table.to_pandas().to_dict('records'):
                if record['query'] and record['response']:  # Skip empty initialization entry
                    contents.append({
                        "query": record['query'],
                        "response": record['response'],
                        "timestamp": record['timestamp']
                    })
            return contents
        except Exception as e:
            logger.error(f"Error getting cache contents: {str(e)}")
            raise DatabaseError(f"Failed to get cache contents: {str(e)}")

    def get_ttl(self) -> int:
        """Get the current cache TTL in seconds"""
        return self.cache_ttl

    def set_ttl(self, ttl: int) -> int:
        """
        Set a new cache TTL in seconds.
        
        Args:
            ttl: New TTL value in seconds. Must be positive.
            
        Returns:
            The new TTL value
            
        Raises:
            ValueError: If TTL is not a positive integer
        """
        if ttl <= 0:
            raise ValueError("TTL must be a positive integer")
        self.cache_ttl = ttl
        return self.cache_ttl
