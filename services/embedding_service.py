from typing import List
from fastembed import TextEmbedding
from models.exceptions import EmbeddingError
from models.cache_metrics import CacheMetrics
import logging
import time
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class EmbeddingService:
    def __init__(self, metrics: CacheMetrics):
        self.metrics = metrics
        try:
            self.embedding_model = TextEmbedding()
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            self.metrics.record_error("embedding")
            raise EmbeddingError(f"Failed to initialize embedding model: {str(e)}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using FastEmbed.
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            List of floats representing the embedding
            
        Raises:
            EmbeddingError: If text is None, empty, or embedding generation fails
        """
        if text is None:
            raise EmbeddingError("Input text cannot be None")
        if not text.strip():
            raise EmbeddingError("Input text cannot be empty")
            
        with tracer.start_as_current_span("get_embedding") as span:
            span.set_attribute("text_length", len(text))
            start_time = time.time()
            try:
                embeddings = list(self.embedding_model.embed([text]))
                latency = time.time() - start_time
                span.set_attribute("latency", latency)
                logger.debug(f"Embedding generated in {latency:.3f}s")
                return embeddings[0].tolist()
            except Exception as e:
                span.record_exception(e)
                self.metrics.record_error("embedding")
                raise EmbeddingError(f"Failed to generate embedding: {str(e)}")

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch using FastEmbed.
        
        Args:
            texts: List of input texts to generate embeddings for
            
        Returns:
            List of embeddings, where each embedding is a list of floats
            
        Raises:
            EmbeddingError: If texts list is empty or contains invalid texts
        """
        if not texts:
            raise EmbeddingError("Input texts list cannot be empty")
            
        # Validate individual texts
        for text in texts:
            if text is None:
                raise EmbeddingError("Input texts cannot contain None values")
            if not text.strip():
                raise EmbeddingError("Input texts cannot contain empty strings")
                
        start_time = time.time()
        try:
            embeddings = list(self.embedding_model.embed(texts))
            logger.debug(f"Batch embeddings generated in {time.time() - start_time:.3f}s")
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            self.metrics.record_error("embedding")
            raise EmbeddingError(f"Failed to generate batch embeddings: {str(e)}")
