from typing import Dict
import logging
import logging.config
from config.settings import LOGGING_CONFIG

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class CacheMetrics:
    """Simple metrics tracker for cache operations"""
    
    def __init__(self):
        # Initialize basic counters
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def record_request(self, cache_hit: bool = False) -> None:
        """
        Record a cache request and whether it was a hit or miss
        
        Args:
            cache_hit: True if request was served from cache, False otherwise
        """
        try:
            self.total_requests += 1
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
        except Exception as e:
            logger.error(f"Failed to record request: {str(e)}")

    def get_metrics(self) -> Dict:
        """
        Get current cache metrics
        
        Returns:
            Dictionary containing:
            - total_requests: Total number of requests processed
            - cache_hits: Number of cache hits
            - cache_misses: Number of cache misses
        """
        try:
            return {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            # Return zeros if there's an error
            return {
                "total_requests": 0,
                "cache_hits": 0,
                "cache_misses": 0
            }
