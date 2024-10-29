class CacheError(Exception):
    """Base exception for cache-related errors"""
    pass

class DatabaseError(CacheError):
    """Exception for database-related errors"""
    pass

class EmbeddingError(CacheError):
    """Exception for embedding generation errors"""
    pass
