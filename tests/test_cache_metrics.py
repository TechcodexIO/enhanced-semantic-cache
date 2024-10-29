import pytest
from models.cache_metrics import CacheMetrics

def test_cache_metrics_initialization():
    metrics = CacheMetrics()
    assert metrics.total_requests == 0
    assert metrics.cache_hits == 0
    assert metrics.cache_misses == 0

def test_record_cache_hit():
    metrics = CacheMetrics()
    metrics.record_request(cache_hit=True)
    
    assert metrics.total_requests == 1
    assert metrics.cache_hits == 1
    assert metrics.cache_misses == 0

def test_record_cache_miss():
    metrics = CacheMetrics()
    metrics.record_request(cache_hit=False)
    
    assert metrics.total_requests == 1
    assert metrics.cache_hits == 0
    assert metrics.cache_misses == 1

def test_get_metrics():
    metrics = CacheMetrics()
    metrics.record_request(cache_hit=True)
    metrics.record_request(cache_hit=False)
    
    result = metrics.get_metrics()
    assert result == {
        "total_requests": 2,
        "cache_hits": 1,
        "cache_misses": 1
    }

def test_multiple_requests():
    metrics = CacheMetrics()
    
    # Record multiple hits and misses
    for _ in range(3):
        metrics.record_request(cache_hit=True)
    for _ in range(2):
        metrics.record_request(cache_hit=False)
    
    result = metrics.get_metrics()
    assert result == {
        "total_requests": 5,
        "cache_hits": 3,
        "cache_misses": 2
    }
