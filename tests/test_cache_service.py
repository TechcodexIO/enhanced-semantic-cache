import pytest
from unittest.mock import Mock, patch
import numpy as np
from services.cache_service import CacheService
from models.exceptions import DatabaseError
from models.cache_metrics import CacheMetrics

@pytest.fixture
def mock_embedding_service():
    mock = Mock()
    mock.get_embedding.return_value = np.array([0.1] * 384)
    return mock

@pytest.fixture
def mock_openai_service():
    mock = Mock()
    mock.get_completion.return_value = "Test response"
    return mock

@pytest.fixture
def mock_metrics():
    return Mock(spec=CacheMetrics)

@pytest.fixture
def mock_lancedb():
    mock_table = Mock()
    mock_table.search.return_value.where.return_value.limit.return_value.to_list.return_value = []
    
    mock_db = Mock()
    mock_db.create_table.return_value = mock_table
    
    with patch('lancedb.connect', return_value=mock_db):
        yield mock_db

@pytest.fixture
def cache_service(mock_embedding_service, mock_openai_service, mock_metrics, mock_lancedb):
    service = CacheService(
        db_path="test.db",
        collection_name="test_collection",
        similarity_threshold=0.8,
        cache_ttl=3600,
        metrics=mock_metrics,
        embedding_service=mock_embedding_service,
        openai_service=mock_openai_service
    )
    return service

def test_initialization(cache_service):
    assert cache_service.similarity_threshold == 0.8
    assert cache_service.cache_ttl == 3600

def test_search_cache(cache_service, mock_lancedb):
    # Test cache miss
    result, score = cache_service.search_cache("test query")
    assert result is None
    assert score is None

def test_add_to_cache(cache_service):
    query = "test query"
    response = "test response"
    embedding = [0.1] * 384
    
    cache_service.add_to_cache(query, response, embedding)
    assert cache_service.table.add.called

def test_clear_cache(cache_service):
    cache_service.clear_cache()
    assert cache_service.table is not None

def test_get_set_ttl(cache_service):
    new_ttl = 7200
    result = cache_service.set_ttl(new_ttl)
    assert result == new_ttl
    assert cache_service.get_ttl() == new_ttl

def test_invalid_ttl(cache_service):
    with pytest.raises(ValueError):
        cache_service.set_ttl(-1)

def test_get_metrics(cache_service, mock_metrics):
    cache_service.get_metrics()
    assert mock_metrics.get_metrics.called
