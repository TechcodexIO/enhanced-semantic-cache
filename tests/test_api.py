import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from api import app
from models.cache_metrics import CacheMetrics

@pytest.fixture
def mock_cache_service():
    mock = Mock()
    mock.get.return_value = {
        "response": "Test response",
        "cache_hit": True
    }
    mock.get_metrics.return_value = {
        "cache_hits": 0,
        "cache_misses": 0,
        "total_requests": 0
    }
    mock.get_contents.return_value = [
        {"query": "test query", "response": "test response", "timestamp": "2023-01-01"}
    ]
    mock.clear_cache.return_value = None
    return mock

@pytest.fixture
def mock_openai_service():
    mock = Mock()
    mock.get_completion.return_value = "Test response"
    mock.get_current_model.return_value = "gpt-3.5-turbo"
    mock.set_model.return_value = True
    mock.analyze_cache_contents.return_value = {
        "success": True,
        "analysis": "Cache analysis results"
    }
    return mock

@pytest.fixture
def mock_embedding_service():
    mock = Mock()
    mock.get_embedding.return_value = [0.1] * 384
    return mock

@pytest.fixture
def client():
    # Create test client without mocks first
    test_client = TestClient(app)
    return test_client

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_chat_completion(client, mock_cache_service, mock_openai_service):
    with patch('api.cache_service', mock_cache_service), \
         patch('api.openai_service', mock_openai_service):
        response = client.post(
            "/query",
            json={"query": "Test prompt"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert isinstance(data["response"], str)
        assert "cache_hit" in data
        assert isinstance(data["cache_hit"], bool)

def test_chat_completion_no_prompt(client):
    response = client.post(
        "/query",
        json={}
    )
    assert response.status_code == 422

def test_chat_completion_empty_prompt(client):
    response = client.post(
        "/query",
        json={"query": ""}
    )
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data

def test_metrics(client, mock_cache_service):
    mock_metrics = {
        "cache_hits": 0,
        "cache_misses": 0,
        "total_requests": 0
    }
    mock_cache_service.get_metrics.return_value = mock_metrics
    
    with patch('api.cache_service', mock_cache_service):
        response = client.get("/cache/stats")
        
        assert response.status_code == 200
        assert response.json() == mock_metrics
        assert mock_cache_service.get_metrics.called

# New tests for missing endpoints

def test_get_model(client, mock_openai_service):
    with patch('api.openai_service', mock_openai_service):
        response = client.get("/model")
        assert response.status_code == 200
        assert response.json() == {"model": "gpt-3.5-turbo"}
        assert mock_openai_service.get_current_model.called

def test_get_model_error(client, mock_openai_service):
    mock_openai_service.get_current_model.side_effect = Exception("Test error")
    with patch('api.openai_service', mock_openai_service):
        response = client.get("/model")
        assert response.status_code == 500
        assert "detail" in response.json()

def test_update_model(client, mock_openai_service):
    with patch('api.openai_service', mock_openai_service):
        response = client.put("/model", json={"model": "gpt-4"})
        assert response.status_code == 200
        assert response.json() == {"status": "success", "model": "gpt-4"}
        mock_openai_service.set_model.assert_called_with("gpt-4")

def test_update_model_failure(client, mock_openai_service):
    mock_openai_service.set_model.side_effect = Exception("Invalid model")
    with patch('api.openai_service', mock_openai_service):
        response = client.put("/model", json={"model": "invalid-model"})
        assert response.status_code == 500
        assert "detail" in response.json()

def test_get_cache_contents(client, mock_cache_service):
    with patch('api.cache_service', mock_cache_service):
        response = client.get("/cache/contents")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert mock_cache_service.get_contents.called

def test_get_cache_contents_error(client, mock_cache_service):
    mock_cache_service.get_contents.side_effect = Exception("Test error")
    with patch('api.cache_service', mock_cache_service):
        response = client.get("/cache/contents")
        assert response.status_code == 500
        assert "detail" in response.json()

def test_analyze_cache(client, mock_openai_service, mock_cache_service):
    with patch('api.openai_service', mock_openai_service), \
         patch('api.cache_service', mock_cache_service):
        response = client.get("/cache/analyze")
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert "analysis" in response.json()

def test_analyze_cache_empty(client, mock_openai_service, mock_cache_service):
    mock_cache_service.get_contents.return_value = []
    mock_openai_service.analyze_cache_contents.return_value = {
        "success": False,
        "error": "No entries found in cache"
    }
    with patch('api.openai_service', mock_openai_service), \
         patch('api.cache_service', mock_cache_service):
        response = client.get("/cache/analyze")
        assert response.status_code == 200
        assert response.json()["success"] is False
        assert response.json()["error"] == "No entries found in cache"

def test_batch_query(client, mock_cache_service):
    with patch('api.cache_service', mock_cache_service):
        response = client.post(
            "/batch",
            json={"queries": ["Test prompt 1", "Test prompt 2"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert all("response" in item for item in data)

def test_batch_query_empty(client):
    response = client.post("/batch", json={"queries": []})
    assert response.status_code == 200
    assert response.json() == []

def test_batch_query_error(client, mock_cache_service):
    mock_cache_service.get.side_effect = Exception("Test error")
    with patch('api.cache_service', mock_cache_service):
        response = client.post(
            "/batch",
            json={"queries": ["Test prompt"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert "error" in data[0]

def test_clear_cache(client, mock_cache_service):
    with patch('api.cache_service', mock_cache_service):
        response = client.post("/cache/clear")
        assert response.status_code == 200
        assert response.json() == {"status": "success", "message": "Cache cleared"}
        assert mock_cache_service.clear_cache.called

def test_clear_cache_error(client, mock_cache_service):
    mock_cache_service.clear_cache.side_effect = Exception("Test error")
    with patch('api.cache_service', mock_cache_service):
        response = client.post("/cache/clear")
        assert response.status_code == 500
        assert "detail" in response.json()
