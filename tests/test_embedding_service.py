import pytest
from unittest.mock import Mock, patch
import numpy as np
from services.embedding_service import EmbeddingService
from models.exceptions import EmbeddingError
from models.cache_metrics import CacheMetrics

@pytest.fixture
def mock_metrics():
    return Mock(spec=CacheMetrics)

@pytest.fixture
def mock_text_embedding():
    mock = Mock()
    mock.embed.return_value = [np.array([0.1] * 384)]
    return mock

@pytest.fixture
def embedding_service(mock_metrics):
    with patch('fastembed.TextEmbedding', return_value=Mock()) as mock_text_embedding:
        mock_text_embedding.return_value.embed.return_value = [np.array([0.1] * 384)]
        service = EmbeddingService(metrics=mock_metrics)
        return service

def test_get_embedding(embedding_service):
    embedding = embedding_service.get_embedding("test text")
    assert isinstance(embedding, list)
    assert len(embedding) == 384

def test_empty_text(embedding_service):
    with pytest.raises(EmbeddingError):
        embedding_service.get_embedding("")

def test_none_text(embedding_service):
    with pytest.raises(EmbeddingError):
        embedding_service.get_embedding(None)

def test_get_batch_embeddings(embedding_service):
    texts = ["text1", "text2"]
    embeddings = embedding_service.get_batch_embeddings(texts)
    assert len(embeddings) == 2
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) == 384 for emb in embeddings)

def test_batch_embeddings_empty_list(embedding_service):
    with pytest.raises(EmbeddingError):
        embedding_service.get_batch_embeddings([])
