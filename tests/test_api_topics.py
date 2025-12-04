"""Tests for topics API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.app import app


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_has_status(self, client):
        """Health response should have status field."""
        response = client.get("/api/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_has_cache_status(self, client):
        """Health response should have cache_complete field."""
        response = client.get("/api/health")
        data = response.json()
        assert "cache_complete" in data


class TestStatusEndpoint:
    """Tests for status endpoint."""

    def test_status_returns_200(self, client):
        """Status endpoint should return 200."""
        response = client.get("/api/status")
        assert response.status_code == 200

    def test_status_has_required_fields(self, client):
        """Status response should have all required fields."""
        response = client.get("/api/status")
        data = response.json()
        assert "complete" in data
        assert "dictionary" in data
        assert "corpus" in data
        assert "tokenized" in data
        assert "coherence" in data
        assert "models" in data
        assert "distributions" in data
        assert "projections" in data


class TestCoherenceEndpoint:
    """Tests for coherence scores endpoint."""

    def test_coherence_returns_503_when_not_cached(self, client):
        """Should return 503 when cache is empty."""
        with patch("backend.api.topics.load_coherence_scores", return_value=None):
            response = client.get("/api/topics/coherence")
            assert response.status_code == 503

    def test_coherence_returns_data_when_cached(self, client):
        """Should return coherence data when cached."""
        mock_scores = {2: 0.35, 3: 0.42, 4: 0.38}

        with patch("backend.api.topics.load_coherence_scores", return_value=mock_scores):
            response = client.get("/api/topics/coherence")
            assert response.status_code == 200
            data = response.json()
            assert "topic_counts" in data
            assert "coherence" in data
            assert "optimal_topics" in data
            assert data["optimal_topics"] == 3  # Highest coherence


class TestTopicWordsEndpoint:
    """Tests for topic words endpoint."""

    def test_invalid_n_topics_returns_400(self, client):
        """Should return 400 for invalid n_topics."""
        response = client.get("/api/topics/1/words")  # Below minimum
        assert response.status_code == 400

        response = client.get("/api/topics/25/words")  # Above maximum
        assert response.status_code == 400

    def test_topic_words_returns_503_when_not_cached(self, client):
        """Should return 503 when model not cached."""
        with patch("backend.api.topics.load_lda_model", return_value=None):
            response = client.get("/api/topics/5/words")
            assert response.status_code == 503

    def test_topic_words_returns_data_when_cached(self, client):
        """Should return topic words when model cached."""
        mock_model = MagicMock()
        mock_model.show_topic.return_value = [("word1", 0.1), ("word2", 0.08)]

        with patch("backend.api.topics.load_lda_model", return_value=mock_model):
            response = client.get("/api/topics/5/words?num_words=2")
            assert response.status_code == 200
            data = response.json()
            assert "n_topics" in data
            assert "topics" in data
            assert data["n_topics"] == 5
