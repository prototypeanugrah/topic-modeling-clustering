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
        assert "corpus_train" in data
        assert "corpus_test" in data
        assert "tokenized_train" in data
        assert "tokenized_test" in data
        assert "coherence_val" in data
        assert "coherence_test" in data
        assert "perplexity_val" in data
        assert "perplexity_test" in data
        assert "models" in data
        assert "distributions_train" in data
        assert "distributions_test" in data
        assert "projections_train" in data
        assert "projections_test" in data


class TestCoherenceEndpoint:
    """Tests for coherence scores endpoint."""

    def test_coherence_returns_503_when_not_cached(self, client):
        """Should return 503 when cache is empty."""
        with patch("backend.api.topics.load_coherence_scores", return_value=None):
            response = client.get("/api/topics/coherence")
            assert response.status_code == 503

    def test_coherence_returns_data_when_cached(self, client):
        """Should return coherence data when cached."""
        mock_val_scores = {2: 0.32, 3: 0.40, 4: 0.36}
        mock_test_scores = {2: 0.35, 3: 0.42, 4: 0.38}
        mock_perplexity_val = {2: 100.0, 3: 95.0, 4: 98.0}
        mock_perplexity_test = {2: 105.0, 3: 92.0, 4: 96.0}

        with patch("backend.api.topics.load_coherence_scores") as mock_coh, \
             patch("backend.api.topics.load_perplexity_scores") as mock_perp:
            # Mock returns based on split argument
            mock_coh.side_effect = lambda split: mock_val_scores if split == "val" else mock_test_scores
            mock_perp.side_effect = lambda split: mock_perplexity_val if split == "val" else mock_perplexity_test

            response = client.get("/api/topics/coherence")
            assert response.status_code == 200
            data = response.json()
            assert "topic_counts" in data
            assert "coherence_val" in data
            assert "coherence_test" in data
            assert "perplexity_val" in data
            assert "perplexity_test" in data
            assert "optimal_topics" in data
            assert data["optimal_topics"] == 3  # Highest test coherence


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
