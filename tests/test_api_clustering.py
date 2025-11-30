"""Tests for clustering API endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from backend.app import app


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_distribution():
    """Mock document-topic distribution."""
    np.random.seed(42)
    return np.random.dirichlet([0.5, 0.5, 0.5], size=100)


class TestClusteringEndpoint:
    """Tests for clustering endpoint."""

    def test_clustering_returns_503_when_not_cached(self, client):
        """Should return 503 when distribution not cached."""
        with patch("backend.api.clustering.load_doc_topic_distribution", return_value=None):
            response = client.post(
                "/api/clustering",
                json={"n_topics": 5, "n_clusters": 3}
            )
            assert response.status_code == 503

    def test_clustering_returns_data_when_cached(self, client, mock_distribution):
        """Should return clustering results when distribution cached."""
        with patch("backend.api.clustering.load_doc_topic_distribution", return_value=mock_distribution):
            response = client.post(
                "/api/clustering",
                json={"n_topics": 5, "n_clusters": 3}
            )
            assert response.status_code == 200
            data = response.json()
            assert "labels" in data
            assert "silhouette" in data
            assert "inertia" in data
            assert "cluster_sizes" in data
            assert len(data["labels"]) == 100
            assert sum(data["cluster_sizes"]) == 100

    def test_clustering_validates_input(self, client):
        """Should validate n_topics and n_clusters."""
        response = client.post(
            "/api/clustering",
            json={"n_topics": 1, "n_clusters": 3}  # n_topics too low
        )
        assert response.status_code == 422

        response = client.post(
            "/api/clustering",
            json={"n_topics": 5, "n_clusters": 1}  # n_clusters too low
        )
        assert response.status_code == 422


class TestClusterMetricsEndpoint:
    """Tests for cluster metrics endpoint."""

    def test_metrics_returns_503_when_not_cached(self, client):
        """Should return 503 when distribution not cached."""
        with patch("backend.api.clustering.load_doc_topic_distribution", return_value=None):
            response = client.get("/api/clustering/metrics/5")
            assert response.status_code == 503

    def test_metrics_returns_data_when_cached(self, client, mock_distribution):
        """Should return metrics for all cluster counts."""
        with patch("backend.api.clustering.load_doc_topic_distribution", return_value=mock_distribution):
            response = client.get("/api/clustering/metrics/5?min_clusters=2&max_clusters=5")
            assert response.status_code == 200
            data = response.json()
            assert "cluster_counts" in data
            assert "silhouette_scores" in data
            assert "inertia_scores" in data
            assert "elbow_point" in data
            assert data["cluster_counts"] == [2, 3, 4, 5]

    def test_metrics_validates_n_topics(self, client):
        """Should validate n_topics."""
        response = client.get("/api/clustering/metrics/1")
        assert response.status_code == 400

        response = client.get("/api/clustering/metrics/25")
        assert response.status_code == 400
