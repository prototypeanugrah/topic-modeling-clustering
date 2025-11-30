"""Tests for visualization API endpoints."""

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
def mock_projection():
    """Mock UMAP projection."""
    np.random.seed(42)
    return np.random.randn(100, 2)


@pytest.fixture
def mock_distribution():
    """Mock document-topic distribution."""
    np.random.seed(42)
    return np.random.dirichlet([0.5, 0.5, 0.5], size=100)


class TestVisualizationEndpoint:
    """Tests for visualization endpoint."""

    def test_visualization_returns_503_when_not_cached(self, client):
        """Should return 503 when projection not cached."""
        with patch("backend.api.visualization.load_umap_projection", return_value=None):
            response = client.get("/api/visualization/5")
            assert response.status_code == 503

    def test_visualization_returns_data_when_cached(self, client, mock_projection):
        """Should return projection data when cached."""
        with patch("backend.api.visualization.load_umap_projection", return_value=mock_projection):
            response = client.get("/api/visualization/5")
            assert response.status_code == 200
            data = response.json()
            assert "n_topics" in data
            assert "projections" in data
            assert "document_ids" in data
            assert len(data["projections"]) == 100
            assert len(data["projections"][0]) == 2  # 2D coordinates

    def test_visualization_validates_n_topics(self, client):
        """Should validate n_topics."""
        response = client.get("/api/visualization/1")
        assert response.status_code == 400

        response = client.get("/api/visualization/25")
        assert response.status_code == 400


class TestClusteredVisualizationEndpoint:
    """Tests for clustered visualization endpoint."""

    def test_clustered_returns_503_when_not_cached(self, client):
        """Should return 503 when data not cached."""
        with patch("backend.api.visualization.load_umap_projection", return_value=None):
            response = client.post(
                "/api/visualization/clustered",
                json={"n_topics": 5, "n_clusters": 3}
            )
            assert response.status_code == 503

    def test_clustered_returns_data_when_cached(
        self, client, mock_projection, mock_distribution
    ):
        """Should return clustered visualization when cached."""
        with patch("backend.api.visualization.load_umap_projection", return_value=mock_projection):
            with patch("backend.api.visualization.load_doc_topic_distribution", return_value=mock_distribution):
                response = client.post(
                    "/api/visualization/clustered",
                    json={"n_topics": 5, "n_clusters": 3}
                )
                assert response.status_code == 200
                data = response.json()
                assert "projections" in data
                assert "cluster_labels" in data
                assert len(data["projections"]) == 100
                assert len(data["cluster_labels"]) == 100
                # All labels should be 0, 1, or 2
                assert all(0 <= label < 3 for label in data["cluster_labels"])
