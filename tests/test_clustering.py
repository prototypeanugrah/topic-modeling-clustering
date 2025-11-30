"""Tests for clustering module."""

import numpy as np
import pytest
from backend.core.clustering import (
    perform_kmeans,
    get_cluster_sizes,
    ClusteringResult,
)


class TestPerformKmeans:
    """Tests for K-Means clustering."""

    def test_returns_clustering_result(self, sample_doc_topic_matrix):
        """Should return ClusteringResult namedtuple."""
        result = perform_kmeans(sample_doc_topic_matrix, n_clusters=3)
        assert isinstance(result, ClusteringResult)

    def test_has_correct_number_of_labels(self, sample_doc_topic_matrix):
        """Should have one label per sample."""
        result = perform_kmeans(sample_doc_topic_matrix, n_clusters=3)
        assert len(result.labels) == len(sample_doc_topic_matrix)

    def test_labels_in_valid_range(self, sample_doc_topic_matrix):
        """Labels should be in [0, n_clusters)."""
        n_clusters = 3
        result = perform_kmeans(sample_doc_topic_matrix, n_clusters=n_clusters)
        assert all(0 <= label < n_clusters for label in result.labels)

    def test_has_correct_number_of_centers(self, sample_doc_topic_matrix):
        """Should have n_clusters centers."""
        n_clusters = 3
        result = perform_kmeans(sample_doc_topic_matrix, n_clusters=n_clusters)
        assert result.centers.shape[0] == n_clusters

    def test_centers_have_correct_dimensions(self, sample_doc_topic_matrix):
        """Centers should have same dimensions as input features."""
        result = perform_kmeans(sample_doc_topic_matrix, n_clusters=3)
        assert result.centers.shape[1] == sample_doc_topic_matrix.shape[1]

    def test_has_inertia(self, sample_doc_topic_matrix):
        """Should have inertia (sum of squared distances)."""
        result = perform_kmeans(sample_doc_topic_matrix, n_clusters=3)
        assert result.inertia >= 0

    def test_reproducible_with_random_state(self, sample_doc_topic_matrix):
        """Results should be reproducible with same random_state."""
        result1 = perform_kmeans(sample_doc_topic_matrix, n_clusters=3, random_state=42)
        result2 = perform_kmeans(sample_doc_topic_matrix, n_clusters=3, random_state=42)
        np.testing.assert_array_equal(result1.labels, result2.labels)

    def test_different_cluster_counts(self, sample_doc_topic_matrix):
        """Should work with different numbers of clusters."""
        for n_clusters in [2, 3, 5]:
            result = perform_kmeans(sample_doc_topic_matrix, n_clusters=n_clusters)
            assert len(np.unique(result.labels)) <= n_clusters


class TestGetClusterSizes:
    """Tests for cluster size calculation."""

    def test_returns_list(self):
        """Should return a list of sizes."""
        labels = np.array([0, 0, 1, 1, 1, 2])
        sizes = get_cluster_sizes(labels)
        assert isinstance(sizes, list)

    def test_correct_sizes(self):
        """Should count cluster sizes correctly."""
        labels = np.array([0, 0, 1, 1, 1, 2])
        sizes = get_cluster_sizes(labels)
        assert sizes[0] == 2
        assert sizes[1] == 3
        assert sizes[2] == 1

    def test_sums_to_total(self):
        """Sizes should sum to total samples."""
        labels = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
        sizes = get_cluster_sizes(labels)
        assert sum(sizes) == len(labels)
