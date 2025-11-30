"""Tests for metrics module."""

import numpy as np
import pytest
from backend.core.metrics import (
    calculate_silhouette,
    find_elbow_point,
    compute_metrics_for_all_clusters,
)


class TestCalculateSilhouette:
    """Tests for silhouette score calculation."""

    def test_returns_float(self, sample_doc_topic_matrix):
        """Should return a float score."""
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        score = calculate_silhouette(sample_doc_topic_matrix, labels)
        assert isinstance(score, float)

    def test_score_in_valid_range(self, sample_doc_topic_matrix):
        """Score should be in [-1, 1]."""
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        score = calculate_silhouette(sample_doc_topic_matrix, labels)
        assert -1 <= score <= 1

    def test_returns_zero_for_single_cluster(self, sample_doc_topic_matrix):
        """Should return 0 for single cluster."""
        labels = np.zeros(len(sample_doc_topic_matrix), dtype=int)
        score = calculate_silhouette(sample_doc_topic_matrix, labels)
        assert score == 0.0


class TestFindElbowPoint:
    """Tests for elbow point detection."""

    def test_returns_int(self):
        """Should return an integer index."""
        values = [100, 50, 30, 20, 15, 12, 10, 9, 8]
        elbow = find_elbow_point(values)
        assert isinstance(elbow, int)

    def test_valid_index(self):
        """Should return valid index within range."""
        values = [100, 50, 30, 20, 15, 12, 10, 9, 8]
        elbow = find_elbow_point(values)
        assert 0 <= elbow < len(values)

    def test_finds_elbow_in_decreasing_curve(self):
        """Should find elbow in typical inertia curve."""
        # Typical elbow curve
        values = [1000, 500, 300, 200, 150, 130, 120, 115, 112, 110]
        elbow = find_elbow_point(values, decreasing=True)
        # Elbow should be around index 3-5
        assert 2 <= elbow <= 6

    def test_handles_short_list(self):
        """Should handle lists with few elements."""
        values = [10, 5]
        elbow = find_elbow_point(values)
        assert elbow == 0


class TestComputeMetricsForAllClusters:
    """Tests for computing metrics across cluster range."""

    def test_returns_dict(self, sample_doc_topic_matrix):
        """Should return a dictionary with expected keys."""
        result = compute_metrics_for_all_clusters(
            sample_doc_topic_matrix,
            min_clusters=2,
            max_clusters=5
        )
        assert "cluster_counts" in result
        assert "silhouette_scores" in result
        assert "inertia_scores" in result
        assert "elbow_point" in result

    def test_correct_cluster_counts(self, sample_doc_topic_matrix):
        """Should have correct cluster count range."""
        result = compute_metrics_for_all_clusters(
            sample_doc_topic_matrix,
            min_clusters=2,
            max_clusters=5
        )
        assert result["cluster_counts"] == [2, 3, 4, 5]

    def test_has_score_for_each_cluster_count(self, sample_doc_topic_matrix):
        """Should have metrics for each cluster count."""
        result = compute_metrics_for_all_clusters(
            sample_doc_topic_matrix,
            min_clusters=2,
            max_clusters=5
        )
        assert len(result["silhouette_scores"]) == 4
        assert len(result["inertia_scores"]) == 4

    def test_inertia_decreases(self, sample_doc_topic_matrix):
        """Inertia should generally decrease with more clusters."""
        result = compute_metrics_for_all_clusters(
            sample_doc_topic_matrix,
            min_clusters=2,
            max_clusters=5
        )
        inertias = result["inertia_scores"]
        # First inertia should be >= last (more clusters = lower inertia)
        assert inertias[0] >= inertias[-1]
