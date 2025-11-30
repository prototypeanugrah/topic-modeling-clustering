"""Evaluation metrics for clustering."""

import numpy as np
from sklearn.metrics import silhouette_score
from typing import NamedTuple

from backend.config import MIN_CLUSTERS, MAX_CLUSTERS


class ClusterMetrics(NamedTuple):
    """Container for cluster evaluation metrics."""
    silhouette: float
    inertia: float


def calculate_silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate silhouette score for clustering.

    Args:
        data: Feature matrix
        labels: Cluster labels

    Returns:
        Silhouette score (-1 to 1, higher is better)
    """
    # Need at least 2 clusters and 2 samples per cluster
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(data):
        return 0.0

    return silhouette_score(data, labels)


def find_elbow_point(values: list[float], decreasing: bool = True) -> int:
    """
    Find the elbow point in a curve using the kneedle algorithm (simplified).

    Args:
        values: List of metric values
        decreasing: Whether the curve is expected to decrease

    Returns:
        Index of the elbow point
    """
    if len(values) < 3:
        return 0

    # Normalize the values
    y = np.array(values)
    x = np.arange(len(y))

    # Normalize to 0-1 range
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
    x_norm = x / (len(x) - 1)

    if decreasing:
        # For decreasing curves, we want the point furthest from the line
        # connecting the first and last points
        line_start = np.array([x_norm[0], y_norm[0]])
        line_end = np.array([x_norm[-1], y_norm[-1]])
    else:
        line_start = np.array([x_norm[0], y_norm[0]])
        line_end = np.array([x_norm[-1], y_norm[-1]])

    # Calculate distance from each point to the line
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-10:
        return len(values) // 2

    distances = []
    for i in range(len(y)):
        point = np.array([x_norm[i], y_norm[i]])
        # Distance from point to line (perpendicular distance formula)
        # Using 2D cross product: |a x b| = a[0]*b[1] - a[1]*b[0]
        vec_to_point = line_start - point
        d = np.abs(line_vec[0] * vec_to_point[1] - line_vec[1] * vec_to_point[0]) / line_len
        distances.append(d)

    return int(np.argmax(distances))


def compute_metrics_for_all_clusters(
    data: np.ndarray,
    min_clusters: int = MIN_CLUSTERS,
    max_clusters: int = MAX_CLUSTERS,
) -> dict[str, list]:
    """
    Compute clustering metrics for a range of cluster counts.

    Args:
        data: Feature matrix
        min_clusters: Minimum number of clusters
        max_clusters: Maximum number of clusters

    Returns:
        Dictionary with cluster_counts, silhouette_scores, inertia_scores, and elbow_point
    """
    from backend.core.clustering import perform_kmeans

    cluster_counts = list(range(min_clusters, max_clusters + 1))
    silhouette_scores = []
    inertia_scores = []

    for n_clusters in cluster_counts:
        result = perform_kmeans(data, n_clusters)
        silhouette = calculate_silhouette(data, result.labels)
        silhouette_scores.append(silhouette)
        inertia_scores.append(result.inertia)

    # Find elbow point for inertia
    elbow_idx = find_elbow_point(inertia_scores, decreasing=True)
    elbow_point = cluster_counts[elbow_idx]

    return {
        "cluster_counts": cluster_counts,
        "silhouette_scores": silhouette_scores,
        "inertia_scores": inertia_scores,
        "elbow_point": elbow_point,
    }
