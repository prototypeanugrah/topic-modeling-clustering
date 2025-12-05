"""K-Means clustering utilities."""

import numpy as np
from sklearn.cluster import KMeans
from typing import NamedTuple

from backend.config import KMEANS_N_INIT
from backend.core.random_seed import get_random_state


class ClusteringResult(NamedTuple):
    """Container for clustering results."""
    labels: np.ndarray
    centers: np.ndarray
    inertia: float


def perform_kmeans(
    data: np.ndarray,
    n_clusters: int,
    random_state: int | None = None,
    n_init: int = KMEANS_N_INIT,
) -> ClusteringResult:
    """
    Perform K-Means clustering.

    Args:
        data: Feature matrix of shape (n_samples, n_features)
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        n_init: Number of initializations

    Returns:
        ClusteringResult with labels, centers, and inertia
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state if random_state is not None else get_random_state(),
        n_init=n_init,
    )
    labels = kmeans.fit_predict(data)

    return ClusteringResult(
        labels=labels,
        centers=kmeans.cluster_centers_,
        inertia=kmeans.inertia_,
    )


def get_cluster_sizes(labels: np.ndarray) -> list[int]:
    """
    Get the size of each cluster.

    Args:
        labels: Cluster labels for each sample

    Returns:
        List of cluster sizes
    """
    unique, counts = np.unique(labels, return_counts=True)
    # Ensure we return in order (0, 1, 2, ...)
    sizes = [0] * (max(unique) + 1)
    for label, count in zip(unique, counts):
        sizes[label] = count
    return sizes
