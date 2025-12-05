"""Dimensionality reduction for visualization."""

import numpy as np
import umap

from backend.config import (
    UMAP_N_NEIGHBORS,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
)
from backend.core.random_seed import get_random_state


def compute_umap(
    data: np.ndarray,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
    n_components: int = UMAP_N_COMPONENTS,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Compute UMAP projection for visualization.

    Args:
        data: Feature matrix of shape (n_samples, n_features)
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance between points
        n_components: Number of output dimensions (usually 2)
        random_state: Random seed for reproducibility

    Returns:
        Projected coordinates of shape (n_samples, n_components)
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state if random_state is not None else get_random_state(),
        metric="cosine",  # Good for topic distributions
    )
    return reducer.fit_transform(data)


def project_cluster_centers(
    centers: np.ndarray,
    data: np.ndarray,
    projections: np.ndarray,
) -> np.ndarray:
    """
    Project cluster centers to 2D space.

    Uses a simple approach: find the centroid of projected points
    for each cluster.

    Args:
        centers: Cluster centers in original space (n_clusters, n_features)
        data: Original data matrix
        projections: UMAP projections of data (n_samples, 2)

    Returns:
        Projected cluster centers (n_clusters, 2)
    """
    # This is a placeholder - for accurate projection, we'd need to
    # transform centers through the fitted UMAP, but that requires
    # keeping the reducer around. For now, we skip this.
    # Centers in visualization will be computed on the frontend
    # based on average of cluster members.
    return np.zeros((centers.shape[0], 2))
