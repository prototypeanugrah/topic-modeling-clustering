"""Evaluation metrics for GMM clustering."""

import numpy as np
from sklearn.metrics import silhouette_score
from typing import Literal

from backend.config import MIN_CLUSTERS, MAX_CLUSTERS


def calculate_silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate silhouette score for clustering.

    Args:
        data: Feature matrix
        labels: Cluster labels

    Returns:
        Silhouette score (-1 to 1, higher is better)
    """
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(data):
        return 0.0

    return silhouette_score(data, labels)


def find_optimal_bic(bic_values: list[float]) -> int:
    """
    Find the optimal cluster count based on minimum BIC.

    Lower BIC indicates a better model (balances fit vs. complexity).

    Args:
        bic_values: List of BIC values for different cluster counts

    Returns:
        Index of the optimal cluster count (minimum BIC)
    """
    if len(bic_values) == 0:
        return 0
    return int(np.argmin(bic_values))


def find_optimal_aic(aic_values: list[float]) -> int:
    """
    Find the optimal cluster count based on minimum AIC.

    Lower AIC indicates a better model (less penalty for complexity than BIC).

    Args:
        aic_values: List of AIC values for different cluster counts

    Returns:
        Index of the optimal cluster count (minimum AIC)
    """
    if len(aic_values) == 0:
        return 0
    return int(np.argmin(aic_values))


def compute_gmm_metrics_for_all_clusters(
    data: np.ndarray,
    covariance_type: Literal["full", "diag", "spherical"] = "full",
    min_clusters: int = MIN_CLUSTERS,
    max_clusters: int = MAX_CLUSTERS,
) -> dict[str, list]:
    """
    Compute GMM metrics for a range of cluster counts.

    Args:
        data: Feature matrix
        covariance_type: GMM covariance type
        min_clusters: Minimum number of clusters
        max_clusters: Maximum number of clusters

    Returns:
        Dictionary with cluster_counts, silhouette_scores, bic_scores,
        aic_scores, optimal_bic, and optimal_aic
    """
    from backend.core.gmm import perform_gmm

    cluster_counts = list(range(min_clusters, max_clusters + 1))
    silhouette_scores = []
    bic_scores = []
    aic_scores = []

    for n_clusters in cluster_counts:
        result = perform_gmm(data, n_clusters, covariance_type=covariance_type)
        silhouette = calculate_silhouette(data, result.labels)
        silhouette_scores.append(silhouette)
        bic_scores.append(result.bic)
        aic_scores.append(result.aic)

    # Find optimal cluster counts (minimum BIC and AIC)
    optimal_bic_idx = find_optimal_bic(bic_scores)
    optimal_bic = cluster_counts[optimal_bic_idx]
    optimal_aic_idx = find_optimal_aic(aic_scores)
    optimal_aic = cluster_counts[optimal_aic_idx]

    return {
        "cluster_counts": cluster_counts,
        "silhouette_scores": silhouette_scores,
        "bic_scores": bic_scores,
        "aic_scores": aic_scores,
        "optimal_bic": optimal_bic,
        "optimal_aic": optimal_aic,
    }
