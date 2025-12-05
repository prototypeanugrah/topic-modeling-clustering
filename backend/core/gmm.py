"""Gaussian Mixture Model clustering utilities."""

import numpy as np
from sklearn.mixture import GaussianMixture
from typing import NamedTuple, Literal

from backend.config import GMM_N_INIT, GMM_MAX_ITER
from backend.core.random_seed import get_random_state


class GMMResult(NamedTuple):
    """Container for GMM clustering results."""
    labels: np.ndarray           # Hard cluster assignments
    probabilities: np.ndarray    # Soft probabilities (n_samples, n_clusters)
    means: np.ndarray            # Cluster centers (n_clusters, n_features)
    bic: float                   # Bayesian Information Criterion
    aic: float                   # Akaike Information Criterion


def perform_gmm(
    data: np.ndarray,
    n_clusters: int,
    covariance_type: Literal["full", "diag", "spherical"] = "full",
    random_state: int | None = None,
    n_init: int = GMM_N_INIT,
    max_iter: int = GMM_MAX_ITER,
) -> GMMResult:
    """
    Perform Gaussian Mixture Model clustering.

    Args:
        data: Feature matrix of shape (n_samples, n_features)
        n_clusters: Number of mixture components
        covariance_type: Type of covariance parameters:
            - 'full': Each component has its own general covariance matrix
            - 'diag': Each component has its own diagonal covariance matrix
            - 'spherical': Each component has its own single variance
        random_state: Random seed for reproducibility
        n_init: Number of initializations
        max_iter: Maximum EM iterations

    Returns:
        GMMResult with labels, probabilities, means, BIC, and AIC
    """
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        random_state=random_state if random_state is not None else get_random_state(),
        n_init=n_init,
        max_iter=max_iter,
    )

    gmm.fit(data)

    labels = gmm.predict(data)
    probabilities = gmm.predict_proba(data)

    return GMMResult(
        labels=labels,
        probabilities=probabilities,
        means=gmm.means_,
        bic=gmm.bic(data),
        aic=gmm.aic(data),
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
