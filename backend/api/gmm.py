"""GMM Clustering API endpoints."""

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.cache.manager import (
    load_doc_topic_distribution,
    load_gmm_metrics,
    load_gmm_labels,
    load_gmm_probabilities,
)
from backend.core.gmm import perform_gmm, get_cluster_sizes
from backend.core.gmm_metrics import compute_gmm_metrics_for_all_clusters
from backend.models.requests import GMMRequest
from backend.models.responses import (
    GMMResponse,
    GMMMetricsResponse,
    GMMAllCovarianceMetricsResponse,
    ClusterProbability,
)
from backend.config import MIN_TOPICS, MAX_TOPICS, MIN_CLUSTERS, MAX_CLUSTERS

router = APIRouter(prefix="/gmm", tags=["gmm"])


def _get_top_probabilities(
    probs: np.ndarray, top_n: int = 3
) -> list[list[ClusterProbability]]:
    """
    Extract top N cluster probabilities for each document.

    Args:
        probs: Probability matrix of shape (n_samples, n_clusters)
        top_n: Number of top probabilities to return per document

    Returns:
        List of lists of ClusterProbability for each document
    """
    result = []
    for doc_probs in probs:
        # Get indices of top N probabilities
        top_indices = np.argsort(doc_probs)[::-1][:top_n]
        doc_top_probs = [
            ClusterProbability(
                cluster_id=int(idx),
                probability=round(float(doc_probs[idx]), 4),
            )
            for idx in top_indices
            if doc_probs[idx] > 0.01  # Only include if probability > 1%
        ]
        result.append(doc_top_probs)
    return result


@router.post("", response_model=GMMResponse)
async def gmm_cluster_documents(request: GMMRequest):
    """
    Perform GMM clustering on document-topic distributions.

    Uses precomputed results when available, falls back to runtime computation.
    """
    distribution = load_doc_topic_distribution(request.n_topics)

    if distribution is None:
        raise HTTPException(
            status_code=503,
            detail=f"Distribution for {request.n_topics} topics not available. Run precomputation first.",
        )

    # Try precomputed labels and probabilities first
    labels = load_gmm_labels(
        request.n_topics, request.n_clusters, request.covariance_type
    )
    probs = load_gmm_probabilities(
        request.n_topics, request.n_clusters, request.covariance_type
    )

    if labels is not None and probs is not None:
        # Use precomputed results
        # Compute BIC/AIC from cached metrics if available
        cached_metrics = load_gmm_metrics(request.n_topics, request.covariance_type)
        if cached_metrics is not None:
            # Find the index for this cluster count
            idx = cached_metrics["cluster_counts"].index(request.n_clusters)
            bic = cached_metrics["bic_scores"][idx]
            aic = cached_metrics["aic_scores"][idx]
        else:
            # Fallback: compute fresh
            result = perform_gmm(
                distribution, request.n_clusters, request.covariance_type
            )
            bic = result.bic
            aic = result.aic
    else:
        # Fallback to runtime computation
        result = perform_gmm(
            distribution, request.n_clusters, request.covariance_type
        )
        labels = result.labels
        probs = result.probabilities
        bic = result.bic
        aic = result.aic

    # Get top 3 probabilities per document
    top_probs = _get_top_probabilities(probs, top_n=3)

    # Get cluster sizes
    sizes = get_cluster_sizes(labels)

    return GMMResponse(
        n_topics=request.n_topics,
        n_clusters=request.n_clusters,
        covariance_type=request.covariance_type,
        labels=labels.tolist() if hasattr(labels, "tolist") else list(labels),
        probabilities=top_probs,
        bic=bic,
        aic=aic,
        cluster_sizes=sizes,
    )


@router.get("/metrics/{n_topics}", response_model=GMMMetricsResponse)
async def get_gmm_metrics(
    n_topics: int,
    covariance_type: Literal["full", "diag", "spherical"] = "full",
    min_clusters: int = MIN_CLUSTERS,
    max_clusters: int = MAX_CLUSTERS,
):
    """
    Get GMM clustering metrics (BIC, AIC) for a range of cluster counts.

    Used for the "optimal number of clusters" chart.
    """
    if n_topics < MIN_TOPICS or n_topics > MAX_TOPICS:
        raise HTTPException(
            status_code=400,
            detail=f"n_topics must be between {MIN_TOPICS} and {MAX_TOPICS}",
        )

    # Try precomputed cache first (if requesting default range)
    if min_clusters == MIN_CLUSTERS and max_clusters == MAX_CLUSTERS:
        cached = load_gmm_metrics(n_topics, covariance_type)
        if cached is not None:
            return GMMMetricsResponse(
                n_topics=n_topics,
                covariance_type=covariance_type,
                cluster_counts=cached["cluster_counts"],
                bic_scores=cached["bic_scores"],
                aic_scores=cached["aic_scores"],
                optimal_bic=cached["optimal_bic"],
                optimal_aic=cached["optimal_aic"],
            )

    # Fallback to runtime computation
    if min_clusters < MIN_CLUSTERS:
        min_clusters = MIN_CLUSTERS
    if max_clusters > MAX_CLUSTERS:
        max_clusters = MAX_CLUSTERS

    distribution = load_doc_topic_distribution(n_topics)

    if distribution is None:
        raise HTTPException(
            status_code=503,
            detail=f"Distribution for {n_topics} topics not available. Run precomputation first.",
        )

    # Compute metrics for all cluster counts
    metrics = compute_gmm_metrics_for_all_clusters(
        distribution,
        covariance_type=covariance_type,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )

    return GMMMetricsResponse(
        n_topics=n_topics,
        covariance_type=covariance_type,
        cluster_counts=metrics["cluster_counts"],
        bic_scores=[round(float(b), 2) for b in metrics["bic_scores"]],
        aic_scores=[round(float(a), 2) for a in metrics["aic_scores"]],
        optimal_bic=metrics["optimal_bic"],
        optimal_aic=metrics["optimal_aic"],
    )


@router.get("/metrics/all/{n_topics}", response_model=GMMAllCovarianceMetricsResponse)
async def get_gmm_all_metrics(n_topics: int):
    """
    Get GMM clustering metrics for ALL covariance types at once.

    Useful for comparing covariance types in the frontend.
    """
    if n_topics < MIN_TOPICS or n_topics > MAX_TOPICS:
        raise HTTPException(
            status_code=400,
            detail=f"n_topics must be between {MIN_TOPICS} and {MAX_TOPICS}",
        )

    distribution = load_doc_topic_distribution(n_topics)

    if distribution is None:
        raise HTTPException(
            status_code=503,
            detail=f"Distribution for {n_topics} topics not available. Run precomputation first.",
        )

    results = {}
    for cov_type in ["full", "diag", "spherical"]:
        # Try cache first
        cached = load_gmm_metrics(n_topics, cov_type)
        if cached is not None:
            results[cov_type] = GMMMetricsResponse(
                n_topics=n_topics,
                covariance_type=cov_type,
                cluster_counts=cached["cluster_counts"],
                bic_scores=cached["bic_scores"],
                aic_scores=cached["aic_scores"],
                optimal_bic=cached["optimal_bic"],
                optimal_aic=cached["optimal_aic"],
            )
        else:
            # Compute fresh
            metrics = compute_gmm_metrics_for_all_clusters(
                distribution, covariance_type=cov_type
            )
            results[cov_type] = GMMMetricsResponse(
                n_topics=n_topics,
                covariance_type=cov_type,
                cluster_counts=metrics["cluster_counts"],
                bic_scores=[round(float(b), 2) for b in metrics["bic_scores"]],
                aic_scores=[round(float(a), 2) for a in metrics["aic_scores"]],
                optimal_bic=metrics["optimal_bic"],
                optimal_aic=metrics["optimal_aic"],
            )

    return GMMAllCovarianceMetricsResponse(
        n_topics=n_topics,
        full=results["full"],
        diag=results["diag"],
        spherical=results["spherical"],
    )
