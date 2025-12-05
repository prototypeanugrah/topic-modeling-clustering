"""Pydantic request models."""

from typing import Literal

from pydantic import BaseModel, Field

from backend.config import MIN_TOPICS, MAX_TOPICS, MIN_CLUSTERS, MAX_CLUSTERS


class ClusteringRequest(BaseModel):
    """Request body for clustering endpoint."""

    n_topics: int = Field(
        ...,
        ge=MIN_TOPICS,
        le=MAX_TOPICS,
        description=f"Number of topics ({MIN_TOPICS}-{MAX_TOPICS})"
    )
    n_clusters: int = Field(
        ...,
        ge=MIN_CLUSTERS,
        le=MAX_CLUSTERS,
        description=f"Number of clusters ({MIN_CLUSTERS}-{MAX_CLUSTERS})"
    )


class VisualizationRequest(BaseModel):
    """Request body for clustered visualization endpoint."""

    n_topics: int = Field(
        ...,
        ge=MIN_TOPICS,
        le=MAX_TOPICS,
        description=f"Number of topics ({MIN_TOPICS}-{MAX_TOPICS})"
    )
    n_clusters: int = Field(
        ...,
        ge=MIN_CLUSTERS,
        le=MAX_CLUSTERS,
        description=f"Number of clusters ({MIN_CLUSTERS}-{MAX_CLUSTERS})"
    )


class GMMRequest(BaseModel):
    """Request body for GMM clustering endpoint."""

    n_topics: int = Field(
        ...,
        ge=MIN_TOPICS,
        le=MAX_TOPICS,
        description=f"Number of topics ({MIN_TOPICS}-{MAX_TOPICS})"
    )
    n_clusters: int = Field(
        ...,
        ge=MIN_CLUSTERS,
        le=MAX_CLUSTERS,
        description=f"Number of clusters ({MIN_CLUSTERS}-{MAX_CLUSTERS})"
    )
    covariance_type: Literal["full", "diag", "spherical"] = Field(
        default="full",
        description="GMM covariance type: 'full', 'diag', or 'spherical'"
    )


class GMMVisualizationRequest(BaseModel):
    """Request body for GMM clustered visualization endpoint."""

    n_topics: int = Field(
        ...,
        ge=MIN_TOPICS,
        le=MAX_TOPICS,
        description=f"Number of topics ({MIN_TOPICS}-{MAX_TOPICS})"
    )
    n_clusters: int = Field(
        ...,
        ge=MIN_CLUSTERS,
        le=MAX_CLUSTERS,
        description=f"Number of clusters ({MIN_CLUSTERS}-{MAX_CLUSTERS})"
    )
    covariance_type: Literal["full", "diag", "spherical"] = Field(
        default="full",
        description="GMM covariance type: 'full', 'diag', or 'spherical'"
    )
