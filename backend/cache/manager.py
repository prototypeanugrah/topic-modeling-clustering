"""Cache management for precomputed artifacts."""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from gensim import corpora
from gensim.models import LdaModel

from backend.config import (
    CACHE_DIR,
    MODELS_DIR,
    DISTRIBUTIONS_DIR,
    PROJECTIONS_DIR,
    METRICS_DIR,
    PYLDAVIS_DIR,
    TOPICS_DIR,
    CLUSTERS_DIR,
    ENRICHMENT_DIR,
    MIN_TOPICS,
    MAX_TOPICS,
)


def ensure_cache_dirs() -> None:
    """Create cache directories if they don't exist."""
    for dir_path in [
        MODELS_DIR,
        DISTRIBUTIONS_DIR,
        PROJECTIONS_DIR,
        METRICS_DIR,
        PYLDAVIS_DIR,
        TOPICS_DIR,
        CLUSTERS_DIR,
        ENRICHMENT_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


def save_dictionary(dictionary: corpora.Dictionary) -> Path:
    """Save gensim dictionary to cache."""
    ensure_cache_dirs()
    path = MODELS_DIR / "dictionary.pkl"
    dictionary.save(str(path))
    return path


def load_dictionary() -> corpora.Dictionary | None:
    """Load gensim dictionary from cache."""
    path = MODELS_DIR / "dictionary.pkl"
    if path.exists():
        return corpora.Dictionary.load(str(path))
    return None


def save_corpus(corpus: list) -> Path:
    """Save corpus to cache.

    Args:
        corpus: Bag-of-words corpus
    """
    ensure_cache_dirs()
    path = MODELS_DIR / "corpus.pkl"
    with open(path, "wb") as f:
        pickle.dump(corpus, f)
    return path


def load_corpus() -> list | None:
    """Load corpus from cache."""
    path = MODELS_DIR / "corpus.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_tokenized_docs(tokenized_docs: list[list[str]]) -> Path:
    """Save tokenized documents to cache.

    Args:
        tokenized_docs: List of tokenized documents
    """
    ensure_cache_dirs()
    path = MODELS_DIR / "tokenized.pkl"
    with open(path, "wb") as f:
        pickle.dump(tokenized_docs, f)
    return path


def load_tokenized_docs() -> list[list[str]] | None:
    """Load tokenized documents from cache."""
    path = MODELS_DIR / "tokenized.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_lda_model(model: LdaModel, num_topics: int) -> Path:
    """Save LDA model to cache."""
    ensure_cache_dirs()
    path = MODELS_DIR / f"lda_k{num_topics}.pkl"
    model.save(str(path))
    return path


def load_lda_model(num_topics: int) -> LdaModel | None:
    """Load LDA model from cache."""
    path = MODELS_DIR / f"lda_k{num_topics}.pkl"
    if path.exists():
        return LdaModel.load(str(path))
    return None


def save_doc_topic_distribution(dist: np.ndarray, num_topics: int) -> Path:
    """Save document-topic distribution to cache.

    Args:
        dist: Document-topic distribution matrix
        num_topics: Number of topics
    """
    ensure_cache_dirs()
    path = DISTRIBUTIONS_DIR / f"doc_topic_k{num_topics}.npy"
    np.save(path, dist)
    return path


def load_doc_topic_distribution(num_topics: int) -> np.ndarray | None:
    """Load document-topic distribution from cache.

    Args:
        num_topics: Number of topics
    """
    path = DISTRIBUTIONS_DIR / f"doc_topic_k{num_topics}.npy"
    if path.exists():
        return np.load(path)
    return None


def save_umap_projection(projection: np.ndarray, num_topics: int) -> Path:
    """Save UMAP projection to cache.

    Args:
        projection: UMAP 2D projection
        num_topics: Number of topics
    """
    ensure_cache_dirs()
    path = PROJECTIONS_DIR / f"umap_k{num_topics}.npy"
    np.save(path, projection)
    return path


def load_umap_projection(num_topics: int) -> np.ndarray | None:
    """Load UMAP projection from cache.

    Args:
        num_topics: Number of topics
    """
    path = PROJECTIONS_DIR / f"umap_k{num_topics}.npy"
    if path.exists():
        return np.load(path)
    return None


def save_coherence_scores(scores: dict[int, float]) -> Path:
    """Save coherence scores to cache.

    Args:
        scores: Dictionary mapping topic count to coherence score
    """
    ensure_cache_dirs()
    path = METRICS_DIR / "coherence.json"
    # Convert int keys to strings for JSON
    json_scores = {str(k): v for k, v in scores.items()}
    with open(path, "w") as f:
        json.dump(json_scores, f, indent=2)
    return path


def load_coherence_scores() -> dict[int, float] | None:
    """Load coherence scores from cache."""
    path = METRICS_DIR / "coherence.json"
    if path.exists():
        with open(path, "r") as f:
            json_scores = json.load(f)
        # Convert string keys back to ints
        return {int(k): v for k, v in json_scores.items()}
    return None


def is_cache_complete() -> bool:
    """Check if all precomputed artifacts exist."""
    # Check dictionary
    if not (MODELS_DIR / "dictionary.pkl").exists():
        return False

    # Check corpus and tokenized docs
    if not (MODELS_DIR / "corpus.pkl").exists():
        return False
    if not (MODELS_DIR / "tokenized.pkl").exists():
        return False

    # Check LDA models and distributions for all topic counts
    for n in range(MIN_TOPICS, MAX_TOPICS + 1):
        if not (MODELS_DIR / f"lda_k{n}.pkl").exists():
            return False
        if not (DISTRIBUTIONS_DIR / f"doc_topic_k{n}.npy").exists():
            return False
        if not (PROJECTIONS_DIR / f"umap_k{n}.npy").exists():
            return False

    # Check coherence scores
    if not (METRICS_DIR / "coherence.json").exists():
        return False

    return True


def get_cache_status() -> dict[str, Any]:
    """Get detailed cache status."""
    status = {
        "complete": is_cache_complete(),
        "dictionary": (MODELS_DIR / "dictionary.pkl").exists(),
        "corpus": (MODELS_DIR / "corpus.pkl").exists(),
        "tokenized": (MODELS_DIR / "tokenized.pkl").exists(),
        "coherence": (METRICS_DIR / "coherence.json").exists(),
        "models": {},
        "distributions": {},
        "projections": {},
    }

    for n in range(MIN_TOPICS, MAX_TOPICS + 1):
        status["models"][n] = (MODELS_DIR / f"lda_k{n}.pkl").exists()
        status["distributions"][n] = (DISTRIBUTIONS_DIR / f"doc_topic_k{n}.npy").exists()
        status["projections"][n] = (PROJECTIONS_DIR / f"umap_k{n}.npy").exists()

    return status


def save_pyldavis_html(html_content: str, num_topics: int) -> Path:
    """Save pyLDAvis HTML to cache."""
    ensure_cache_dirs()
    path = PYLDAVIS_DIR / f"k{num_topics}.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return path


def load_pyldavis_html(num_topics: int) -> str | None:
    """Load pyLDAvis HTML from cache."""
    path = PYLDAVIS_DIR / f"k{num_topics}.html"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def get_pyldavis_path(num_topics: int) -> Path | None:
    """Get path to pyLDAvis HTML file."""
    path = PYLDAVIS_DIR / f"k{num_topics}.html"
    if path.exists():
        return path
    return None


def save_eda_stats(stats: dict) -> Path:
    """Save EDA statistics to cache.

    Args:
        stats: Dictionary with EDA statistics for all stages
    """
    ensure_cache_dirs()
    path = METRICS_DIR / "eda_stats.json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    return path


def load_eda_stats() -> dict | None:
    """Load EDA statistics from cache."""
    path = METRICS_DIR / "eda_stats.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_document_labels(labels: list[str]) -> Path:
    """Save newsgroup labels for documents.

    Args:
        labels: List of newsgroup label strings (e.g., ["alt.atheism", "sci.space", ...])
    """
    ensure_cache_dirs()
    path = MODELS_DIR / "labels.json"
    with open(path, "w") as f:
        json.dump(labels, f)
    return path


def load_document_labels() -> list[str] | None:
    """Load newsgroup labels for documents.

    Returns:
        List of newsgroup label strings, or None if not cached
    """
    path = MODELS_DIR / "labels.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_box_plot_data(data: dict) -> Path:
    """Save box plot data for EDA visualizations.

    Args:
        data: Dictionary with token counts by stage and category

    Returns:
        Path to saved file
    """
    ensure_cache_dirs()
    path = METRICS_DIR / "box_plot_data.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def load_box_plot_data() -> dict | None:
    """Load box plot data for EDA visualizations.

    Returns:
        Dictionary with token counts, or None if not cached
    """
    path = METRICS_DIR / "box_plot_data.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def clear_cache() -> None:
    """Clear all cached artifacts."""
    import shutil

    for dir_path in [
        MODELS_DIR,
        DISTRIBUTIONS_DIR,
        PROJECTIONS_DIR,
        METRICS_DIR,
        PYLDAVIS_DIR,
        TOPICS_DIR,
        CLUSTERS_DIR,
        ENRICHMENT_DIR,
    ]:
        if dir_path.exists():
            shutil.rmtree(dir_path)

    ensure_cache_dirs()


# === Topic Words Cache ===


def save_topic_words(words: dict, num_topics: int) -> Path:
    """Save precomputed topic words to cache.

    Args:
        words: Dictionary with n_topics and topics list
        num_topics: Number of topics
    """
    ensure_cache_dirs()
    path = TOPICS_DIR / f"words_k{num_topics}.json"
    with open(path, "w") as f:
        json.dump(words, f)
    return path


def load_topic_words(num_topics: int) -> dict | None:
    """Load precomputed topic words from cache."""
    path = TOPICS_DIR / f"words_k{num_topics}.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


# === Cluster Metrics Cache ===


def save_cluster_metrics(metrics: dict, num_topics: int) -> Path:
    """Save precomputed cluster metrics to cache.

    Args:
        metrics: Dictionary with cluster metrics (silhouette, inertia, etc.)
        num_topics: Number of topics
    """
    ensure_cache_dirs()
    path = METRICS_DIR / f"cluster_metrics_k{num_topics}.json"
    with open(path, "w") as f:
        json.dump(metrics, f)
    return path


def load_cluster_metrics(num_topics: int) -> dict | None:
    """Load precomputed cluster metrics from cache."""
    path = METRICS_DIR / f"cluster_metrics_k{num_topics}.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


# === Cluster Labels Cache ===


def save_cluster_labels(labels: np.ndarray, num_topics: int, num_clusters: int) -> Path:
    """Save precomputed cluster labels to cache.

    Args:
        labels: Cluster label array for each document
        num_topics: Number of topics
        num_clusters: Number of clusters
    """
    ensure_cache_dirs()
    path = CLUSTERS_DIR / f"labels_k{num_topics}_c{num_clusters}.npy"
    np.save(path, labels)
    return path


def load_cluster_labels(num_topics: int, num_clusters: int) -> np.ndarray | None:
    """Load precomputed cluster labels from cache."""
    path = CLUSTERS_DIR / f"labels_k{num_topics}_c{num_clusters}.npy"
    if path.exists():
        return np.load(path)
    return None


# === Document Enrichment Cache ===


def save_document_enrichment(enrichment: dict, num_topics: int) -> Path:
    """Save precomputed document enrichment data to cache.

    Args:
        enrichment: Dictionary with top_topics and dominant_topic_words
        num_topics: Number of topics
    """
    ensure_cache_dirs()
    path = ENRICHMENT_DIR / f"docs_k{num_topics}.json"
    with open(path, "w") as f:
        json.dump(enrichment, f)
    return path


def load_document_enrichment(num_topics: int) -> dict | None:
    """Load precomputed document enrichment data from cache."""
    path = ENRICHMENT_DIR / f"docs_k{num_topics}.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


# === GMM Metrics Cache ===


def save_gmm_metrics(metrics: dict, num_topics: int, cov_type: str) -> Path:
    """Save precomputed GMM metrics to cache.

    Args:
        metrics: Dictionary with GMM metrics (silhouette, bic, aic, etc.)
        num_topics: Number of topics
        cov_type: Covariance type ('full', 'diag', 'spherical')

    Returns:
        Path to saved file
    """
    ensure_cache_dirs()
    path = METRICS_DIR / f"gmm_metrics_k{num_topics}_cov_{cov_type}.json"
    with open(path, "w") as f:
        json.dump(metrics, f)
    return path


def load_gmm_metrics(num_topics: int, cov_type: str) -> dict | None:
    """Load precomputed GMM metrics from cache.

    Args:
        num_topics: Number of topics
        cov_type: Covariance type ('full', 'diag', 'spherical')

    Returns:
        Dictionary with GMM metrics, or None if not cached
    """
    path = METRICS_DIR / f"gmm_metrics_k{num_topics}_cov_{cov_type}.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


# === GMM Labels Cache ===


def save_gmm_labels(
    labels: np.ndarray, num_topics: int, num_clusters: int, cov_type: str
) -> Path:
    """Save precomputed GMM cluster labels to cache.

    Args:
        labels: Cluster label array for each document
        num_topics: Number of topics
        num_clusters: Number of clusters
        cov_type: Covariance type ('full', 'diag', 'spherical')

    Returns:
        Path to saved file
    """
    ensure_cache_dirs()
    path = CLUSTERS_DIR / f"gmm_labels_k{num_topics}_c{num_clusters}_cov_{cov_type}.npy"
    np.save(path, labels)
    return path


def load_gmm_labels(
    num_topics: int, num_clusters: int, cov_type: str
) -> np.ndarray | None:
    """Load precomputed GMM cluster labels from cache.

    Args:
        num_topics: Number of topics
        num_clusters: Number of clusters
        cov_type: Covariance type ('full', 'diag', 'spherical')

    Returns:
        NumPy array of cluster labels, or None if not cached
    """
    path = CLUSTERS_DIR / f"gmm_labels_k{num_topics}_c{num_clusters}_cov_{cov_type}.npy"
    if path.exists():
        return np.load(path)
    return None


# === GMM Probabilities Cache ===


def save_gmm_probabilities(
    probs: np.ndarray, num_topics: int, num_clusters: int, cov_type: str
) -> Path:
    """Save precomputed GMM cluster probabilities (soft assignments) to cache.

    Args:
        probs: Probability matrix of shape (n_samples, n_clusters)
        num_topics: Number of topics
        num_clusters: Number of clusters
        cov_type: Covariance type ('full', 'diag', 'spherical')

    Returns:
        Path to saved file
    """
    ensure_cache_dirs()
    path = CLUSTERS_DIR / f"gmm_probs_k{num_topics}_c{num_clusters}_cov_{cov_type}.npy"
    np.save(path, probs)
    return path


def load_gmm_probabilities(
    num_topics: int, num_clusters: int, cov_type: str
) -> np.ndarray | None:
    """Load precomputed GMM cluster probabilities from cache.

    Args:
        num_topics: Number of topics
        num_clusters: Number of clusters
        cov_type: Covariance type ('full', 'diag', 'spherical')

    Returns:
        NumPy array of probabilities (n_samples, n_clusters), or None if not cached
    """
    path = CLUSTERS_DIR / f"gmm_probs_k{num_topics}_c{num_clusters}_cov_{cov_type}.npy"
    if path.exists():
        return np.load(path)
    return None
