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
    MIN_TOPICS,
    MAX_TOPICS,
)


def ensure_cache_dirs() -> None:
    """Create cache directories if they don't exist."""
    for dir_path in [MODELS_DIR, DISTRIBUTIONS_DIR, PROJECTIONS_DIR, METRICS_DIR, PYLDAVIS_DIR]:
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
    """Save corpus to cache."""
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
    """Save tokenized documents to cache."""
    ensure_cache_dirs()
    path = MODELS_DIR / "tokenized_docs.pkl"
    with open(path, "wb") as f:
        pickle.dump(tokenized_docs, f)
    return path


def load_tokenized_docs() -> list[list[str]] | None:
    """Load tokenized documents from cache."""
    path = MODELS_DIR / "tokenized_docs.pkl"
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
    """Save document-topic distribution to cache."""
    ensure_cache_dirs()
    path = DISTRIBUTIONS_DIR / f"doc_topic_k{num_topics}.npy"
    np.save(path, dist)
    return path


def load_doc_topic_distribution(num_topics: int) -> np.ndarray | None:
    """Load document-topic distribution from cache."""
    path = DISTRIBUTIONS_DIR / f"doc_topic_k{num_topics}.npy"
    if path.exists():
        return np.load(path)
    return None


def save_umap_projection(projection: np.ndarray, num_topics: int) -> Path:
    """Save UMAP projection to cache."""
    ensure_cache_dirs()
    path = PROJECTIONS_DIR / f"umap_k{num_topics}.npy"
    np.save(path, projection)
    return path


def load_umap_projection(num_topics: int) -> np.ndarray | None:
    """Load UMAP projection from cache."""
    path = PROJECTIONS_DIR / f"umap_k{num_topics}.npy"
    if path.exists():
        return np.load(path)
    return None


def save_coherence_scores(scores: dict[int, float]) -> Path:
    """Save coherence scores to cache."""
    ensure_cache_dirs()
    path = METRICS_DIR / "coherence_scores.json"
    # Convert int keys to strings for JSON
    json_scores = {str(k): v for k, v in scores.items()}
    with open(path, "w") as f:
        json.dump(json_scores, f, indent=2)
    return path


def load_coherence_scores() -> dict[int, float] | None:
    """Load coherence scores from cache."""
    path = METRICS_DIR / "coherence_scores.json"
    if path.exists():
        with open(path, "r") as f:
            json_scores = json.load(f)
        # Convert string keys back to ints
        return {int(k): v for k, v in json_scores.items()}
    return None


def save_perplexity_scores(scores: dict[int, float]) -> Path:
    """Save perplexity scores to cache."""
    ensure_cache_dirs()
    path = METRICS_DIR / "perplexity_scores.json"
    # Convert int keys to strings for JSON
    json_scores = {str(k): v for k, v in scores.items()}
    with open(path, "w") as f:
        json.dump(json_scores, f, indent=2)
    return path


def load_perplexity_scores() -> dict[int, float] | None:
    """Load perplexity scores from cache."""
    path = METRICS_DIR / "perplexity_scores.json"
    if path.exists():
        with open(path, "r") as f:
            json_scores = json.load(f)
        # Convert string keys back to ints
        return {int(k): v for k, v in json_scores.items()}
    return None


def is_cache_complete() -> bool:
    """Check if all precomputed artifacts exist."""
    # Check dictionary and corpus
    if not (MODELS_DIR / "dictionary.pkl").exists():
        return False
    if not (MODELS_DIR / "corpus.pkl").exists():
        return False
    if not (MODELS_DIR / "tokenized_docs.pkl").exists():
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
    if not (METRICS_DIR / "coherence_scores.json").exists():
        return False

    return True


def get_cache_status() -> dict[str, Any]:
    """Get detailed cache status."""
    status = {
        "complete": is_cache_complete(),
        "dictionary": (MODELS_DIR / "dictionary.pkl").exists(),
        "corpus": (MODELS_DIR / "corpus.pkl").exists(),
        "tokenized_docs": (MODELS_DIR / "tokenized_docs.pkl").exists(),
        "coherence_scores": (METRICS_DIR / "coherence_scores.json").exists(),
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


def clear_cache() -> None:
    """Clear all cached artifacts."""
    import shutil

    for dir_path in [MODELS_DIR, DISTRIBUTIONS_DIR, PROJECTIONS_DIR, METRICS_DIR, PYLDAVIS_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)

    ensure_cache_dirs()
