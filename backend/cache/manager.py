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


def save_corpus(corpus: list, dataset: str = "train") -> Path:
    """Save corpus to cache.

    Args:
        corpus: Bag-of-words corpus
        dataset: 'train' or 'test'
    """
    ensure_cache_dirs()
    path = MODELS_DIR / f"corpus_{dataset}.pkl"
    with open(path, "wb") as f:
        pickle.dump(corpus, f)
    return path


def load_corpus(dataset: str = "train") -> list | None:
    """Load corpus from cache.

    Args:
        dataset: 'train' or 'test'
    """
    path = MODELS_DIR / f"corpus_{dataset}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_tokenized_docs(tokenized_docs: list[list[str]], dataset: str = "train") -> Path:
    """Save tokenized documents to cache.

    Args:
        tokenized_docs: List of tokenized documents
        dataset: 'train' or 'test'
    """
    ensure_cache_dirs()
    path = MODELS_DIR / f"tokenized_{dataset}.pkl"
    with open(path, "wb") as f:
        pickle.dump(tokenized_docs, f)
    return path


def load_tokenized_docs(dataset: str = "train") -> list[list[str]] | None:
    """Load tokenized documents from cache.

    Args:
        dataset: 'train' or 'test'
    """
    path = MODELS_DIR / f"tokenized_{dataset}.pkl"
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


def save_doc_topic_distribution(dist: np.ndarray, num_topics: int, dataset: str = "train") -> Path:
    """Save document-topic distribution to cache.

    Args:
        dist: Document-topic distribution matrix
        num_topics: Number of topics
        dataset: 'train' or 'test'
    """
    ensure_cache_dirs()
    path = DISTRIBUTIONS_DIR / f"doc_topic_{dataset}_k{num_topics}.npy"
    np.save(path, dist)
    return path


def load_doc_topic_distribution(num_topics: int, dataset: str = "train") -> np.ndarray | None:
    """Load document-topic distribution from cache.

    Args:
        num_topics: Number of topics
        dataset: 'train' or 'test'
    """
    path = DISTRIBUTIONS_DIR / f"doc_topic_{dataset}_k{num_topics}.npy"
    if path.exists():
        return np.load(path)
    return None


def save_umap_projection(projection: np.ndarray, num_topics: int, dataset: str = "train") -> Path:
    """Save UMAP projection to cache.

    Args:
        projection: UMAP 2D projection
        num_topics: Number of topics
        dataset: 'train' or 'test'
    """
    ensure_cache_dirs()
    path = PROJECTIONS_DIR / f"umap_{dataset}_k{num_topics}.npy"
    np.save(path, projection)
    return path


def load_umap_projection(num_topics: int, dataset: str = "train") -> np.ndarray | None:
    """Load UMAP projection from cache.

    Args:
        num_topics: Number of topics
        dataset: 'train' or 'test'
    """
    path = PROJECTIONS_DIR / f"umap_{dataset}_k{num_topics}.npy"
    if path.exists():
        return np.load(path)
    return None


def save_coherence_scores(scores: dict[int, float], split: str = "test") -> Path:
    """Save coherence scores to cache.

    Args:
        scores: Dictionary mapping topic count to coherence score
        split: 'val' (averaged from CV) or 'test' (final evaluation)
    """
    ensure_cache_dirs()
    path = METRICS_DIR / f"coherence_{split}.json"
    # Convert int keys to strings for JSON
    json_scores = {str(k): v for k, v in scores.items()}
    with open(path, "w") as f:
        json.dump(json_scores, f, indent=2)
    return path


def load_coherence_scores(split: str = "test") -> dict[int, float] | None:
    """Load coherence scores from cache.

    Args:
        split: 'val' (averaged from CV) or 'test' (final evaluation)
    """
    path = METRICS_DIR / f"coherence_{split}.json"
    if path.exists():
        with open(path, "r") as f:
            json_scores = json.load(f)
        # Convert string keys back to ints
        return {int(k): v for k, v in json_scores.items()}
    return None


def save_perplexity_scores(scores: dict[int, float], split: str = "test") -> Path:
    """Save perplexity scores to cache.

    Args:
        scores: Dictionary mapping topic count to perplexity score
        split: 'val' (averaged from CV) or 'test' (final evaluation)
    """
    ensure_cache_dirs()
    path = METRICS_DIR / f"perplexity_{split}.json"
    # Convert int keys to strings for JSON
    json_scores = {str(k): v for k, v in scores.items()}
    with open(path, "w") as f:
        json.dump(json_scores, f, indent=2)
    return path


def load_perplexity_scores(split: str = "test") -> dict[int, float] | None:
    """Load perplexity scores from cache.

    Args:
        split: 'val' (averaged from CV) or 'test' (final evaluation)
    """
    path = METRICS_DIR / f"perplexity_{split}.json"
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

    # Check train and test corpora and tokenized docs
    for dataset in ["train", "test"]:
        if not (MODELS_DIR / f"corpus_{dataset}.pkl").exists():
            return False
        if not (MODELS_DIR / f"tokenized_{dataset}.pkl").exists():
            return False

    # Check LDA models and distributions for all topic counts
    for n in range(MIN_TOPICS, MAX_TOPICS + 1):
        if not (MODELS_DIR / f"lda_k{n}.pkl").exists():
            return False
        # Check train and test distributions and projections
        for dataset in ["train", "test"]:
            if not (DISTRIBUTIONS_DIR / f"doc_topic_{dataset}_k{n}.npy").exists():
                return False
            if not (PROJECTIONS_DIR / f"umap_{dataset}_k{n}.npy").exists():
                return False

    # Check val and test coherence/perplexity scores
    for split in ["val", "test"]:
        if not (METRICS_DIR / f"coherence_{split}.json").exists():
            return False
        if not (METRICS_DIR / f"perplexity_{split}.json").exists():
            return False

    return True


def get_cache_status() -> dict[str, Any]:
    """Get detailed cache status."""
    status = {
        "complete": is_cache_complete(),
        "dictionary": (MODELS_DIR / "dictionary.pkl").exists(),
        "corpus_train": (MODELS_DIR / "corpus_train.pkl").exists(),
        "corpus_test": (MODELS_DIR / "corpus_test.pkl").exists(),
        "tokenized_train": (MODELS_DIR / "tokenized_train.pkl").exists(),
        "tokenized_test": (MODELS_DIR / "tokenized_test.pkl").exists(),
        "coherence_val": (METRICS_DIR / "coherence_val.json").exists(),
        "coherence_test": (METRICS_DIR / "coherence_test.json").exists(),
        "perplexity_val": (METRICS_DIR / "perplexity_val.json").exists(),
        "perplexity_test": (METRICS_DIR / "perplexity_test.json").exists(),
        "models": {},
        "distributions_train": {},
        "distributions_test": {},
        "projections_train": {},
        "projections_test": {},
    }

    for n in range(MIN_TOPICS, MAX_TOPICS + 1):
        status["models"][n] = (MODELS_DIR / f"lda_k{n}.pkl").exists()
        status["distributions_train"][n] = (DISTRIBUTIONS_DIR / f"doc_topic_train_k{n}.npy").exists()
        status["distributions_test"][n] = (DISTRIBUTIONS_DIR / f"doc_topic_test_k{n}.npy").exists()
        status["projections_train"][n] = (PROJECTIONS_DIR / f"umap_train_k{n}.npy").exists()
        status["projections_test"][n] = (PROJECTIONS_DIR / f"umap_test_k{n}.npy").exists()

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


def clear_cache() -> None:
    """Clear all cached artifacts."""
    import shutil

    for dir_path in [MODELS_DIR, DISTRIBUTIONS_DIR, PROJECTIONS_DIR, METRICS_DIR, PYLDAVIS_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)

    ensure_cache_dirs()
