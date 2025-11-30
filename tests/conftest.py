"""Shared pytest fixtures for tests."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_documents():
    """Small sample of documents for testing."""
    return [
        "The quick brown fox jumps over the lazy dog in the park.",
        "Machine learning algorithms are used for data analysis.",
        "Python programming language is popular for scientific computing.",
        "Natural language processing helps understand text data.",
        "Deep learning models require large amounts of training data.",
        "The weather today is sunny with clear blue skies.",
        "Computer vision enables machines to interpret images.",
        "Statistical methods are fundamental to data science.",
        "Neural networks can learn complex patterns in data.",
        "Text classification is a common NLP task.",
    ]


@pytest.fixture
def sample_tokenized_docs():
    """Pre-tokenized sample documents."""
    return [
        ["quick", "brown", "fox", "jump", "lazy", "dog", "park"],
        ["machine", "learn", "algorithm", "data", "analysis"],
        ["python", "program", "language", "popular", "scientific", "compute"],
        ["natural", "language", "process", "help", "understand", "text", "data"],
        ["deep", "learn", "model", "require", "large", "amount", "train", "data"],
        ["weather", "today", "sunny", "clear", "blue", "sky"],
        ["computer", "vision", "enable", "machine", "interpret", "image"],
        ["statistical", "method", "fundamental", "data", "science"],
        ["neural", "network", "learn", "complex", "pattern", "data"],
        ["text", "classification", "common", "nlp", "task"],
    ]


@pytest.fixture
def sample_doc_topic_matrix():
    """Sample document-topic distribution matrix."""
    np.random.seed(42)
    # 10 documents, 3 topics
    matrix = np.random.dirichlet([0.5, 0.5, 0.5], size=10)
    return matrix


@pytest.fixture
def sample_projections():
    """Sample 2D projections."""
    np.random.seed(42)
    return np.random.randn(10, 2)


@pytest.fixture
def temp_cache_dir():
    """Temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def small_corpus_size():
    """Small corpus size for faster tests."""
    return 100
