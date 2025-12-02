"""Configuration settings for the backend."""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "cache"

# Cache subdirectories
MODELS_DIR = CACHE_DIR / "models"
DISTRIBUTIONS_DIR = CACHE_DIR / "distributions"
PROJECTIONS_DIR = CACHE_DIR / "projections"
METRICS_DIR = CACHE_DIR / "metrics"
PYLDAVIS_DIR = CACHE_DIR / "pyldavis"

# LDA settings
MIN_TOPICS = 2
MAX_TOPICS = 20
LDA_PASSES = 10
LDA_ITERATIONS = 5000
LDA_CHUNKSIZE = 11000
LDA_RANDOM_STATE = 42
LDA_ALPHA = "auto"  # Document-topic density ('auto', 'symmetric', or float)
LDA_ETA = "auto"  # Topic-word density ('auto', 'symmetric', or float)

# Clustering settings
MIN_CLUSTERS = 2
MAX_CLUSTERS = 15
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10

# UMAP settings
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 2
UMAP_RANDOM_STATE = 42

# Preprocessing settings
MIN_TOKEN_LENGTH = 3
MAX_TOKEN_LENGTH = 50
CUSTOM_STOPWORDS_FILE = BASE_DIR / "custom_stopwords.txt"

# Dictionary filter settings (used by filter_extremes)
FILTER_NO_BELOW = 5  # Tokens must appear in at least 5 documents
FILTER_NO_ABOVE = 0.5  # Tokens can appear in max 50% of documents

# Document filtering settings
MIN_DOC_TOKENS = 8  # Minimum tokens per document after filter_extremes

# Legacy aliases (for backward compatibility with existing code)
MIN_DOC_FREQ = FILTER_NO_BELOW
MAX_DOC_FREQ_RATIO = FILTER_NO_ABOVE
