"""Unified random seed management for reproducibility."""

import random

import numpy as np

RANDOM_SEED = 42


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seed for all libraries used in the project.

    Call once at startup or before any stochastic operation.

    Args:
        seed: The random seed to use (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)


def get_random_state() -> int:
    """
    Get the global random state value for sklearn/gensim models.

    Returns:
        The random seed integer (42)
    """
    return RANDOM_SEED
