"""Tests for cache manager module."""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from backend.cache.manager import (
    ensure_cache_dirs,
    save_doc_topic_distribution,
    load_doc_topic_distribution,
    save_umap_projection,
    load_umap_projection,
    save_coherence_scores,
    load_coherence_scores,
    is_cache_complete,
    get_cache_status,
)


class TestEnsureCacheDirs:
    """Tests for cache directory creation."""

    def test_creates_directories(self, temp_cache_dir, monkeypatch):
        """Should create cache directories."""
        from backend import config

        # Patch config to use temp directory
        monkeypatch.setattr(config, "CACHE_DIR", temp_cache_dir)
        monkeypatch.setattr(config, "MODELS_DIR", temp_cache_dir / "models")
        monkeypatch.setattr(config, "DISTRIBUTIONS_DIR", temp_cache_dir / "distributions")
        monkeypatch.setattr(config, "PROJECTIONS_DIR", temp_cache_dir / "projections")
        monkeypatch.setattr(config, "METRICS_DIR", temp_cache_dir / "metrics")

        # Need to reimport to use patched config
        from backend.cache import manager
        import importlib
        importlib.reload(manager)

        manager.ensure_cache_dirs()

        assert (temp_cache_dir / "models").exists()
        assert (temp_cache_dir / "distributions").exists()
        assert (temp_cache_dir / "projections").exists()
        assert (temp_cache_dir / "metrics").exists()


class TestDocTopicDistribution:
    """Tests for doc-topic distribution cache."""

    def test_save_and_load(self, sample_doc_topic_matrix, temp_cache_dir, monkeypatch):
        """Should save and load distribution correctly."""
        from backend import config
        monkeypatch.setattr(config, "DISTRIBUTIONS_DIR", temp_cache_dir)

        from backend.cache import manager
        import importlib
        importlib.reload(manager)

        # Save for train dataset
        manager.save_doc_topic_distribution(sample_doc_topic_matrix, num_topics=3, dataset="train")

        # Load
        loaded = manager.load_doc_topic_distribution(3, dataset="train")

        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded, sample_doc_topic_matrix)

    def test_load_nonexistent_returns_none(self, temp_cache_dir, monkeypatch):
        """Should return None for nonexistent file."""
        from backend import config
        monkeypatch.setattr(config, "DISTRIBUTIONS_DIR", temp_cache_dir)

        from backend.cache import manager
        import importlib
        importlib.reload(manager)

        result = manager.load_doc_topic_distribution(999, dataset="train")
        assert result is None


class TestUmapProjection:
    """Tests for UMAP projection cache."""

    def test_save_and_load(self, sample_projections, temp_cache_dir, monkeypatch):
        """Should save and load projections correctly."""
        from backend import config
        monkeypatch.setattr(config, "PROJECTIONS_DIR", temp_cache_dir)

        from backend.cache import manager
        import importlib
        importlib.reload(manager)

        # Save for train dataset
        manager.save_umap_projection(sample_projections, num_topics=5, dataset="train")

        # Load
        loaded = manager.load_umap_projection(5, dataset="train")

        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded, sample_projections)


class TestCoherenceScores:
    """Tests for coherence scores cache."""

    def test_save_and_load(self, temp_cache_dir, monkeypatch):
        """Should save and load coherence scores correctly."""
        from backend import config
        monkeypatch.setattr(config, "METRICS_DIR", temp_cache_dir)

        from backend.cache import manager
        import importlib
        importlib.reload(manager)

        scores = {2: 0.35, 3: 0.42, 4: 0.38, 5: 0.40}

        # Save for test split
        manager.save_coherence_scores(scores, split="test")

        # Load
        loaded = manager.load_coherence_scores(split="test")

        assert loaded is not None
        assert loaded == scores

    def test_handles_int_keys(self, temp_cache_dir, monkeypatch):
        """Should preserve integer keys through JSON."""
        from backend import config
        monkeypatch.setattr(config, "METRICS_DIR", temp_cache_dir)

        from backend.cache import manager
        import importlib
        importlib.reload(manager)

        scores = {2: 0.35, 10: 0.42}
        manager.save_coherence_scores(scores, split="val")
        loaded = manager.load_coherence_scores(split="val")

        # Keys should still be integers
        assert all(isinstance(k, int) for k in loaded.keys())


class TestCacheStatus:
    """Tests for cache status functions."""

    def test_is_cache_complete_false_when_empty(self, temp_cache_dir, monkeypatch):
        """Should return False for empty cache."""
        from backend import config
        monkeypatch.setattr(config, "CACHE_DIR", temp_cache_dir)
        monkeypatch.setattr(config, "MODELS_DIR", temp_cache_dir / "models")
        monkeypatch.setattr(config, "DISTRIBUTIONS_DIR", temp_cache_dir / "distributions")
        monkeypatch.setattr(config, "PROJECTIONS_DIR", temp_cache_dir / "projections")
        monkeypatch.setattr(config, "METRICS_DIR", temp_cache_dir / "metrics")

        from backend.cache import manager
        import importlib
        importlib.reload(manager)

        assert manager.is_cache_complete() is False

    def test_get_cache_status_returns_dict(self, temp_cache_dir, monkeypatch):
        """Should return status dictionary."""
        from backend import config
        monkeypatch.setattr(config, "CACHE_DIR", temp_cache_dir)
        monkeypatch.setattr(config, "MODELS_DIR", temp_cache_dir / "models")
        monkeypatch.setattr(config, "DISTRIBUTIONS_DIR", temp_cache_dir / "distributions")
        monkeypatch.setattr(config, "PROJECTIONS_DIR", temp_cache_dir / "projections")
        monkeypatch.setattr(config, "METRICS_DIR", temp_cache_dir / "metrics")

        from backend.cache import manager
        import importlib
        importlib.reload(manager)

        status = manager.get_cache_status()

        assert isinstance(status, dict)
        assert "complete" in status
        assert "models" in status
        assert "distributions_train" in status
        assert "distributions_test" in status
        assert "projections_train" in status
        assert "projections_test" in status
