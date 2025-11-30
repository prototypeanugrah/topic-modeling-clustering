"""Tests for data_loader module."""

import pytest
from backend.core.data_loader import (
    load_20newsgroups,
    get_document_count,
    get_category_distribution,
    NewsGroupsData,
)


class TestLoad20Newsgroups:
    """Tests for loading 20 Newsgroups dataset."""

    def test_load_returns_named_tuple(self):
        """Should return NewsGroupsData namedtuple."""
        data = load_20newsgroups(subset="train")
        assert isinstance(data, NewsGroupsData)

    def test_load_has_documents(self):
        """Should have non-empty documents list."""
        data = load_20newsgroups(subset="train")
        assert len(data.documents) > 0
        assert all(isinstance(doc, str) for doc in data.documents)

    def test_load_has_targets(self):
        """Should have target labels."""
        data = load_20newsgroups(subset="train")
        assert len(data.target) == len(data.documents)
        # Check targets are integers (numpy or Python int)
        import numpy as np
        assert all(isinstance(t, (int, np.integer)) for t in data.target)

    def test_load_has_20_categories(self):
        """Should have 20 category names."""
        data = load_20newsgroups(subset="train")
        assert len(data.target_names) == 20

    def test_load_subset_train(self):
        """Train subset should have expected size."""
        data = load_20newsgroups(subset="train")
        # Train set has ~11,314 documents
        assert 10000 < len(data.documents) < 12000

    def test_load_subset_all(self):
        """All subset should be larger than train."""
        train_data = load_20newsgroups(subset="train")
        all_data = load_20newsgroups(subset="all")
        assert len(all_data.documents) > len(train_data.documents)


class TestGetDocumentCount:
    """Tests for document count function."""

    def test_returns_correct_count(self):
        """Should return correct number of documents."""
        data = load_20newsgroups(subset="train")
        count = get_document_count(data)
        assert count == len(data.documents)


class TestGetCategoryDistribution:
    """Tests for category distribution function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        data = load_20newsgroups(subset="train")
        dist = get_category_distribution(data)
        assert isinstance(dist, dict)

    def test_has_all_categories(self):
        """Should have entries for all categories."""
        data = load_20newsgroups(subset="train")
        dist = get_category_distribution(data)
        assert len(dist) == 20

    def test_counts_sum_to_total(self):
        """Category counts should sum to total documents."""
        data = load_20newsgroups(subset="train")
        dist = get_category_distribution(data)
        assert sum(dist.values()) == len(data.documents)
