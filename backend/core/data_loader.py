"""Data loading utilities for 20 Newsgroups dataset."""

from sklearn.datasets import fetch_20newsgroups
from typing import NamedTuple


class NewsGroupsData(NamedTuple):
    """Container for 20 Newsgroups dataset."""
    documents: list[str]
    target: list[int]
    target_names: list[str]


def load_20newsgroups(subset: str = "all", remove_headers: bool = True) -> NewsGroupsData:
    """
    Load the 20 Newsgroups dataset.

    Args:
        subset: Which subset to load ('train', 'test', or 'all')
        remove_headers: Whether to remove headers, footers, and quotes

    Returns:
        NewsGroupsData with documents, target labels, and category names
    """
    remove = ("headers", "footers", "quotes") if remove_headers else ()

    dataset = fetch_20newsgroups(
        subset=subset,
        remove=remove,
        shuffle=True,
        random_state=42
    )

    return NewsGroupsData(
        documents=list(dataset.data),
        target=list(dataset.target),
        target_names=list(dataset.target_names)
    )


def get_document_count(data: NewsGroupsData) -> int:
    """Get the number of documents in the dataset."""
    return len(data.documents)


def get_category_distribution(data: NewsGroupsData) -> dict[str, int]:
    """Get the distribution of documents across categories."""
    distribution = {}
    for label in data.target:
        category = data.target_names[label]
        distribution[category] = distribution.get(category, 0) + 1
    return distribution


def load_all_data(remove_headers: bool = True) -> NewsGroupsData:
    """Load the complete dataset (18,846 documents)."""
    return load_20newsgroups(subset="all", remove_headers=remove_headers)
