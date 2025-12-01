"""Data loading utilities for 20 Newsgroups dataset."""

from sklearn.datasets import fetch_20newsgroups
from typing import NamedTuple


class NewsGroupsData(NamedTuple):
    """Container for 20 Newsgroups dataset."""
    documents: list[str]
    target: list[int]
    target_names: list[str]


class TrainTestData(NamedTuple):
    """Container for train and test splits."""
    train: NewsGroupsData
    test: NewsGroupsData


def load_20newsgroups(subset: str = "train", remove_headers: bool = True) -> NewsGroupsData:
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


def load_train_data(remove_headers: bool = True) -> NewsGroupsData:
    """Load the training set (11,314 documents)."""
    return load_20newsgroups(subset="train", remove_headers=remove_headers)


def load_test_data(remove_headers: bool = True) -> NewsGroupsData:
    """Load the test set (7,532 documents)."""
    return load_20newsgroups(subset="test", remove_headers=remove_headers)


def load_train_test_data(remove_headers: bool = True) -> TrainTestData:
    """
    Load both train and test sets.

    Returns:
        TrainTestData with train (11,314 docs) and test (7,532 docs)
    """
    return TrainTestData(
        train=load_train_data(remove_headers=remove_headers),
        test=load_test_data(remove_headers=remove_headers),
    )
