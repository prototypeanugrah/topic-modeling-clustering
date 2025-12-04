"""LDA topic model training using Gensim."""

from typing import NamedTuple

import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

from backend.config import (
    FILTER_NO_ABOVE,
    FILTER_NO_BELOW,
    LDA_ALPHA,
    LDA_CHUNKSIZE,
    LDA_ETA,
    LDA_ITERATIONS,
    LDA_PASSES,
    LDA_RANDOM_STATE,
)


class LDAResult(NamedTuple):
    """Container for LDA training results."""

    model: LdaModel
    dictionary: corpora.Dictionary
    corpus: list[list[tuple[int, int]]]
    doc_topic_distribution: np.ndarray
    coherence_score: float


def create_dictionary(tokenized_docs: list[list[str]]) -> corpora.Dictionary:
    """
    Create a Gensim dictionary from tokenized documents.

    Args:
        tokenized_docs: List of tokenized documents

    Returns:
        Gensim Dictionary
    """
    dictionary = corpora.Dictionary(tokenized_docs)

    # Filter extremes: remove tokens appearing in fewer than FILTER_NO_BELOW docs
    # or more than FILTER_NO_ABOVE fraction of docs
    dictionary.filter_extremes(no_below=FILTER_NO_BELOW, no_above=FILTER_NO_ABOVE)

    return dictionary


def create_corpus(
    tokenized_docs: list[list[str]], dictionary: corpora.Dictionary
) -> list[list[tuple[int, int]]]:
    """
    Create a bag-of-words corpus from tokenized documents.

    Args:
        tokenized_docs: List of tokenized documents
        dictionary: Gensim Dictionary

    Returns:
        List of bag-of-words vectors
    """
    return [dictionary.doc2bow(doc) for doc in tokenized_docs]


def train_lda(
    corpus: list[list[tuple[int, int]]],
    dictionary: corpora.Dictionary,
    num_topics: int,
    passes: int = LDA_PASSES,
    iterations: int = LDA_ITERATIONS,
    chunksize: int = LDA_CHUNKSIZE,
    random_state: int = LDA_RANDOM_STATE,
    alpha: str | float | list[float] = LDA_ALPHA,
    eta: str | float | list[float] = LDA_ETA,
) -> LdaModel:
    """
    Train an LDA model.

    Args:
        corpus: Bag-of-words corpus
        dictionary: Gensim Dictionary
        num_topics: Number of topics to extract
        passes: Number of passes through the corpus
        iterations: Maximum iterations per document
        chunksize: Number of documents per chunk
        random_state: Random seed for reproducibility
        alpha: Document-topic density ('auto', 'symmetric', 'asymmetric', or float/list)
        eta: Topic-word density ('auto', 'symmetric', or float/list)

    Returns:
        Trained LDA model
    """
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations,
        chunksize=chunksize,
        random_state=random_state,
        alpha=alpha,
        eta=eta,
        per_word_topics=False,
    )
    return model


def get_doc_topic_distribution(
    model: LdaModel,
    corpus: list[list[tuple[int, int]]],
    num_topics: int,
) -> np.ndarray:
    """
    Get document-topic distribution matrix.

    Args:
        model: Trained LDA model
        corpus: Bag-of-words corpus
        num_topics: Number of topics in the model

    Returns:
        NumPy array of shape (num_docs, num_topics)
    """
    num_docs = len(corpus)
    doc_topics = np.zeros((num_docs, num_topics))

    for i, bow in enumerate(corpus):
        topic_dist = model.get_document_topics(bow, minimum_probability=0.0)
        for topic_id, prob in topic_dist:
            doc_topics[i, topic_id] = prob

    return doc_topics


def calculate_coherence(
    model: LdaModel,
    tokenized_docs: list[list[str]],
    dictionary: corpora.Dictionary,
    coherence_type: str = "c_v",
) -> float:
    """
    Calculate topic coherence score.

    Args:
        model: Trained LDA model
        tokenized_docs: List of tokenized documents
        dictionary: Gensim Dictionary
        coherence_type: Type of coherence metric ('c_v', 'u_mass', etc.)

    Returns:
        Coherence score
    """
    coherence_model = CoherenceModel(
        model=model,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence=coherence_type,
    )
    return coherence_model.get_coherence()


def get_topic_words(
    model: LdaModel, num_words: int = 10
) -> list[list[tuple[str, float]]]:
    """
    Get top words for each topic.

    Args:
        model: Trained LDA model
        num_words: Number of top words per topic

    Returns:
        List of (word, probability) tuples for each topic
    """
    topics = []
    for topic_id in range(model.num_topics):
        topic_words = model.show_topic(topic_id, topn=num_words)
        topics.append(topic_words)
    return topics


def train_lda_full(
    tokenized_docs: list[list[str]],
    num_topics: int,
    dictionary: corpora.Dictionary | None = None,
    corpus: list[list[tuple[int, int]]] | None = None,
) -> LDAResult:
    """
    Full LDA training pipeline.

    Args:
        tokenized_docs: List of tokenized documents
        num_topics: Number of topics to extract
        dictionary: Pre-computed dictionary (optional)
        corpus: Pre-computed corpus (optional)

    Returns:
        LDAResult with model, dictionary, corpus, distributions, and coherence
    """
    # Create or use existing dictionary and corpus
    if dictionary is None:
        dictionary = create_dictionary(tokenized_docs)
    if corpus is None:
        corpus = create_corpus(tokenized_docs, dictionary)

    # Train model
    model = train_lda(corpus, dictionary, num_topics)

    # Get document-topic distributions
    doc_topic_dist = get_doc_topic_distribution(model, corpus, num_topics)

    # Calculate coherence
    coherence = calculate_coherence(model, tokenized_docs, dictionary)

    return LDAResult(
        model=model,
        dictionary=dictionary,
        corpus=corpus,
        doc_topic_distribution=doc_topic_dist,
        coherence_score=coherence,
    )


