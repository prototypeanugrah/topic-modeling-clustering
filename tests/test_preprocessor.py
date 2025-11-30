"""Tests for text_preprocessor module."""

import pytest
from backend.core.text_preprocessor import (
    clean_text,
    tokenize_and_lemmatize,
    preprocess_document,
    preprocess_documents,
)


class TestCleanText:
    """Tests for text cleaning function."""

    def test_removes_email(self):
        """Should remove email addresses."""
        text = "Contact us at test@example.com for more info."
        cleaned = clean_text(text)
        assert "@" not in cleaned
        assert "example.com" not in cleaned

    def test_removes_urls(self):
        """Should remove URLs."""
        text = "Visit https://example.com or www.test.org"
        cleaned = clean_text(text)
        assert "http" not in cleaned
        assert "www" not in cleaned

    def test_removes_special_chars(self):
        """Should remove special characters."""
        text = "Hello! How are you? Fine, thanks."
        cleaned = clean_text(text)
        assert "!" not in cleaned
        assert "?" not in cleaned
        assert "," not in cleaned

    def test_removes_digits(self):
        """Should remove digits."""
        text = "There are 123 items in the list."
        cleaned = clean_text(text)
        assert "123" not in cleaned

    def test_lowercases(self):
        """Should convert to lowercase."""
        text = "HELLO World"
        cleaned = clean_text(text)
        assert cleaned == "hello world"

    def test_normalizes_whitespace(self):
        """Should normalize whitespace."""
        text = "Multiple   spaces   here"
        cleaned = clean_text(text)
        assert "  " not in cleaned


class TestTokenizeAndLemmatize:
    """Tests for tokenization and lemmatization."""

    def test_returns_list(self):
        """Should return a list of tokens."""
        tokens = tokenize_and_lemmatize("The quick brown fox jumps")
        assert isinstance(tokens, list)

    def test_removes_stopwords(self):
        """Should remove stopwords."""
        tokens = tokenize_and_lemmatize("the a an is are was were")
        # Most common stopwords should be removed
        assert "the" not in tokens
        assert "is" not in tokens

    def test_lemmatizes_words(self):
        """Should lemmatize words."""
        tokens = tokenize_and_lemmatize("running jumped swimming")
        # Should get base forms
        assert "run" in tokens or "running" in tokens
        assert "jump" in tokens or "jumped" in tokens

    def test_removes_short_tokens(self):
        """Should remove tokens shorter than MIN_TOKEN_LENGTH."""
        tokens = tokenize_and_lemmatize("a ab abc word longer")
        # 'a' and 'ab' should be filtered (< 3 chars)
        assert all(len(t) >= 3 for t in tokens)


class TestPreprocessDocument:
    """Tests for full preprocessing pipeline."""

    def test_combines_cleaning_and_tokenizing(self):
        """Should clean and tokenize in one call."""
        text = "Hello! Visit https://example.com for more INFO."
        tokens = preprocess_document(text)
        assert isinstance(tokens, list)
        # Should have removed URL and lowercased
        assert "https" not in tokens
        assert all(t.islower() for t in tokens)


class TestPreprocessDocuments:
    """Tests for batch preprocessing."""

    def test_processes_multiple_docs(self, sample_documents):
        """Should process multiple documents."""
        results = list(preprocess_documents(sample_documents))
        assert len(results) == len(sample_documents)
        assert all(isinstance(doc, list) for doc in results)

    def test_yields_results(self, sample_documents):
        """Should be a generator."""
        gen = preprocess_documents(sample_documents)
        # Should be able to iterate
        first = next(gen)
        assert isinstance(first, list)
