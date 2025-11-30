"""Text preprocessing utilities using spaCy."""

import re
import contractions
import spacy
from spacy.language import Language
from typing import Generator
from tqdm import tqdm

from backend.config import MIN_TOKEN_LENGTH, MAX_TOKEN_LENGTH, CUSTOM_STOPWORDS_FILE


# Global spaCy model (lazy loaded)
_nlp: Language | None = None
# Global custom stopwords set
_custom_stopwords: set[str] | None = None


def load_custom_stopwords() -> set[str]:
    """
    Load custom stopwords from file.

    The file should contain one stopword per line.
    Lines starting with # are treated as comments.
    Empty lines are ignored.

    Returns:
        Set of custom stopwords (lowercase)
    """
    global _custom_stopwords
    if _custom_stopwords is not None:
        return _custom_stopwords

    _custom_stopwords = set()

    if CUSTOM_STOPWORDS_FILE.exists():
        with open(CUSTOM_STOPWORDS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                # Skip empty lines and comments
                if word and not word.startswith("#"):
                    _custom_stopwords.add(word)
        if _custom_stopwords:
            print(f"      Loaded {len(_custom_stopwords)} custom stopwords from {CUSTOM_STOPWORDS_FILE.name}")

    return _custom_stopwords


def get_nlp() -> Language:
    """Get or initialize the spaCy model with custom stopwords."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
        except OSError:
            # Fallback to small model if md not available
            try:
                _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except OSError:
                import subprocess
                try:
                    subprocess.run(
                        ["python", "-m", "spacy", "download", "en_core_web_sm"],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(
                        f"Failed to download spaCy model 'en_core_web_sm'. "
                        f"Please install manually: python -m spacy download en_core_web_sm\n"
                        f"Error: {e.stderr or e.stdout or str(e)}"
                    ) from e
                except OSError as e:
                    raise RuntimeError(
                        f"Failed to load spaCy model after download. Error: {e}"
                    ) from e

        # Add custom stopwords to spaCy's stopword list
        custom_stops = load_custom_stopwords()
        for word in custom_stops:
            _nlp.vocab[word].is_stop = True

    return _nlp


def clean_text(text: str) -> str:
    """
    Clean raw text by removing noise.

    Args:
        text: Raw document text

    Returns:
        Cleaned text
    """
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Expand contractions (e.g., "don't" -> "do not", "I'll" -> "I will")
    text = contractions.fix(text)
    # Remove special characters and digits (keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def tokenize_and_lemmatize(text: str) -> list[str]:
    """
    Tokenize and lemmatize a single document.

    Args:
        text: Cleaned text

    Returns:
        List of lemmatized tokens
    """
    nlp = get_nlp()
    doc = nlp(text)

    tokens = []
    for token in doc:
        # Skip stopwords, punctuation, and short/long tokens
        if (
            token.is_stop
            or token.is_punct
            or token.is_space
            or len(token.lemma_) < MIN_TOKEN_LENGTH
            or len(token.lemma_) > MAX_TOKEN_LENGTH
        ):
            continue

        lemma = token.lemma_.lower().strip()
        if lemma:
            tokens.append(lemma)

    return tokens


def preprocess_document(text: str) -> list[str]:
    """
    Full preprocessing pipeline for a single document.

    Args:
        text: Raw document text

    Returns:
        List of preprocessed tokens
    """
    cleaned = clean_text(text)
    return tokenize_and_lemmatize(cleaned)


def preprocess_documents(
    documents: list[str],
    batch_size: int = 100,
    show_progress: bool = True,
) -> Generator[list[str], None, None]:
    """
    Preprocess multiple documents using spaCy's pipe for efficiency.

    Args:
        documents: List of raw document texts
        batch_size: Batch size for spaCy pipe
        show_progress: Whether to show tqdm progress bar

    Yields:
        Lists of preprocessed tokens for each document
    """
    nlp = get_nlp()

    # Clean all documents first
    print("      Cleaning text...")
    cleaned_docs = [clean_text(doc) for doc in tqdm(documents, disable=not show_progress, desc="      Cleaning")]

    # Process in batches with progress bar
    print("      Tokenizing and lemmatizing...")
    doc_iter = nlp.pipe(cleaned_docs, batch_size=batch_size)
    if show_progress:
        doc_iter = tqdm(doc_iter, total=len(cleaned_docs), desc="      Processing")

    for doc in doc_iter:
        tokens = []
        for token in doc:
            if (
                token.is_stop
                or token.is_punct
                or token.is_space
                or len(token.lemma_) < MIN_TOKEN_LENGTH
                or len(token.lemma_) > MAX_TOKEN_LENGTH
            ):
                continue

            lemma = token.lemma_.lower().strip()
            if lemma:
                tokens.append(lemma)

        yield tokens
