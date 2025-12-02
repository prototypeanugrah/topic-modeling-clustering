"""EDA Pipeline: Token Distribution Analysis at Each Preprocessing Stage.

This script analyzes token distributions at each stage of the preprocessing pipeline:
1. Raw data (whitespace-split tokens)
2. After preprocessing (tokenization, lemmatization, stopword removal)
3. After dictionary filter_extremes
4. After document filtering (removing docs with insufficient tokens)

Run with: uv run python scripts/eda.py
Options:
    --min-tokens N    Minimum tokens per document after filtering (default: 8)
    --verbose         Show sample documents at each stage
    --save-filtered   Save filtered data for precompute.py to use

All artifacts are saved to cache/eda/ for debugging.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.cache.manager import (
    ensure_cache_dirs,
    load_corpus,
    load_dictionary,
    load_tokenized_docs,
    save_corpus,
    save_dictionary,
    save_document_labels,
    save_eda_stats,
    save_tokenized_docs,
)
from backend.config import (
    CACHE_DIR,
    FILTER_NO_ABOVE,
    FILTER_NO_BELOW,
    MIN_DOC_TOKENS,
    MODELS_DIR,
)
from backend.core.data_loader import load_test_data, load_train_data
from backend.core.lda_trainer import create_corpus
from backend.core.text_preprocessor import preprocess_documents
from gensim import corpora

# EDA artifacts directory
EDA_DIR = CACHE_DIR / "eda"


def ensure_eda_dir() -> Path:
    """Create EDA directory if it doesn't exist."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    return EDA_DIR


def save_artifact(name: str, data: any, stage: int) -> Path:
    """Save an artifact to the EDA directory.

    Args:
        name: Name of the artifact (without extension)
        data: Data to save (dict for JSON, other for pickle)
        stage: Stage number (1-4)

    Returns:
        Path to saved file
    """
    ensure_eda_dir()
    stage_dir = EDA_DIR / f"stage{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(data, dict):
        path = stage_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        path = stage_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)

    return path


def save_token_counts(counts: list[int], name: str, stage: int) -> Path:
    """Save token counts as both JSON stats and raw pickle.

    Args:
        counts: List of token counts per document
        name: Base name for the files
        stage: Stage number

    Returns:
        Path to JSON stats file
    """
    ensure_eda_dir()
    stage_dir = EDA_DIR / f"stage{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Save raw counts as pickle
    counts_path = stage_dir / f"{name}_counts.pkl"
    with open(counts_path, "wb") as f:
        pickle.dump(counts, f)

    # Save as numpy array too for easy analysis
    np_path = stage_dir / f"{name}_counts.npy"
    np.save(np_path, np.array(counts))

    return counts_path


class FilteredData(NamedTuple):
    """Container for filtered data."""

    tokenized_docs: list[list[str]]
    corpus: list[list[tuple[int, int]]]
    kept_indices: list[int]
    removed_indices: list[int]


def compute_stage_stats(lengths: list[int], n_bins: int = 50) -> dict:
    """Compute statistics for one preprocessing stage.

    Args:
        lengths: List of document lengths (token counts)
        n_bins: Number of histogram bins

    Returns:
        Dictionary with statistics and histogram data
    """
    lengths_arr = np.array(lengths)
    n_docs = len(lengths_arr)
    empty_count = int(np.sum(lengths_arr == 0))

    # Compute basic stats
    stats = {
        "n_documents": n_docs,
        "mean": float(np.mean(lengths_arr)),
        "median": float(np.median(lengths_arr)),
        "min": int(np.min(lengths_arr)),
        "max": int(np.max(lengths_arr)),
        "std": float(np.std(lengths_arr)),
        "empty_count": empty_count,
        "empty_pct": float(100 * empty_count / n_docs) if n_docs > 0 else 0.0,
    }

    # Compute percentiles
    percentile_values = [10, 25, 50, 75, 90, 95, 99]
    stats["percentiles"] = {
        str(p): float(np.percentile(lengths_arr, p)) for p in percentile_values
    }

    # Compute histogram (exclude zeros for better visualization, cap at 99th percentile)
    non_zero = lengths_arr[lengths_arr > 0]
    if len(non_zero) > 0:
        max_val = np.percentile(non_zero, 99)  # Cap at 99th percentile
        clipped = np.clip(non_zero, 0, max_val)
        counts, bin_edges = np.histogram(clipped, bins=n_bins)
        stats["histogram_bins"] = [float(b) for b in bin_edges]
        stats["histogram_counts"] = [int(c) for c in counts]
    else:
        stats["histogram_bins"] = [0.0, 1.0]
        stats["histogram_counts"] = [0]

    return stats


def print_stage_stats(name: str, stats: dict, indent: str = "  ") -> None:
    """Pretty print statistics for a stage.

    Args:
        name: Stage name
        stats: Statistics dictionary
        indent: Indentation string
    """
    print(f"\n{name}")
    print("-" * 80)
    print(f"{indent}Documents: {stats['n_documents']:,}")
    print(f"{indent}Mean tokens: {stats['mean']:.1f}")
    print(f"{indent}Median tokens: {stats['median']:.0f}")
    print(f"{indent}Min: {stats['min']}, Max: {stats['max']:,}")
    print(f"{indent}Std: {stats['std']:.1f}")
    print(
        f"{indent}Empty docs: {stats['empty_count']} ({stats['empty_pct']:.2f}%)"
    )
    print(f"\n{indent}Percentiles:")
    p = stats["percentiles"]
    print(
        f"{indent}  10th: {p['10']:.0f}    25th: {p['25']:.0f}    50th: {p['50']:.0f}"
    )
    print(
        f"{indent}  75th: {p['75']:.0f}    90th: {p['90']:.0f}    95th: {p['95']:.0f}    99th: {p['99']:.0f}"
    )


def get_raw_token_counts(documents: list[str]) -> list[int]:
    """Get token counts from raw documents using whitespace split.

    Args:
        documents: List of raw document strings

    Returns:
        List of token counts per document
    """
    return [len(doc.split()) for doc in documents]


def get_corpus_token_counts(corpus: list[list[tuple[int, int]]]) -> list[int]:
    """Get token counts from bag-of-words corpus.

    Args:
        corpus: Bag-of-words corpus

    Returns:
        List of token counts per document
    """
    return [sum(count for _, count in doc) for doc in corpus]


def filter_short_documents(
    tokenized_docs: list[list[str]],
    corpus: list[list[tuple[int, int]]],
    min_tokens: int,
) -> FilteredData:
    """Filter documents with insufficient tokens.

    Args:
        tokenized_docs: List of tokenized documents
        corpus: Bag-of-words corpus
        min_tokens: Minimum tokens required

    Returns:
        FilteredData with filtered documents and indices
    """
    kept_indices = []
    removed_indices = []
    filtered_tokenized = []
    filtered_corpus = []

    for i, (tokens, bow) in enumerate(zip(tokenized_docs, corpus)):
        token_count = sum(count for _, count in bow)
        if token_count >= min_tokens:
            kept_indices.append(i)
            filtered_tokenized.append(tokens)
            filtered_corpus.append(bow)
        else:
            removed_indices.append(i)

    return FilteredData(
        tokenized_docs=filtered_tokenized,
        corpus=filtered_corpus,
        kept_indices=kept_indices,
        removed_indices=removed_indices,
    )


def create_dictionary_with_filter(
    tokenized_docs: list[list[str]],
    no_below: int = FILTER_NO_BELOW,
    no_above: float = FILTER_NO_ABOVE,
) -> corpora.Dictionary:
    """Create dictionary with filter_extremes applied.

    Args:
        tokenized_docs: List of tokenized documents
        no_below: Minimum document frequency
        no_above: Maximum document frequency ratio

    Returns:
        Filtered gensim Dictionary
    """
    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    return dictionary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EDA Pipeline: Token Distribution Analysis"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=MIN_DOC_TOKENS,
        help=f"Minimum tokens per document after filtering (default: {MIN_DOC_TOKENS})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show sample documents at each stage",
    )
    parser.add_argument(
        "--save-filtered",
        action="store_true",
        help="Save filtered data for precompute.py to use",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cache exists",
    )
    return parser.parse_args()


def main():
    """Run the EDA pipeline."""
    args = parse_args()
    min_tokens = args.min_tokens

    print("=" * 80)
    print("EDA Pipeline: Token Distribution Analysis")
    print(f"Minimum tokens per document: {min_tokens}")
    print("=" * 80)

    start_time = time.time()
    ensure_cache_dirs()

    # =========================================================================
    # Load or preprocess data
    # =========================================================================

    # Check for cached preprocessed data
    train_tokenized = load_tokenized_docs("train")
    test_tokenized = load_tokenized_docs("test")

    if train_tokenized and test_tokenized and not args.force:
        print("\n[1/4] Using cached preprocessed data...")
        train_data = load_train_data()
        test_data = load_test_data()
    else:
        print("\n[1/4] Loading and preprocessing data...")
        step_start = time.time()

        train_data = load_train_data()
        test_data = load_test_data()

        print(f"      Train: {len(train_data.documents)} documents")
        print(f"      Test: {len(test_data.documents)} documents")

        print("      Preprocessing train documents...")
        train_tokenized = list(
            preprocess_documents(train_data.documents, show_progress=True)
        )
        save_tokenized_docs(train_tokenized, "train")

        print("      Preprocessing test documents...")
        test_tokenized = list(
            preprocess_documents(test_data.documents, show_progress=True)
        )
        save_tokenized_docs(test_tokenized, "test")

        print(f"      Done in {time.time() - step_start:.1f}s")

    # =========================================================================
    # Stage 1: Raw Documents (whitespace-split tokens)
    # =========================================================================

    print("\n" + "=" * 80)
    print("Stage 1: Raw Documents (whitespace-split tokens)")
    print("=" * 80)

    raw_train_counts = get_raw_token_counts(train_data.documents)
    raw_test_counts = get_raw_token_counts(test_data.documents)

    raw_train_stats = compute_stage_stats(raw_train_counts)
    raw_test_stats = compute_stage_stats(raw_test_counts)

    print_stage_stats("Train Set", raw_train_stats)
    print_stage_stats("Test Set", raw_test_stats)

    # Save Stage 1 artifacts
    save_token_counts(raw_train_counts, "train_raw", stage=1)
    save_token_counts(raw_test_counts, "test_raw", stage=1)
    save_artifact("train_stats", raw_train_stats, stage=1)
    save_artifact("test_stats", raw_test_stats, stage=1)
    print(f"\n  Artifacts saved to {EDA_DIR}/stage1/")

    if args.verbose:
        print("\n  Sample document (raw):")
        sample_idx = 0
        sample_doc = train_data.documents[sample_idx][:500]
        print(f"  '{sample_doc}...'")
        print(f"  Whitespace tokens: {raw_train_counts[sample_idx]}")

    # =========================================================================
    # Stage 2: After Preprocessing (tokenization + lemmatization + stopwords)
    # =========================================================================

    print("\n" + "=" * 80)
    print("Stage 2: After Preprocessing (tokenize + lemmatize + stopwords)")
    print("=" * 80)

    tokenized_train_counts = [len(doc) for doc in train_tokenized]
    tokenized_test_counts = [len(doc) for doc in test_tokenized]

    tokenized_train_stats = compute_stage_stats(tokenized_train_counts)
    tokenized_test_stats = compute_stage_stats(tokenized_test_counts)

    # Compute vocabulary size before filter
    all_tokens_train = [w for doc in train_tokenized for w in doc]
    vocab_before = len(set(all_tokens_train))

    # Compute token frequency distribution
    from collections import Counter
    token_freq = Counter(all_tokens_train)
    top_tokens = token_freq.most_common(100)

    print_stage_stats("Train Set", tokenized_train_stats)
    print_stage_stats("Test Set", tokenized_test_stats)
    print(f"\n  Vocabulary size (train): {vocab_before:,} unique tokens")
    print(f"  Total tokens (train): {len(all_tokens_train):,}")

    # Save Stage 2 artifacts
    save_token_counts(tokenized_train_counts, "train_tokenized", stage=2)
    save_token_counts(tokenized_test_counts, "test_tokenized", stage=2)
    save_artifact("train_stats", tokenized_train_stats, stage=2)
    save_artifact("test_stats", tokenized_test_stats, stage=2)
    save_artifact("vocabulary_info", {
        "vocab_size": vocab_before,
        "total_tokens": len(all_tokens_train),
        "top_100_tokens": top_tokens,
    }, stage=2)
    save_artifact("token_frequencies", dict(token_freq), stage=2)
    print(f"\n  Artifacts saved to {EDA_DIR}/stage2/")

    if args.verbose:
        print("\n  Sample document (preprocessed):")
        sample_tokens = train_tokenized[0][:20]
        print(f"  {sample_tokens}")
        print(f"  Token count: {tokenized_train_counts[0]}")
        print("\n  Top 10 tokens:")
        for token, count in top_tokens[:10]:
            print(f"    {token}: {count:,}")

    # =========================================================================
    # Stage 3: After filter_extremes
    # =========================================================================

    print("\n" + "=" * 80)
    print(
        f"Stage 3: After filter_extremes (no_below={FILTER_NO_BELOW}, no_above={FILTER_NO_ABOVE})"
    )
    print("=" * 80)

    # Create dictionary WITHOUT filter_extremes first (for debugging)
    print("  Creating dictionary before filter_extremes...")
    dictionary_unfiltered = corpora.Dictionary(train_tokenized)
    vocab_unfiltered = len(dictionary_unfiltered)

    # Save unfiltered dictionary
    ensure_eda_dir()
    stage3_dir = EDA_DIR / "stage3"
    stage3_dir.mkdir(parents=True, exist_ok=True)
    dictionary_unfiltered.save(str(stage3_dir / "dictionary_unfiltered.pkl"))

    # Create dictionary WITH filter_extremes
    print("  Applying filter_extremes...")
    dictionary = corpora.Dictionary(train_tokenized)

    # Track which tokens will be removed
    tokens_before = set(dictionary.token2id.keys())

    dictionary.filter_extremes(no_below=FILTER_NO_BELOW, no_above=FILTER_NO_ABOVE)

    tokens_after = set(dictionary.token2id.keys())
    tokens_removed = tokens_before - tokens_after

    # Get document frequencies for removed tokens
    removed_token_info = []
    for token in list(tokens_removed)[:1000]:  # Limit to 1000 for storage
        if token in dictionary_unfiltered.token2id:
            token_id = dictionary_unfiltered.token2id[token]
            doc_freq = dictionary_unfiltered.dfs.get(token_id, 0)
            removed_token_info.append({
                "token": token,
                "doc_freq": doc_freq,
                "doc_freq_pct": 100 * doc_freq / len(train_tokenized),
            })

    # Sort by doc_freq to see what was removed
    removed_token_info.sort(key=lambda x: x["doc_freq"], reverse=True)

    save_dictionary(dictionary)
    dictionary.save(str(stage3_dir / "dictionary_filtered.pkl"))

    vocab_after = len(dictionary)
    vocab_reduction = 100 * (1 - vocab_after / vocab_before) if vocab_before > 0 else 0

    print(f"\n  Vocabulary: {vocab_before:,} -> {vocab_after:,} ({vocab_reduction:.1f}% reduction)")
    print(f"  Tokens removed: {len(tokens_removed):,}")

    # Create corpora
    train_corpus = create_corpus(train_tokenized, dictionary)
    test_corpus = create_corpus(test_tokenized, dictionary)

    # Save corpora
    save_corpus(train_corpus, "train")
    save_corpus(test_corpus, "test")

    # Get token counts after filter
    filtered_train_counts = get_corpus_token_counts(train_corpus)
    filtered_test_counts = get_corpus_token_counts(test_corpus)

    filtered_train_stats = compute_stage_stats(filtered_train_counts)
    filtered_test_stats = compute_stage_stats(filtered_test_counts)

    print_stage_stats("Train Set", filtered_train_stats)
    print_stage_stats("Test Set", filtered_test_stats)

    # Count documents below threshold
    train_below_threshold = sum(1 for c in filtered_train_counts if c < min_tokens)
    test_below_threshold = sum(1 for c in filtered_test_counts if c < min_tokens)

    print(f"\n  Documents with <{min_tokens} tokens:")
    print(
        f"    Train: {train_below_threshold} ({100 * train_below_threshold / len(filtered_train_counts):.2f}%)"
    )
    print(
        f"    Test: {test_below_threshold} ({100 * test_below_threshold / len(filtered_test_counts):.2f}%)"
    )

    # Save Stage 3 artifacts
    save_token_counts(filtered_train_counts, "train_filtered", stage=3)
    save_token_counts(filtered_test_counts, "test_filtered", stage=3)
    save_artifact("train_stats", filtered_train_stats, stage=3)
    save_artifact("test_stats", filtered_test_stats, stage=3)
    save_artifact("filter_info", {
        "no_below": FILTER_NO_BELOW,
        "no_above": FILTER_NO_ABOVE,
        "vocab_before": vocab_before,
        "vocab_after": vocab_after,
        "vocab_reduction_pct": vocab_reduction,
        "tokens_removed_count": len(tokens_removed),
        "removed_tokens_sample": removed_token_info[:100],
    }, stage=3)
    save_artifact("removed_tokens", list(tokens_removed), stage=3)
    save_artifact("docs_below_threshold", {
        "threshold": min_tokens,
        "train_count": train_below_threshold,
        "train_pct": 100 * train_below_threshold / len(filtered_train_counts),
        "test_count": test_below_threshold,
        "test_pct": 100 * test_below_threshold / len(filtered_test_counts),
        "train_indices": [i for i, c in enumerate(filtered_train_counts) if c < min_tokens],
        "test_indices": [i for i, c in enumerate(filtered_test_counts) if c < min_tokens],
    }, stage=3)
    print(f"\n  Artifacts saved to {EDA_DIR}/stage3/")

    # =========================================================================
    # Stage 4: After Document Filtering
    # =========================================================================

    print("\n" + "=" * 80)
    print(f"Stage 4: After Document Filtering (min_tokens={min_tokens})")
    print("=" * 80)

    # Filter documents
    train_filtered = filter_short_documents(
        train_tokenized, train_corpus, min_tokens
    )
    test_filtered = filter_short_documents(
        test_tokenized, test_corpus, min_tokens
    )

    # Get final token counts
    final_train_counts = get_corpus_token_counts(train_filtered.corpus)
    final_test_counts = get_corpus_token_counts(test_filtered.corpus)

    final_train_stats = compute_stage_stats(final_train_counts)
    final_test_stats = compute_stage_stats(final_test_counts)

    print(
        f"\n  Train: {len(train_filtered.kept_indices):,} kept, "
        f"{len(train_filtered.removed_indices):,} removed "
        f"({100 * len(train_filtered.removed_indices) / len(train_corpus):.2f}%)"
    )
    print(
        f"  Test: {len(test_filtered.kept_indices):,} kept, "
        f"{len(test_filtered.removed_indices):,} removed "
        f"({100 * len(test_filtered.removed_indices) / len(test_corpus):.2f}%)"
    )

    print_stage_stats("Train Set (Filtered)", final_train_stats)
    print_stage_stats("Test Set (Filtered)", final_test_stats)

    # Get info about removed documents
    removed_train_info = []
    for idx in train_filtered.removed_indices:
        token_count = filtered_train_counts[idx]
        removed_train_info.append({
            "index": idx,
            "token_count": token_count,
            "original_tokens": len(train_tokenized[idx]),
        })

    removed_test_info = []
    for idx in test_filtered.removed_indices:
        token_count = filtered_test_counts[idx]
        removed_test_info.append({
            "index": idx,
            "token_count": token_count,
            "original_tokens": len(test_tokenized[idx]),
        })

    # Save Stage 4 artifacts
    save_token_counts(final_train_counts, "train_final", stage=4)
    save_token_counts(final_test_counts, "test_final", stage=4)
    save_artifact("train_stats", final_train_stats, stage=4)
    save_artifact("test_stats", final_test_stats, stage=4)
    save_artifact("filtering_info", {
        "min_tokens_threshold": min_tokens,
        "train_kept": len(train_filtered.kept_indices),
        "train_removed": len(train_filtered.removed_indices),
        "train_removed_pct": 100 * len(train_filtered.removed_indices) / len(train_corpus),
        "test_kept": len(test_filtered.kept_indices),
        "test_removed": len(test_filtered.removed_indices),
        "test_removed_pct": 100 * len(test_filtered.removed_indices) / len(test_corpus),
    }, stage=4)
    save_artifact("train_kept_indices", train_filtered.kept_indices, stage=4)
    save_artifact("train_removed_indices", train_filtered.removed_indices, stage=4)
    save_artifact("test_kept_indices", test_filtered.kept_indices, stage=4)
    save_artifact("test_removed_indices", test_filtered.removed_indices, stage=4)
    save_artifact("removed_docs_train", removed_train_info, stage=4)
    save_artifact("removed_docs_test", removed_test_info, stage=4)
    print(f"\n  Artifacts saved to {EDA_DIR}/stage4/")

    # =========================================================================
    # Save EDA stats
    # =========================================================================

    eda_stats = {
        "raw_train": raw_train_stats,
        "raw_test": raw_test_stats,
        "tokenized_train": tokenized_train_stats,
        "tokenized_test": tokenized_test_stats,
        "vocab_before_filter": vocab_before,
        "vocab_after_filter": vocab_after,
        "vocab_reduction_pct": vocab_reduction,
        "filtered_train": filtered_train_stats,
        "filtered_test": filtered_test_stats,
        "final_train": final_train_stats,
        "final_test": final_test_stats,
        "min_tokens_threshold": min_tokens,
        "train_docs_removed": len(train_filtered.removed_indices),
        "test_docs_removed": len(test_filtered.removed_indices),
        "filter_no_below": FILTER_NO_BELOW,
        "filter_no_above": FILTER_NO_ABOVE,
    }
    save_eda_stats(eda_stats)
    print("\n  EDA stats saved to cache/metrics/eda_stats.json")

    # =========================================================================
    # Save filtered data (if requested)
    # =========================================================================

    if args.save_filtered:
        print("\n" + "=" * 80)
        print("Saving filtered data for precompute.py...")
        print("=" * 80)

        # Save filtered tokenized docs
        save_tokenized_docs(train_filtered.tokenized_docs, "train_filtered")
        save_tokenized_docs(test_filtered.tokenized_docs, "test_filtered")

        # Save filtered corpus
        save_corpus(train_filtered.corpus, "train_filtered")
        save_corpus(test_filtered.corpus, "test_filtered")

        # Save indices for reference
        indices_path = MODELS_DIR / "filter_indices.json"
        with open(indices_path, "w") as f:
            json.dump(
                {
                    "train_kept": train_filtered.kept_indices,
                    "train_removed": train_filtered.removed_indices,
                    "test_kept": test_filtered.kept_indices,
                    "test_removed": test_filtered.removed_indices,
                    "min_tokens": min_tokens,
                },
                f,
            )

        # Save newsgroup labels for filtered documents
        train_labels = [
            train_data.target_names[train_data.target[idx]]
            for idx in train_filtered.kept_indices
        ]
        test_labels = [
            test_data.target_names[test_data.target[idx]]
            for idx in test_filtered.kept_indices
        ]
        save_document_labels(train_labels, "train")
        save_document_labels(test_labels, "test")

        print(f"  Saved filtered tokenized docs (train_filtered, test_filtered)")
        print(f"  Saved filtered corpus (train_filtered, test_filtered)")
        print(f"  Saved filter indices to {indices_path}")
        print(f"  Saved document labels (train: {len(train_labels)}, test: {len(test_labels)})")

    # =========================================================================
    # Summary
    # =========================================================================

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("EDA Pipeline Complete!")
    print(f"Total time: {total_time:.1f}s")
    print("=" * 80)

    print("\nSummary:")
    print("-" * 40)
    print(f"  Raw train docs: {raw_train_stats['n_documents']:,}")
    print(f"  After preprocessing: {tokenized_train_stats['n_documents']:,}")
    print(f"  After filter_extremes: {filtered_train_stats['n_documents']:,}")
    print(f"  After doc filtering: {final_train_stats['n_documents']:,}")
    print(f"\n  Vocabulary reduction: {vocab_before:,} -> {vocab_after:,} ({vocab_reduction:.1f}%)")
    print(f"  Documents removed (train): {len(train_filtered.removed_indices)}")
    print(f"  Documents removed (test): {len(test_filtered.removed_indices)}")

    print(f"\nArtifacts saved to: {EDA_DIR}/")
    print("  stage1/ - Raw document token counts")
    print("  stage2/ - Preprocessed token counts and vocabulary")
    print("  stage3/ - Filtered dictionary and corpus info")
    print("  stage4/ - Final filtered document info")

    if not args.save_filtered:
        print("\n  Tip: Run with --save-filtered to save filtered data for precompute.py")


if __name__ == "__main__":
    main()
