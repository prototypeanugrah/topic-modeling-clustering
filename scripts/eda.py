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
    save_box_plot_data,
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
from backend.core.data_loader import load_all_data
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

    # Compute histogram (exclude zeros for better visualization, cap at 95th percentile)
    non_zero = lengths_arr[lengths_arr > 0]
    if len(non_zero) > 0:
        max_val = np.percentile(non_zero, 95)  # Cap at 95th percentile
        capped = np.minimum(non_zero, max_val)  # Cap values above 95th percentile
        min_val = capped.min()
        counts, bin_edges = np.histogram(capped, bins=n_bins, range=(min_val, max_val))
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
    # Load and preprocess data
    # =========================================================================

    # Always load raw data and preprocess for EDA
    # (Don't use cached tokenized docs - they might be filtered from a previous run)
    print("\n[1/4] Loading and preprocessing data...")
    step_start = time.time()

    data = load_all_data()
    print(f"      Total: {len(data.documents)} documents")

    print("      Preprocessing documents...")
    tokenized = list(
        preprocess_documents(data.documents, show_progress=True)
    )

    print(f"      Done in {time.time() - step_start:.1f}s")

    # =========================================================================
    # Stage 1: Raw Documents (whitespace-split tokens)
    # =========================================================================

    print("\n" + "=" * 80)
    print("Stage 1: Raw Documents (whitespace-split tokens)")
    print("=" * 80)

    raw_counts = get_raw_token_counts(data.documents)
    raw_stats = compute_stage_stats(raw_counts)

    print_stage_stats("All Documents", raw_stats)

    # Save Stage 1 artifacts
    save_token_counts(raw_counts, "raw", stage=1)
    save_artifact("stats", raw_stats, stage=1)
    print(f"\n  Artifacts saved to {EDA_DIR}/stage1/")

    if args.verbose:
        print("\n  Sample document (raw):")
        sample_idx = 0
        sample_doc = data.documents[sample_idx][:500]
        print(f"  '{sample_doc}...'")
        print(f"  Whitespace tokens: {raw_counts[sample_idx]}")

    # =========================================================================
    # Stage 2: After Preprocessing (tokenization + lemmatization + stopwords)
    # =========================================================================

    print("\n" + "=" * 80)
    print("Stage 2: After Preprocessing (tokenize + lemmatize + stopwords)")
    print("=" * 80)

    tokenized_counts = [len(doc) for doc in tokenized]
    tokenized_stats = compute_stage_stats(tokenized_counts)

    # Compute vocabulary size before filter
    all_tokens = [w for doc in tokenized for w in doc]
    vocab_before = len(set(all_tokens))

    # Compute token frequency distribution
    from collections import Counter
    token_freq = Counter(all_tokens)
    top_tokens = token_freq.most_common(100)

    print_stage_stats("All Documents", tokenized_stats)
    print(f"\n  Vocabulary size: {vocab_before:,} unique tokens")
    print(f"  Total tokens: {len(all_tokens):,}")

    # Save Stage 2 artifacts
    save_token_counts(tokenized_counts, "tokenized", stage=2)
    save_artifact("stats", tokenized_stats, stage=2)
    save_artifact("vocabulary_info", {
        "vocab_size": vocab_before,
        "total_tokens": len(all_tokens),
        "top_100_tokens": top_tokens,
    }, stage=2)
    save_artifact("token_frequencies", dict(token_freq), stage=2)
    print(f"\n  Artifacts saved to {EDA_DIR}/stage2/")

    if args.verbose:
        print("\n  Sample document (preprocessed):")
        sample_tokens = tokenized[0][:20]
        print(f"  {sample_tokens}")
        print(f"  Token count: {tokenized_counts[0]}")
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
    dictionary_unfiltered = corpora.Dictionary(tokenized)
    vocab_unfiltered = len(dictionary_unfiltered)

    # Save unfiltered dictionary
    ensure_eda_dir()
    stage3_dir = EDA_DIR / "stage3"
    stage3_dir.mkdir(parents=True, exist_ok=True)
    dictionary_unfiltered.save(str(stage3_dir / "dictionary_unfiltered.pkl"))

    # Create dictionary WITH filter_extremes
    print("  Applying filter_extremes...")
    dictionary = corpora.Dictionary(tokenized)

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
                "doc_freq_pct": 100 * doc_freq / len(tokenized),
            })

    # Sort by doc_freq to see what was removed
    removed_token_info.sort(key=lambda x: x["doc_freq"], reverse=True)

    save_dictionary(dictionary)
    dictionary.save(str(stage3_dir / "dictionary_filtered.pkl"))

    vocab_after = len(dictionary)
    vocab_reduction = 100 * (1 - vocab_after / vocab_before) if vocab_before > 0 else 0

    print(f"\n  Vocabulary: {vocab_before:,} -> {vocab_after:,} ({vocab_reduction:.1f}% reduction)")
    print(f"  Tokens removed: {len(tokens_removed):,}")

    # Create corpus
    corpus = create_corpus(tokenized, dictionary)

    # Save corpus
    save_corpus(corpus)

    # Get token counts after filter
    filtered_counts = get_corpus_token_counts(corpus)
    filtered_stats = compute_stage_stats(filtered_counts)

    print_stage_stats("All Documents", filtered_stats)

    # Count documents below threshold
    below_threshold = sum(1 for c in filtered_counts if c < min_tokens)

    print(f"\n  Documents with <{min_tokens} tokens:")
    print(
        f"    Count: {below_threshold} ({100 * below_threshold / len(filtered_counts):.2f}%)"
    )

    # Save Stage 3 artifacts
    save_token_counts(filtered_counts, "filtered", stage=3)
    save_artifact("stats", filtered_stats, stage=3)
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
        "count": below_threshold,
        "pct": 100 * below_threshold / len(filtered_counts),
        "indices": [i for i, c in enumerate(filtered_counts) if c < min_tokens],
    }, stage=3)
    print(f"\n  Artifacts saved to {EDA_DIR}/stage3/")

    # =========================================================================
    # Stage 4: After Document Filtering
    # =========================================================================

    print("\n" + "=" * 80)
    print(f"Stage 4: After Document Filtering (min_tokens={min_tokens})")
    print("=" * 80)

    # Filter documents
    filtered_data = filter_short_documents(tokenized, corpus, min_tokens)

    # Get final token counts
    final_counts = get_corpus_token_counts(filtered_data.corpus)
    final_stats = compute_stage_stats(final_counts)

    print(
        f"\n  Documents: {len(filtered_data.kept_indices):,} kept, "
        f"{len(filtered_data.removed_indices):,} removed "
        f"({100 * len(filtered_data.removed_indices) / len(corpus):.2f}%)"
    )

    print_stage_stats("Filtered Documents", final_stats)

    # Get info about removed documents
    removed_info = []
    for idx in filtered_data.removed_indices:
        token_count = filtered_counts[idx]
        removed_info.append({
            "index": idx,
            "token_count": token_count,
            "original_tokens": len(tokenized[idx]),
        })

    # Save Stage 4 artifacts
    save_token_counts(final_counts, "final", stage=4)
    save_artifact("stats", final_stats, stage=4)
    save_artifact("filtering_info", {
        "min_tokens_threshold": min_tokens,
        "kept": len(filtered_data.kept_indices),
        "removed": len(filtered_data.removed_indices),
        "removed_pct": 100 * len(filtered_data.removed_indices) / len(corpus),
    }, stage=4)
    save_artifact("kept_indices", filtered_data.kept_indices, stage=4)
    save_artifact("removed_indices", filtered_data.removed_indices, stage=4)
    save_artifact("removed_docs", removed_info, stage=4)
    print(f"\n  Artifacts saved to {EDA_DIR}/stage4/")

    # =========================================================================
    # Save EDA stats
    # =========================================================================

    eda_stats = {
        "raw": raw_stats,
        "tokenized": tokenized_stats,
        "vocab_before_filter": vocab_before,
        "vocab_after_filter": vocab_after,
        "vocab_reduction_pct": vocab_reduction,
        "filtered": filtered_stats,
        "final": final_stats,
        "min_tokens_threshold": min_tokens,
        "docs_removed": len(filtered_data.removed_indices),
        "filter_no_below": FILTER_NO_BELOW,
        "filter_no_above": FILTER_NO_ABOVE,
    }
    save_eda_stats(eda_stats)
    print("\n  EDA stats saved to cache/metrics/eda_stats.json")

    # =========================================================================
    # Save box plot data
    # =========================================================================

    print("\n  Generating box plot data...")

    # Collect token counts by preprocessing stage
    stage_token_counts = {
        "raw": raw_counts,
        "tokenized": tokenized_counts,
        "filtered": filtered_counts,
        "final": final_counts,
    }

    # Collect token counts by newsgroup category (using final stage counts)
    category_token_counts: dict[str, list[int]] = {}
    for idx, count in zip(filtered_data.kept_indices, final_counts):
        category = data.target_names[data.target[idx]]
        if category not in category_token_counts:
            category_token_counts[category] = []
        category_token_counts[category].append(count)

    box_plot_data = {
        "stage_token_counts": stage_token_counts,
        "category_token_counts": category_token_counts,
    }
    save_box_plot_data(box_plot_data)
    print("  Box plot data saved to cache/metrics/box_plot_data.json")

    # =========================================================================
    # Save filtered data (if requested)
    # =========================================================================

    if args.save_filtered:
        print("\n" + "=" * 80)
        print("Saving filtered data for precompute.py...")
        print("=" * 80)

        # Save filtered tokenized docs
        save_tokenized_docs(filtered_data.tokenized_docs)

        # Save filtered corpus
        save_corpus(filtered_data.corpus)

        # Save indices for reference
        indices_path = MODELS_DIR / "filter_indices.json"
        with open(indices_path, "w") as f:
            json.dump(
                {
                    "kept": filtered_data.kept_indices,
                    "removed": filtered_data.removed_indices,
                    "min_tokens": min_tokens,
                },
                f,
            )

        # Save newsgroup labels for filtered documents
        labels = [
            data.target_names[data.target[idx]]
            for idx in filtered_data.kept_indices
        ]
        save_document_labels(labels)

        print(f"  Saved filtered tokenized docs")
        print(f"  Saved filtered corpus")
        print(f"  Saved filter indices to {indices_path}")
        print(f"  Saved document labels ({len(labels)} documents)")

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
    print(f"  Raw docs: {raw_stats['n_documents']:,}")
    print(f"  After preprocessing: {tokenized_stats['n_documents']:,}")
    print(f"  After filter_extremes: {filtered_stats['n_documents']:,}")
    print(f"  After doc filtering: {final_stats['n_documents']:,}")
    print(f"\n  Vocabulary reduction: {vocab_before:,} -> {vocab_after:,} ({vocab_reduction:.1f}%)")
    print(f"  Documents removed: {len(filtered_data.removed_indices)}")

    print(f"\nArtifacts saved to: {EDA_DIR}/")
    print("  stage1/ - Raw document token counts")
    print("  stage2/ - Preprocessed token counts and vocabulary")
    print("  stage3/ - Filtered dictionary and corpus info")
    print("  stage4/ - Final filtered document info")

    if not args.save_filtered:
        print("\n  Tip: Run with --save-filtered to save filtered data for precompute.py")


if __name__ == "__main__":
    main()
