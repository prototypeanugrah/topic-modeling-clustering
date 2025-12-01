"""Precomputation script for LDA models with cross-validation.

This script implements proper ML evaluation:
1. Load train/test sets separately
2. Build vocabulary from full train set (frozen)
3. Run 5-fold CV on train set for each topic count
4. Train final model on all train data
5. Evaluate on held-out test set

Run with: uv run python scripts/precompute.py
Test with: uv run python scripts/precompute.py --min-topics 2 --max-topics 4
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import MIN_TOPICS, MAX_TOPICS
from backend.core.data_loader import load_train_data, load_test_data
from backend.core.text_preprocessor import preprocess_documents
from backend.core.lda_trainer import (
    create_dictionary,
    create_corpus,
    train_lda,
    get_doc_topic_distribution,
    calculate_coherence,
    calculate_perplexity,
    run_cross_validation,
)
from backend.core.projections import compute_umap
from backend.cache.manager import (
    ensure_cache_dirs,
    save_dictionary,
    save_corpus,
    save_tokenized_docs,
    save_lda_model,
    save_doc_topic_distribution,
    save_umap_projection,
    save_coherence_scores,
    save_perplexity_scores,
    save_pyldavis_html,
    save_eda_stats,
    load_coherence_scores,
    load_perplexity_scores,
    load_dictionary,
    load_corpus,
    load_tokenized_docs,
    load_lda_model,
    load_doc_topic_distribution,
    load_umap_projection,
    get_pyldavis_path,
    load_eda_stats,
)


def compute_stage_stats(lengths: list[int], n_bins: int = 50) -> dict:
    """Compute statistics for one preprocessing stage.

    Args:
        lengths: List of document lengths (chars or tokens)
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
        "avg_length": float(np.mean(lengths_arr)),
        "median_length": float(np.median(lengths_arr)),
        "min_length": int(np.min(lengths_arr)),
        "max_length": int(np.max(lengths_arr)),
        "std_length": float(np.std(lengths_arr)),
        "empty_count": empty_count,
        "empty_pct": float(100 * empty_count / n_docs) if n_docs > 0 else 0.0,
    }

    # Compute percentiles
    percentile_values = [10, 25, 50, 75, 90, 95, 99]
    stats["percentiles"] = {
        p: float(np.percentile(lengths_arr, p)) for p in percentile_values
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute LDA models with cross-validation"
    )
    parser.add_argument(
        "--min-topics",
        type=int,
        default=MIN_TOPICS,
        help=f"Minimum number of topics (default: {MIN_TOPICS})"
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=MAX_TOPICS,
        help=f"Maximum number of topics (default: {MAX_TOPICS})"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation (only train final models)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cache exists"
    )
    return parser.parse_args()


def main():
    """Run the full precomputation pipeline with CV."""
    args = parse_args()

    min_topics = args.min_topics
    max_topics = args.max_topics
    n_folds = args.n_folds

    print("=" * 70)
    print("Topic Modeling Precomputation Pipeline (with Cross-Validation)")
    print(f"Topics range: {min_topics} to {max_topics}")
    print(f"CV folds: {n_folds}")
    print("=" * 70)

    start_time = time.time()
    ensure_cache_dirs()

    # ==========================================================================
    # STEP 1: Load and preprocess train/test data
    # ==========================================================================

    # Check if preprocessed data exists
    dictionary = load_dictionary()
    train_corpus = load_corpus("train")
    test_corpus = load_corpus("test")
    train_tokenized = load_tokenized_docs("train")
    test_tokenized = load_tokenized_docs("test")

    # Check for cached EDA stats
    eda_stats = load_eda_stats()

    if all([dictionary, train_corpus, test_corpus, train_tokenized, test_tokenized]):
        print("\n[1/8] Using cached preprocessed data...")
        print(f"      Dictionary size: {len(dictionary)} terms")
        print(f"      Train corpus: {len(train_corpus)} documents")
        print(f"      Test corpus: {len(test_corpus)} documents")

        # Compute EDA if not cached
        if eda_stats is None or args.force:
            print("\n[EDA] Computing dataset statistics...")
            # Need to reload raw data for raw lengths
            train_data = load_train_data()
            test_data = load_test_data()

            # Raw lengths (characters)
            raw_train_lengths = [len(doc) for doc in train_data.documents]
            raw_test_lengths = [len(doc) for doc in test_data.documents]

            # Tokenized lengths
            tokenized_train_lengths = [len(doc) for doc in train_tokenized]
            tokenized_test_lengths = [len(doc) for doc in test_tokenized]

            # Filtered lengths (corpus token counts)
            filtered_train_lengths = [sum(count for _, count in doc) for doc in train_corpus]
            filtered_test_lengths = [sum(count for _, count in doc) for doc in test_corpus]

            # Vocab size before filter_extremes
            vocab_before = len(set(w for doc in train_tokenized for w in doc))

            # Compute all stats
            eda_stats = {
                "raw_train": compute_stage_stats(raw_train_lengths),
                "raw_test": compute_stage_stats(raw_test_lengths),
                "vocab_before_filter": vocab_before,
                "tokenized_train": compute_stage_stats(tokenized_train_lengths),
                "tokenized_test": compute_stage_stats(tokenized_test_lengths),
                "vocab_after_filter": len(dictionary),
                "filtered_train": compute_stage_stats(filtered_train_lengths),
                "filtered_test": compute_stage_stats(filtered_test_lengths),
                "vocab_reduction_pct": float(100 * (1 - len(dictionary) / vocab_before)) if vocab_before > 0 else 0.0,
                "token_reduction_pct": float(100 * (1 - sum(filtered_train_lengths) / sum(tokenized_train_lengths))) if sum(tokenized_train_lengths) > 0 else 0.0,
            }
            save_eda_stats(eda_stats)
            print(f"      EDA stats saved")
    else:
        # Load train data
        print("\n[1/8] Loading train/test datasets...")
        step_start = time.time()

        train_data = load_train_data()
        test_data = load_test_data()

        print(f"      Train: {len(train_data.documents)} documents")
        print(f"      Test: {len(test_data.documents)} documents")
        print(f"      Loaded in {time.time() - step_start:.1f}s")

        # Compute raw document lengths for EDA
        raw_train_lengths = [len(doc) for doc in train_data.documents]
        raw_test_lengths = [len(doc) for doc in test_data.documents]

        # Preprocess train data
        print("\n[2/8] Preprocessing train documents...")
        step_start = time.time()
        train_tokenized = list(preprocess_documents(train_data.documents, show_progress=True))
        save_tokenized_docs(train_tokenized, "train")
        print(f"      Preprocessed {len(train_tokenized)} train docs in {time.time() - step_start:.1f}s")

        # Preprocess test data
        print("\n[3/8] Preprocessing test documents...")
        step_start = time.time()
        test_tokenized = list(preprocess_documents(test_data.documents, show_progress=True))
        save_tokenized_docs(test_tokenized, "test")
        print(f"      Preprocessed {len(test_tokenized)} test docs in {time.time() - step_start:.1f}s")

        # Compute tokenized lengths for EDA
        tokenized_train_lengths = [len(doc) for doc in train_tokenized]
        tokenized_test_lengths = [len(doc) for doc in test_tokenized]
        vocab_before = len(set(w for doc in train_tokenized for w in doc))

        # Create dictionary from TRAIN SET ONLY (frozen vocabulary)
        print("\n[4/8] Creating vocabulary from train set (frozen)...")
        step_start = time.time()
        dictionary = create_dictionary(train_tokenized)
        save_dictionary(dictionary)
        print(f"      Dictionary size: {len(dictionary)} terms")

        # Create corpora using frozen dictionary
        train_corpus = create_corpus(train_tokenized, dictionary)
        test_corpus = create_corpus(test_tokenized, dictionary)
        save_corpus(train_corpus, "train")
        save_corpus(test_corpus, "test")
        print(f"      Train corpus: {len(train_corpus)} docs")
        print(f"      Test corpus: {len(test_corpus)} docs")
        print(f"      Created in {time.time() - step_start:.1f}s")

        # Compute filtered lengths for EDA
        filtered_train_lengths = [sum(count for _, count in doc) for doc in train_corpus]
        filtered_test_lengths = [sum(count for _, count in doc) for doc in test_corpus]

        # Save EDA stats
        print("\n[EDA] Computing and saving dataset statistics...")
        eda_stats = {
            "raw_train": compute_stage_stats(raw_train_lengths),
            "raw_test": compute_stage_stats(raw_test_lengths),
            "vocab_before_filter": vocab_before,
            "tokenized_train": compute_stage_stats(tokenized_train_lengths),
            "tokenized_test": compute_stage_stats(tokenized_test_lengths),
            "vocab_after_filter": len(dictionary),
            "filtered_train": compute_stage_stats(filtered_train_lengths),
            "filtered_test": compute_stage_stats(filtered_test_lengths),
            "vocab_reduction_pct": float(100 * (1 - len(dictionary) / vocab_before)) if vocab_before > 0 else 0.0,
            "token_reduction_pct": float(100 * (1 - sum(filtered_train_lengths) / sum(tokenized_train_lengths))) if sum(tokenized_train_lengths) > 0 else 0.0,
        }
        save_eda_stats(eda_stats)
        print(f"      EDA stats saved")

    # ==========================================================================
    # STEP 2: Run Cross-Validation (if not skipped)
    # ==========================================================================

    if not args.skip_cv:
        print(f"\n[5/8] Running {n_folds}-fold Cross-Validation (k={min_topics} to {max_topics})...")

        # Load existing val scores if any
        coherence_val = load_coherence_scores("val") or {}
        perplexity_val = load_perplexity_scores("val") or {}

        for num_topics in range(min_topics, max_topics + 1):
            # Skip if already computed
            if num_topics in coherence_val and num_topics in perplexity_val and not args.force:
                print(f"      k={num_topics}: cached (val_coh={coherence_val[num_topics]:.4f}, val_perp={perplexity_val[num_topics]:.2f})")
                continue

            step_start = time.time()
            print(f"      k={num_topics}: running {n_folds}-fold CV...", end=" ", flush=True)

            # Run cross-validation
            cv_result = run_cross_validation(
                train_tokenized,
                train_corpus,
                dictionary,
                num_topics,
                n_folds=n_folds,
            )

            coherence_val[num_topics] = cv_result.avg_coherence
            perplexity_val[num_topics] = cv_result.avg_perplexity

            elapsed = time.time() - step_start
            print(f"val_coh={cv_result.avg_coherence:.4f}±{cv_result.std_coherence:.4f}, "
                  f"val_perp={cv_result.avg_perplexity:.2f}±{cv_result.std_perplexity:.2f} ({elapsed:.1f}s)")

        # Save validation scores
        save_coherence_scores(coherence_val, "val")
        save_perplexity_scores(perplexity_val, "val")
    else:
        print("\n[5/8] Skipping Cross-Validation...")

    # ==========================================================================
    # STEP 3: Train Final Models on Full Train Set
    # ==========================================================================

    print(f"\n[6/8] Training final models on full train set (k={min_topics} to {max_topics})...")

    # Load existing test scores if any
    coherence_test = load_coherence_scores("test") or {}
    perplexity_test = load_perplexity_scores("test") or {}
    models_trained = 0
    models_skipped = 0

    for num_topics in range(min_topics, max_topics + 1):
        # Check if model and metrics exist
        existing_model = load_lda_model(num_topics)
        existing_train_dist = load_doc_topic_distribution(num_topics, "train")
        existing_test_dist = load_doc_topic_distribution(num_topics, "test")

        all_exist = (
            existing_model is not None and
            existing_train_dist is not None and
            existing_test_dist is not None and
            num_topics in coherence_test and
            num_topics in perplexity_test
        )

        if all_exist and not args.force:
            print(f"      k={num_topics}: cached (test_coh={coherence_test[num_topics]:.4f}, test_perp={perplexity_test[num_topics]:.2f})")
            models_skipped += 1
            continue

        step_start = time.time()
        print(f"      Training k={num_topics}...", end=" ", flush=True)

        # Train final model on ALL train data
        if existing_model is not None and not args.force:
            model = existing_model
        else:
            model = train_lda(train_corpus, dictionary, num_topics)
            save_lda_model(model, num_topics)

        # Get train doc-topic distribution
        if existing_train_dist is None or args.force:
            train_dist = get_doc_topic_distribution(model, train_corpus, num_topics)
            save_doc_topic_distribution(train_dist, num_topics, "train")

        # Get test doc-topic distribution
        if existing_test_dist is None or args.force:
            test_dist = get_doc_topic_distribution(model, test_corpus, num_topics)
            save_doc_topic_distribution(test_dist, num_topics, "test")

        # Calculate test metrics
        if num_topics not in coherence_test or args.force:
            coherence = calculate_coherence(model, test_tokenized, dictionary)
            coherence_test[num_topics] = coherence
        else:
            coherence = coherence_test[num_topics]

        if num_topics not in perplexity_test or args.force:
            perplexity = calculate_perplexity(model, test_corpus)
            perplexity_test[num_topics] = perplexity
        else:
            perplexity = perplexity_test[num_topics]

        elapsed = time.time() - step_start
        print(f"test_coh={coherence:.4f}, test_perp={perplexity:.2f} ({elapsed:.1f}s)")
        models_trained += 1

    # Save test scores
    save_coherence_scores(coherence_test, "test")
    save_perplexity_scores(perplexity_test, "test")
    print(f"      Trained: {models_trained}, Skipped: {models_skipped}")

    # ==========================================================================
    # STEP 4: Compute UMAP Projections for Train and Test
    # ==========================================================================

    print("\n[7/8] Computing UMAP projections...")
    umap_computed = 0
    umap_skipped = 0

    for num_topics in range(min_topics, max_topics + 1):
        for dataset in ["train", "test"]:
            existing_umap = load_umap_projection(num_topics, dataset)
            if existing_umap is not None and not args.force:
                umap_skipped += 1
                continue

            step_start = time.time()
            print(f"      UMAP for k={num_topics} ({dataset})...", end=" ", flush=True)

            # Load doc-topic distribution
            doc_topics = load_doc_topic_distribution(num_topics, dataset)

            # Compute UMAP
            projection = compute_umap(doc_topics)
            save_umap_projection(projection, num_topics, dataset)

            elapsed = time.time() - step_start
            print(f"({elapsed:.1f}s)")
            umap_computed += 1

    print(f"      Computed: {umap_computed}, Skipped: {umap_skipped}")

    # ==========================================================================
    # STEP 5: Generate pyLDAvis Visualizations
    # ==========================================================================

    print("\n[8/8] Generating pyLDAvis visualizations...")
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models

        pyldavis_generated = 0
        pyldavis_skipped = 0

        for num_topics in range(min_topics, max_topics + 1):
            existing_path = get_pyldavis_path(num_topics)
            if existing_path is not None and not args.force:
                pyldavis_skipped += 1
                continue

            step_start = time.time()
            print(f"      pyLDAvis for k={num_topics}...", end=" ", flush=True)

            model = load_lda_model(num_topics)
            vis_data = pyLDAvis.gensim_models.prepare(model, train_corpus, dictionary, sort_topics=False)
            html = pyLDAvis.prepared_data_to_html(vis_data)
            save_pyldavis_html(html, num_topics)

            elapsed = time.time() - step_start
            print(f"({elapsed:.1f}s)")
            pyldavis_generated += 1

        print(f"      Generated: {pyldavis_generated}, Skipped: {pyldavis_skipped}")

    except ImportError:
        print("      pyLDAvis not installed, skipping visualization generation")
    except Exception as e:
        print(f"      Error generating pyLDAvis: {e}")

    # ==========================================================================
    # Summary
    # ==========================================================================

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Precomputation complete!")
    print(f"Total time: {total_time / 60:.1f} minutes")

    # Load final scores
    coherence_val = load_coherence_scores("val") or {}
    perplexity_val = load_perplexity_scores("val") or {}
    coherence_test = load_coherence_scores("test") or {}
    perplexity_test = load_perplexity_scores("test") or {}

    print("\nValidation Scores (5-fold CV averaged):")
    print("-" * 50)
    for k in sorted(coherence_val.keys()):
        coh_val = coherence_val.get(k, 0)
        perp_val = perplexity_val.get(k, 0)
        print(f"  k={k:2d}: coherence={coh_val:.4f}  perplexity={perp_val:8.2f}")

    print("\nTest Scores (held-out):")
    print("-" * 50)
    for k in sorted(coherence_test.keys()):
        coh_test = coherence_test.get(k, 0)
        perp_test = perplexity_test.get(k, 0)
        print(f"  k={k:2d}: coherence={coh_test:.4f}  perplexity={perp_test:8.2f}")

    # Find optimal based on test coherence
    if coherence_test:
        optimal_k = max(coherence_test, key=coherence_test.get)
        print(f"\nOptimal number of topics (by test coherence): {optimal_k}")

    print("=" * 70)


if __name__ == "__main__":
    main()
