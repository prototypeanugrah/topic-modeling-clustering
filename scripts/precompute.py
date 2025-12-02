"""Precomputation script for LDA models with cross-validation.

This script implements proper ML evaluation:
1. Load train/test sets (or filtered versions from EDA pipeline)
2. Build vocabulary from full train set (frozen)
3. Run 5-fold CV on train set for each topic count
4. Train final model on all train data
5. Evaluate on held-out test set

Run with: uv run python scripts/precompute.py
Test with: uv run python scripts/precompute.py --min-topics 2 --max-topics 4

To use filtered data from EDA pipeline:
    uv run python scripts/eda.py --save-filtered
    uv run python scripts/precompute.py --use-filtered
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.cache.manager import (
    ensure_cache_dirs,
    get_pyldavis_path,
    load_coherence_scores,
    load_corpus,
    load_dictionary,
    load_doc_topic_distribution,
    load_lda_model,
    load_tokenized_docs,
    load_umap_projection,
    save_coherence_scores,
    save_corpus,
    save_dictionary,
    save_doc_topic_distribution,
    save_lda_model,
    save_pyldavis_html,
    save_tokenized_docs,
    save_umap_projection,
)
from backend.config import MAX_TOPICS, MIN_TOPICS
from backend.core.data_loader import load_test_data, load_train_data
from backend.core.lda_trainer import (
    calculate_coherence,
    create_corpus,
    create_dictionary,
    get_doc_topic_distribution,
    run_cross_validation,
    train_lda,
)
from backend.core.projections import compute_umap
from backend.core.text_preprocessor import preprocess_documents


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute LDA models with cross-validation"
    )
    parser.add_argument(
        "--min-topics",
        type=int,
        default=MIN_TOPICS,
        help=f"Minimum number of topics (default: {MIN_TOPICS})",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=MAX_TOPICS,
        help=f"Maximum number of topics (default: {MAX_TOPICS})",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation (only train final models)",
    )
    parser.add_argument(
        "--use-filtered",
        action="store_true",
        help="Use filtered data from EDA pipeline (run eda.py --save-filtered first)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cache exists",
    )
    return parser.parse_args()


def main():
    """Run the full precomputation pipeline with CV."""
    args = parse_args()

    min_topics = args.min_topics
    max_topics = args.max_topics
    n_folds = args.n_folds
    use_filtered = args.use_filtered

    # Auto-force regeneration when using filtered data to avoid stale cache
    if use_filtered and not args.force:
        args.force = True

    print("=" * 70)
    print("Topic Modeling Precomputation Pipeline (with Cross-Validation)")
    print(f"Topics range: {min_topics} to {max_topics}")
    print(f"CV folds: {n_folds}")
    if use_filtered:
        print("Using filtered data from EDA pipeline (forcing regeneration)")
    print("=" * 70)

    start_time = time.time()
    ensure_cache_dirs()

    # =========================================================================
    # STEP 1: Load and preprocess train/test data
    # =========================================================================

    # Determine which dataset to use
    if use_filtered:
        train_dataset = "train_filtered"
        test_dataset = "test_filtered"
    else:
        train_dataset = "train"
        test_dataset = "test"

    # Check if preprocessed data exists
    dictionary = load_dictionary()
    train_corpus = load_corpus(train_dataset)
    test_corpus = load_corpus(test_dataset)
    train_tokenized = load_tokenized_docs(train_dataset)
    test_tokenized = load_tokenized_docs(test_dataset)

    if all([dictionary, train_corpus, test_corpus, train_tokenized, test_tokenized]):
        print(f"\n[1/7] Using cached preprocessed data ({train_dataset})...")
        print(f"      Dictionary size: {len(dictionary)} terms")
        print(f"      Train corpus: {len(train_corpus)} documents")
        print(f"      Test corpus: {len(test_corpus)} documents")
    else:
        if use_filtered:
            print("\n[ERROR] Filtered data not found!")
            print("        Run 'uv run python scripts/eda.py --save-filtered' first.")
            sys.exit(1)

        # Load train data
        print("\n[1/7] Loading train/test datasets...")
        step_start = time.time()

        train_data = load_train_data()
        test_data = load_test_data()

        print(f"      Train: {len(train_data.documents)} documents")
        print(f"      Test: {len(test_data.documents)} documents")
        print(f"      Loaded in {time.time() - step_start:.1f}s")

        # Preprocess train data
        print("\n[2/7] Preprocessing train documents...")
        step_start = time.time()
        train_tokenized = list(
            preprocess_documents(train_data.documents, show_progress=True)
        )
        save_tokenized_docs(train_tokenized, "train")
        print(
            f"      Preprocessed {len(train_tokenized)} train docs in {time.time() - step_start:.1f}s"
        )

        # Preprocess test data
        print("\n[3/7] Preprocessing test documents...")
        step_start = time.time()
        test_tokenized = list(
            preprocess_documents(test_data.documents, show_progress=True)
        )
        save_tokenized_docs(test_tokenized, "test")
        print(
            f"      Preprocessed {len(test_tokenized)} test docs in {time.time() - step_start:.1f}s"
        )

        # Create dictionary from TRAIN SET ONLY (frozen vocabulary)
        print("\n[4/7] Creating vocabulary from train set (frozen)...")
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

    # =========================================================================
    # STEP 2: Run Cross-Validation (if not skipped)
    # =========================================================================

    if not args.skip_cv:
        print(
            f"\n[5/7] Running {n_folds}-fold Cross-Validation (k={min_topics} to {max_topics})..."
        )

        # Load existing val scores if any
        coherence_val = load_coherence_scores("val") or {}

        for num_topics in range(min_topics, max_topics + 1):
            # Skip if already computed
            if num_topics in coherence_val and not args.force:
                print(
                    f"      k={num_topics}: cached (val_coh={coherence_val[num_topics]:.4f})"
                )
                continue

            step_start = time.time()
            print(
                f"      k={num_topics}: running {n_folds}-fold CV...",
                end=" ",
                flush=True,
            )

            # Run cross-validation
            cv_result = run_cross_validation(
                train_tokenized,
                train_corpus,
                dictionary,
                num_topics,
                n_folds=n_folds,
            )

            coherence_val[num_topics] = cv_result.avg_coherence

            elapsed = time.time() - step_start
            print(
                f"val_coh={cv_result.avg_coherence:.4f}±{cv_result.std_coherence:.4f} ({elapsed:.1f}s)"
            )

        # Save validation scores
        save_coherence_scores(coherence_val, "val")
    else:
        print("\n[5/7] Skipping Cross-Validation...")

    # =========================================================================
    # STEP 3: Train Final Models on Full Train Set
    # =========================================================================

    print(
        f"\n[6/7] Training final models on full train set (k={min_topics} to {max_topics})..."
    )

    # Load existing test scores if any
    coherence_test = load_coherence_scores("test") or {}
    models_trained = 0
    models_skipped = 0

    for num_topics in range(min_topics, max_topics + 1):
        # Check if model and metrics exist
        existing_model = load_lda_model(num_topics)
        existing_train_dist = load_doc_topic_distribution(num_topics, "train")
        existing_test_dist = load_doc_topic_distribution(num_topics, "test")

        all_exist = (
            existing_model is not None
            and existing_train_dist is not None
            and existing_test_dist is not None
            and num_topics in coherence_test
        )

        if all_exist and not args.force:
            print(
                f"      k={num_topics}: cached (test_coh={coherence_test[num_topics]:.4f})"
            )
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
        # Coherence uses train set for sufficient word co-occurrence stats
        if num_topics not in coherence_test or args.force:
            coherence = calculate_coherence(model, train_tokenized, dictionary)
            coherence_test[num_topics] = coherence
        else:
            coherence = coherence_test[num_topics]

        elapsed = time.time() - step_start
        print(f"test_coh={coherence:.4f} ({elapsed:.1f}s)")
        models_trained += 1

    # Save test scores
    save_coherence_scores(coherence_test, "test")
    print(f"      Trained: {models_trained}, Skipped: {models_skipped}")

    # =========================================================================
    # STEP 4: Compute UMAP Projections for Train and Test
    # =========================================================================

    print("\n[7/7] Computing UMAP projections and pyLDAvis...")
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

    print(f"      UMAP - Computed: {umap_computed}, Skipped: {umap_skipped}")

    # =========================================================================
    # STEP 5: Generate pyLDAvis Visualizations
    # =========================================================================

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
            vis_data = pyLDAvis.gensim_models.prepare(
                model, train_corpus, dictionary, sort_topics=False
            )
            html = pyLDAvis.prepared_data_to_html(vis_data)
            save_pyldavis_html(html, num_topics)

            elapsed = time.time() - step_start
            print(f"({elapsed:.1f}s)")
            pyldavis_generated += 1

        print(f"      pyLDAvis - Generated: {pyldavis_generated}, Skipped: {pyldavis_skipped}")

    except ImportError:
        print("      pyLDAvis not installed, skipping visualization generation")
    except Exception as e:
        print(f"      Error generating pyLDAvis: {e}")

    # =========================================================================
    # Summary
    # =========================================================================

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Precomputation complete!")
    print(f"Total time: {total_time / 60:.1f} minutes")

    # Load final scores
    coherence_val = load_coherence_scores("val") or {}
    coherence_test = load_coherence_scores("test") or {}

    print("\nValidation Scores (5-fold CV averaged):")
    print("-" * 40)
    for k in sorted(coherence_val.keys()):
        coh_val = coherence_val.get(k, 0)
        print(f"  k={k:2d}: coherence={coh_val:.4f}")

    print("\nTest Scores (held-out):")
    print("-" * 40)
    for k in sorted(coherence_test.keys()):
        coh_test = coherence_test.get(k, 0)
        print(f"  k={k:2d}: coherence={coh_test:.4f}")

    # Find optimal based on test coherence
    if coherence_test:
        optimal_k = max(coherence_test, key=coherence_test.get)
        print(f"\nOptimal number of topics (by test coherence): {optimal_k}")

    print("=" * 70)


if __name__ == "__main__":
    main()
