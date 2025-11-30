"""Precomputation script for LDA models and projections.

Run with: uv run python scripts/precompute.py
Test with: uv run python scripts/precompute.py --min-topics 2 --max-topics 4
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import MIN_TOPICS, MAX_TOPICS
from backend.core.data_loader import load_20newsgroups
from backend.core.text_preprocessor import preprocess_documents
from backend.core.lda_trainer import (
    create_dictionary,
    create_corpus,
    train_lda,
    get_doc_topic_distribution,
    calculate_coherence,
    calculate_perplexity,
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
    load_coherence_scores,
    load_perplexity_scores,
    load_dictionary,
    load_corpus,
    load_tokenized_docs,
    load_lda_model,
    load_doc_topic_distribution,
    load_umap_projection,
    get_pyldavis_path,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute LDA models and UMAP projections"
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
        "--force",
        action="store_true",
        help="Force regeneration even if cache exists"
    )
    return parser.parse_args()


def main():
    """Run the full precomputation pipeline."""
    args = parse_args()

    min_topics = args.min_topics
    max_topics = args.max_topics

    print("=" * 60)
    print("Topic Modeling Precomputation Pipeline")
    print(f"Topics range: {min_topics} to {max_topics}")
    print("=" * 60)

    start_time = time.time()
    ensure_cache_dirs()

    # Check if dictionary/corpus already exist
    dictionary = load_dictionary()
    corpus = load_corpus()
    tokenized_docs = load_tokenized_docs()

    if dictionary is not None and corpus is not None and tokenized_docs is not None:
        print("\n[1-3/7] Using cached dictionary, corpus, and tokenized docs...")
        print(f"      Dictionary size: {len(dictionary)} terms")
        print(f"      Corpus size: {len(corpus)} documents")
    else:
        # Step 1: Load data
        print("\n[1/7] Loading 20 Newsgroups dataset...")
        step_start = time.time()
        data = load_20newsgroups()
        print(f"      Loaded {len(data.documents)} documents in {time.time() - step_start:.1f}s")

        # Step 2: Preprocess
        print("\n[2/7] Preprocessing documents...")
        step_start = time.time()
        tokenized_docs = list(preprocess_documents(data.documents, show_progress=True))
        save_tokenized_docs(tokenized_docs)
        print(f"      Done! Preprocessed {len(tokenized_docs)} documents in {time.time() - step_start:.1f}s")

        # Step 3: Create dictionary and corpus
        print("\n[3/7] Creating dictionary and corpus...")
        step_start = time.time()
        dictionary = create_dictionary(tokenized_docs)
        corpus = create_corpus(tokenized_docs, dictionary)
        save_dictionary(dictionary)
        save_corpus(corpus)
        print(f"      Dictionary size: {len(dictionary)} terms")
        print(f"      Created in {time.time() - step_start:.1f}s")

    # Step 4: Train LDA models for specified topic range
    print(f"\n[4/7] Training LDA models (k={min_topics} to {max_topics})...")

    # Load existing scores if any
    coherence_scores = load_coherence_scores() or {}
    perplexity_scores = load_perplexity_scores() or {}
    models_trained = 0
    models_skipped = 0

    for num_topics in range(min_topics, max_topics + 1):
        # Check if model already exists
        existing_model = load_lda_model(num_topics)
        existing_dist = load_doc_topic_distribution(num_topics)

        if existing_model is not None and existing_dist is not None and num_topics in coherence_scores and num_topics in perplexity_scores:
            print(f"      k={num_topics}: cached (coherence={coherence_scores[num_topics]:.4f}, perplexity={perplexity_scores[num_topics]:.2f})")
            models_skipped += 1
            continue

        step_start = time.time()
        print(f"      Training k={num_topics}...", end=" ", flush=True)

        # Train model (or load existing)
        if existing_model is not None:
            model = existing_model
        else:
            model = train_lda(corpus, dictionary, num_topics)
            save_lda_model(model, num_topics)

        # Get doc-topic distribution (or skip if exists)
        if existing_dist is None:
            doc_topics = get_doc_topic_distribution(model, corpus, num_topics)
            save_doc_topic_distribution(doc_topics, num_topics)

        # Calculate coherence if not already computed
        if num_topics not in coherence_scores:
            coherence = calculate_coherence(model, tokenized_docs, dictionary)
            coherence_scores[num_topics] = coherence
        else:
            coherence = coherence_scores[num_topics]

        # Calculate perplexity if not already computed
        if num_topics not in perplexity_scores:
            perplexity = calculate_perplexity(model, corpus)
            perplexity_scores[num_topics] = perplexity
        else:
            perplexity = perplexity_scores[num_topics]

        elapsed = time.time() - step_start
        print(f"coherence={coherence:.4f}, perplexity={perplexity:.2f} ({elapsed:.1f}s)")
        models_trained += 1

    # Save scores
    save_coherence_scores(coherence_scores)
    save_perplexity_scores(perplexity_scores)
    print(f"      Trained: {models_trained}, Skipped: {models_skipped}")

    # Step 5: Compute UMAP projections
    print("\n[5/7] Computing UMAP projections...")
    umap_computed = 0
    umap_skipped = 0

    for num_topics in range(min_topics, max_topics + 1):
        # Check if UMAP already exists
        existing_umap = load_umap_projection(num_topics)
        if existing_umap is not None:
            print(f"      k={num_topics}: cached")
            umap_skipped += 1
            continue

        step_start = time.time()
        print(f"      UMAP for k={num_topics}...", end=" ", flush=True)

        # Load doc-topic distribution
        doc_topics = load_doc_topic_distribution(num_topics)

        # Compute UMAP
        projection = compute_umap(doc_topics)
        save_umap_projection(projection, num_topics)

        elapsed = time.time() - step_start
        print(f"({elapsed:.1f}s)")
        umap_computed += 1

    print(f"      Computed: {umap_computed}, Skipped: {umap_skipped}")

    # Step 6: Generate pyLDAvis visualizations
    print("\n[6/7] Generating pyLDAvis visualizations...")
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models

        pyldavis_generated = 0
        pyldavis_skipped = 0

        for num_topics in range(min_topics, max_topics + 1):
            # Check if pyLDAvis already exists
            existing_path = get_pyldavis_path(num_topics)
            if existing_path is not None:
                print(f"      k={num_topics}: cached")
                pyldavis_skipped += 1
                continue

            step_start = time.time()
            print(f"      pyLDAvis for k={num_topics}...", end=" ", flush=True)

            # Load model
            model = load_lda_model(num_topics)

            # Prepare visualization
            vis_data = pyLDAvis.gensim_models.prepare(model, corpus, dictionary, sort_topics=False)

            # Convert to HTML
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

    # Step 7: Summary
    total_time = time.time() - start_time
    print("\n[7/7] Precomputation complete!")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print("\nCoherence & Perplexity scores:")
    for k in sorted(coherence_scores.keys()):
        coh = coherence_scores[k]
        perp = perplexity_scores.get(k, 0)
        bar = "#" * int(coh * 50)
        print(f"  k={k:2d}: coherence={coh:.4f} perplexity={perp:8.2f} {bar}")

    # Find optimal
    optimal_k = max(coherence_scores, key=coherence_scores.get)
    print(f"\nOptimal number of topics (by coherence): {optimal_k}")
    print("=" * 60)


if __name__ == "__main__":
    main()
