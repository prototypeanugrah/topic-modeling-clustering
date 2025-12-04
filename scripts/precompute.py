"""Precomputation script for LDA models.

This script trains LDA models on the full dataset and computes all required artifacts.

Run with: uv run python scripts/precompute.py
Test with: uv run python scripts/precompute.py --min-topics 2 --max-topics 4
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

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
    load_topic_words,
    load_cluster_metrics,
    load_cluster_labels,
    load_document_enrichment,
    save_coherence_scores,
    save_corpus,
    save_dictionary,
    save_doc_topic_distribution,
    save_lda_model,
    save_pyldavis_html,
    save_tokenized_docs,
    save_umap_projection,
    save_topic_words,
    save_cluster_metrics,
    save_cluster_labels,
    save_document_enrichment,
)
from backend.config import MAX_TOPICS, MIN_TOPICS, MIN_CLUSTERS, MAX_CLUSTERS
from backend.core.clustering import perform_kmeans
from backend.core.metrics import compute_metrics_for_all_clusters
from backend.core.data_loader import load_all_data
from backend.core.lda_trainer import (
    calculate_coherence,
    create_corpus,
    create_dictionary,
    get_doc_topic_distribution,
    train_lda,
)
from backend.core.projections import compute_umap
from backend.core.text_preprocessor import preprocess_documents


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute LDA models"
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
        "--force",
        action="store_true",
        help="Force regeneration even if cache exists",
    )
    return parser.parse_args()


def main():
    """Run the full precomputation pipeline."""
    args = parse_args()

    min_topics = args.min_topics
    max_topics = args.max_topics

    print("=" * 70)
    print("Topic Modeling Precomputation Pipeline")
    print(f"Topics range: {min_topics} to {max_topics}")
    print("=" * 70)

    start_time = time.time()
    ensure_cache_dirs()

    # =========================================================================
    # STEP 1: Load and preprocess data
    # =========================================================================

    # Check if preprocessed data exists
    dictionary = load_dictionary()
    corpus = load_corpus()
    tokenized = load_tokenized_docs()

    if all([dictionary, corpus, tokenized]) and not args.force:
        print("\n[1/5] Using cached preprocessed data...")
        print(f"      Dictionary size: {len(dictionary)} terms")
        print(f"      Corpus: {len(corpus)} documents")
    else:
        # Load all data
        print("\n[1/5] Loading dataset...")
        step_start = time.time()

        data = load_all_data()

        print(f"      Total: {len(data.documents)} documents")
        print(f"      Loaded in {time.time() - step_start:.1f}s")

        # Preprocess data
        print("\n[2/5] Preprocessing documents...")
        step_start = time.time()
        tokenized = list(
            preprocess_documents(data.documents, show_progress=True)
        )
        save_tokenized_docs(tokenized)
        print(
            f"      Preprocessed {len(tokenized)} docs in {time.time() - step_start:.1f}s"
        )

        # Create dictionary
        print("\n[3/5] Creating vocabulary...")
        step_start = time.time()
        dictionary = create_dictionary(tokenized)
        save_dictionary(dictionary)
        print(f"      Dictionary size: {len(dictionary)} terms")

        # Create corpus
        corpus = create_corpus(tokenized, dictionary)
        save_corpus(corpus)
        print(f"      Corpus: {len(corpus)} docs")
        print(f"      Created in {time.time() - step_start:.1f}s")

    # =========================================================================
    # STEP 2: Train LDA Models
    # =========================================================================

    print(
        f"\n[4/5] Training models (k={min_topics} to {max_topics})..."
    )

    # Load existing coherence scores if any
    coherence_scores = load_coherence_scores() or {}
    models_trained = 0
    models_skipped = 0

    for num_topics in range(min_topics, max_topics + 1):
        # Check if model and metrics exist
        existing_model = load_lda_model(num_topics)
        existing_dist = load_doc_topic_distribution(num_topics)

        all_exist = (
            existing_model is not None
            and existing_dist is not None
            and num_topics in coherence_scores
        )

        if all_exist and not args.force:
            print(
                f"      k={num_topics}: cached (coherence={coherence_scores[num_topics]:.4f})"
            )
            models_skipped += 1
            continue

        step_start = time.time()
        print(f"      Training k={num_topics}...", end=" ", flush=True)

        # Train model
        if existing_model is not None and not args.force:
            model = existing_model
        else:
            model = train_lda(corpus, dictionary, num_topics)
            save_lda_model(model, num_topics)

        # Get doc-topic distribution
        if existing_dist is None or args.force:
            dist = get_doc_topic_distribution(model, corpus, num_topics)
            save_doc_topic_distribution(dist, num_topics)

        # Calculate coherence
        if num_topics not in coherence_scores or args.force:
            coherence = calculate_coherence(model, tokenized, dictionary)
            coherence_scores[num_topics] = coherence
        else:
            coherence = coherence_scores[num_topics]

        elapsed = time.time() - step_start
        print(f"coherence={coherence:.4f} ({elapsed:.1f}s)")
        models_trained += 1

    # Save coherence scores
    save_coherence_scores(coherence_scores)
    print(f"      Trained: {models_trained}, Skipped: {models_skipped}")

    # =========================================================================
    # STEP 3: Compute UMAP Projections
    # =========================================================================

    print("\n[5/5] Computing UMAP projections and pyLDAvis...")
    umap_computed = 0
    umap_skipped = 0

    for num_topics in range(min_topics, max_topics + 1):
        existing_umap = load_umap_projection(num_topics)
        if existing_umap is not None and not args.force:
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

    print(f"      UMAP - Computed: {umap_computed}, Skipped: {umap_skipped}")

    # =========================================================================
    # STEP 4: Generate pyLDAvis Visualizations
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
                model, corpus, dictionary, sort_topics=False
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
    # STEP 5: Precompute Topic Words
    # =========================================================================

    print("\n[6/9] Precomputing topic words...")
    words_computed = 0
    words_skipped = 0

    for num_topics in range(min_topics, max_topics + 1):
        existing = load_topic_words(num_topics)
        if existing is not None and not args.force:
            words_skipped += 1
            continue

        model = load_lda_model(num_topics)
        if model is None:
            print(f"      Warning: Model k={num_topics} not found, skipping")
            continue

        # Extract top 10 words per topic
        topics = []
        for topic_id in range(num_topics):
            topic_words = model.show_topic(topic_id, topn=10)
            topics.append([
                {"word": word, "probability": round(float(prob), 6)}
                for word, prob in topic_words
            ])

        save_topic_words({"n_topics": num_topics, "topics": topics}, num_topics)
        words_computed += 1

    print(f"      Topic words - Computed: {words_computed}, Skipped: {words_skipped}")

    # =========================================================================
    # STEP 6: Precompute Cluster Metrics
    # =========================================================================

    print("\n[7/9] Precomputing cluster metrics...")
    metrics_computed = 0
    metrics_skipped = 0

    for num_topics in range(min_topics, max_topics + 1):
        existing = load_cluster_metrics(num_topics)
        if existing is not None and not args.force:
            metrics_skipped += 1
            continue

        distribution = load_doc_topic_distribution(num_topics)
        if distribution is None:
            print(f"      Warning: Distribution k={num_topics} not found, skipping")
            continue

        step_start = time.time()
        print(f"      k={num_topics}...", end=" ", flush=True)

        metrics = compute_metrics_for_all_clusters(
            distribution,
            min_clusters=MIN_CLUSTERS,
            max_clusters=MAX_CLUSTERS,
        )

        save_cluster_metrics({
            "n_topics": num_topics,
            "cluster_counts": metrics["cluster_counts"],
            "silhouette_scores": [round(float(s), 6) for s in metrics["silhouette_scores"]],
            "inertia_scores": [round(float(i), 2) for i in metrics["inertia_scores"]],
            "elbow_point": metrics["elbow_point"],
        }, num_topics)

        elapsed = time.time() - step_start
        print(f"({elapsed:.1f}s)")
        metrics_computed += 1

    print(f"      Cluster metrics - Computed: {metrics_computed}, Skipped: {metrics_skipped}")

    # =========================================================================
    # STEP 7: Precompute Cluster Labels
    # =========================================================================

    print("\n[8/9] Precomputing cluster labels...")
    labels_computed = 0
    labels_skipped = 0

    for num_topics in range(min_topics, max_topics + 1):
        distribution = load_doc_topic_distribution(num_topics)
        if distribution is None:
            continue

        for n_clusters in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
            existing = load_cluster_labels(num_topics, n_clusters)
            if existing is not None and not args.force:
                labels_skipped += 1
                continue

            result = perform_kmeans(distribution, n_clusters)
            save_cluster_labels(result.labels, num_topics, n_clusters)
            labels_computed += 1

    print(f"      Cluster labels - Computed: {labels_computed}, Skipped: {labels_skipped}")

    # =========================================================================
    # STEP 8: Precompute Document Enrichment
    # =========================================================================

    print("\n[9/9] Precomputing document enrichment...")
    enrichment_computed = 0
    enrichment_skipped = 0

    for num_topics in range(min_topics, max_topics + 1):
        existing = load_document_enrichment(num_topics)
        if existing is not None and not args.force:
            enrichment_skipped += 1
            continue

        model = load_lda_model(num_topics)
        distribution = load_doc_topic_distribution(num_topics)

        if model is None or distribution is None:
            continue

        step_start = time.time()
        print(f"      k={num_topics}...", end=" ", flush=True)

        # Compute top 3 topics for each document
        top_3_indices = np.argsort(distribution, axis=1)[:, -3:][:, ::-1]
        top_topics = []
        for i, doc_dist in enumerate(distribution):
            doc_top_topics = [
                {"topic_id": int(idx), "probability": round(float(doc_dist[idx]), 4)}
                for idx in top_3_indices[i]
            ]
            top_topics.append(doc_top_topics)

        # Get top 5 words for each topic
        topic_word_cache = {}
        for topic_id in range(num_topics):
            words = model.show_topic(topic_id, topn=5)
            topic_word_cache[topic_id] = [word for word, _ in words]

        # Map each document to its dominant topic's words
        dominant_topics = np.argmax(distribution, axis=1)
        dominant_topic_words = [
            topic_word_cache[int(topic_id)] for topic_id in dominant_topics
        ]

        save_document_enrichment({
            "n_topics": num_topics,
            "top_topics": top_topics,
            "dominant_topic_words": dominant_topic_words,
        }, num_topics)

        elapsed = time.time() - step_start
        print(f"({elapsed:.1f}s)")
        enrichment_computed += 1

    print(f"      Document enrichment - Computed: {enrichment_computed}, Skipped: {enrichment_skipped}")

    # =========================================================================
    # Summary
    # =========================================================================

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Precomputation complete!")
    print(f"Total time: {total_time / 60:.1f} minutes")

    # Load final scores
    coherence_scores = load_coherence_scores() or {}

    print("\nCoherence Scores:")
    print("-" * 40)
    for k in sorted(coherence_scores.keys()):
        coh = coherence_scores.get(k, 0)
        print(f"  k={k:2d}: coherence={coh:.4f}")

    # Find optimal based on coherence
    if coherence_scores:
        optimal_k = max(coherence_scores, key=coherence_scores.get)
        print(f"\nOptimal number of topics: {optimal_k}")

    print("=" * 70)


if __name__ == "__main__":
    main()
