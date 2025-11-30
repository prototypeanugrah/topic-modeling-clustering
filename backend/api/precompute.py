"""Precomputation control API endpoints."""

import asyncio
from fastapi import APIRouter, BackgroundTasks

from backend.cache.manager import get_cache_status, is_cache_complete
from backend.models.responses import PrecomputeProgressResponse

router = APIRouter(prefix="/precompute", tags=["precompute"])

# Global state for precomputation progress
_precompute_state = {
    "in_progress": False,
    "current_step": None,
    "progress_percent": 0,
    "completed_topics": [],
    "error": None,
}


def _reset_state():
    """Reset precomputation state."""
    global _precompute_state
    _precompute_state = {
        "in_progress": False,
        "current_step": None,
        "progress_percent": 0,
        "completed_topics": [],
        "error": None,
    }


async def _run_precomputation():
    """Run precomputation in background."""
    import sys
    from pathlib import Path

    # Add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from backend.config import MIN_TOPICS, MAX_TOPICS
    from backend.core.data_loader import load_20newsgroups
    from backend.core.text_preprocessor import preprocess_documents
    from backend.core.lda_trainer import (
        create_dictionary,
        create_corpus,
        train_lda,
        get_doc_topic_distribution,
        calculate_coherence,
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
        load_tokenized_docs,
        load_dictionary,
        load_corpus,
    )

    global _precompute_state

    try:
        _precompute_state["in_progress"] = True
        ensure_cache_dirs()

        total_steps = MAX_TOPICS - MIN_TOPICS + 1
        completed = 0

        # Step 1: Load and preprocess data (if not cached)
        _precompute_state["current_step"] = "Loading dataset"
        _precompute_state["progress_percent"] = 5

        tokenized_docs = load_tokenized_docs()
        if tokenized_docs is None:
            data = load_20newsgroups()
            await asyncio.sleep(0)  # Yield control

            _precompute_state["current_step"] = "Preprocessing documents"
            _precompute_state["progress_percent"] = 10

            tokenized_docs = list(preprocess_documents(data.documents))
            save_tokenized_docs(tokenized_docs)

        await asyncio.sleep(0)

        # Step 2: Create dictionary and corpus (if not cached)
        _precompute_state["current_step"] = "Creating dictionary"
        _precompute_state["progress_percent"] = 15

        dictionary = load_dictionary()
        corpus = load_corpus()

        if dictionary is None:
            dictionary = create_dictionary(tokenized_docs)
            save_dictionary(dictionary)

        if corpus is None:
            corpus = create_corpus(tokenized_docs, dictionary)
            save_corpus(corpus)

        await asyncio.sleep(0)

        # Step 3: Train LDA models
        coherence_scores = {}
        base_progress = 20
        lda_progress_share = 60  # 60% of progress for LDA

        for num_topics in range(MIN_TOPICS, MAX_TOPICS + 1):
            _precompute_state["current_step"] = f"Training LDA (k={num_topics})"

            model = train_lda(corpus, dictionary, num_topics)
            save_lda_model(model, num_topics)

            doc_topics = get_doc_topic_distribution(model, corpus, num_topics)
            save_doc_topic_distribution(doc_topics, num_topics)

            coherence = calculate_coherence(model, tokenized_docs, dictionary)
            coherence_scores[num_topics] = coherence

            completed += 1
            _precompute_state["completed_topics"].append(num_topics)
            _precompute_state["progress_percent"] = base_progress + int(
                (completed / total_steps) * lda_progress_share
            )

            await asyncio.sleep(0)  # Yield control

        save_coherence_scores(coherence_scores)

        # Step 4: Compute UMAP projections
        umap_base = 80
        umap_progress_share = 18

        for i, num_topics in enumerate(range(MIN_TOPICS, MAX_TOPICS + 1)):
            _precompute_state["current_step"] = f"Computing UMAP (k={num_topics})"

            from backend.cache.manager import load_doc_topic_distribution
            doc_topics = load_doc_topic_distribution(num_topics)
            projection = compute_umap(doc_topics)
            save_umap_projection(projection, num_topics)

            _precompute_state["progress_percent"] = umap_base + int(
                ((i + 1) / total_steps) * umap_progress_share
            )

            await asyncio.sleep(0)

        _precompute_state["current_step"] = "Complete"
        _precompute_state["progress_percent"] = 100
        _precompute_state["in_progress"] = False

    except Exception as e:
        _precompute_state["error"] = str(e)
        _precompute_state["in_progress"] = False


@router.post("/start")
async def start_precomputation(background_tasks: BackgroundTasks):
    """
    Start precomputation in the background.

    Returns immediately. Use /precompute/progress to monitor progress.
    """
    if _precompute_state["in_progress"]:
        return {"status": "already_running", "message": "Precomputation is already in progress"}

    if is_cache_complete():
        return {"status": "already_complete", "message": "Cache is already complete"}

    _reset_state()
    background_tasks.add_task(_run_precomputation)

    return {"status": "started", "message": "Precomputation started in background"}


@router.get("/progress", response_model=PrecomputeProgressResponse)
async def get_precomputation_progress():
    """
    Get current precomputation progress.

    Poll this endpoint to monitor precomputation status.
    """
    return PrecomputeProgressResponse(**_precompute_state)
