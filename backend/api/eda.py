"""EDA (Exploratory Data Analysis) API endpoints."""

from fastapi import APIRouter, HTTPException

from backend.cache.manager import load_eda_stats
from backend.models.responses import EDAResponse, StageStats

router = APIRouter(prefix="/eda", tags=["eda"])


@router.get("", response_model=EDAResponse)
async def get_eda_stats():
    """
    Get dataset EDA statistics at all 4 preprocessing stages.

    Returns statistics for:
    - Stage 1: Raw documents (token counts via whitespace split)
    - Stage 2: Tokenized documents (after preprocessing, before filter_extremes)
    - Stage 3: Filtered corpus (after filter_extremes)
    - Stage 4: Final corpus (after document filtering)

    Includes histogram data for visualization and document removal counts.
    """
    stats = load_eda_stats()

    if stats is None:
        raise HTTPException(
            status_code=503,
            detail="EDA stats not available. Run 'uv run python scripts/eda.py' first."
        )

    # Convert nested dicts to StageStats models
    return EDAResponse(
        # Stage 1: Raw documents
        raw_train=StageStats(**stats["raw_train"]),
        raw_test=StageStats(**stats["raw_test"]),
        # Stage 2: After preprocessing
        vocab_before_filter=stats["vocab_before_filter"],
        tokenized_train=StageStats(**stats["tokenized_train"]),
        tokenized_test=StageStats(**stats["tokenized_test"]),
        # Stage 3: After filter_extremes
        vocab_after_filter=stats["vocab_after_filter"],
        filtered_train=StageStats(**stats["filtered_train"]),
        filtered_test=StageStats(**stats["filtered_test"]),
        vocab_reduction_pct=stats["vocab_reduction_pct"],
        # Stage 4: After document filtering
        final_train=StageStats(**stats["final_train"]),
        final_test=StageStats(**stats["final_test"]),
        min_tokens_threshold=stats["min_tokens_threshold"],
        train_docs_removed=stats["train_docs_removed"],
        test_docs_removed=stats["test_docs_removed"],
        # Filter settings
        filter_no_below=stats["filter_no_below"],
        filter_no_above=stats["filter_no_above"],
    )
