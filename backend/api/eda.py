"""EDA (Exploratory Data Analysis) API endpoints."""

from fastapi import APIRouter, HTTPException

from backend.cache.manager import load_eda_stats
from backend.models.responses import EDAResponse, StageStats

router = APIRouter(prefix="/eda", tags=["eda"])


@router.get("", response_model=EDAResponse)
async def get_eda_stats():
    """
    Get dataset EDA statistics at all preprocessing stages.

    Returns statistics for:
    - Raw documents (character lengths)
    - Tokenized documents (before filter_extremes)
    - Filtered corpus (after filter_extremes)

    Includes histogram data for visualization.
    """
    stats = load_eda_stats()

    if stats is None:
        raise HTTPException(
            status_code=503,
            detail="EDA stats not available. Run precomputation first."
        )

    # Convert nested dicts to StageStats models
    return EDAResponse(
        raw_train=StageStats(**stats["raw_train"]),
        raw_test=StageStats(**stats["raw_test"]),
        vocab_before_filter=stats["vocab_before_filter"],
        tokenized_train=StageStats(**stats["tokenized_train"]),
        tokenized_test=StageStats(**stats["tokenized_test"]),
        vocab_after_filter=stats["vocab_after_filter"],
        filtered_train=StageStats(**stats["filtered_train"]),
        filtered_test=StageStats(**stats["filtered_test"]),
        vocab_reduction_pct=stats["vocab_reduction_pct"],
        token_reduction_pct=stats["token_reduction_pct"],
    )
