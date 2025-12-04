"""EDA (Exploratory Data Analysis) API endpoints."""

from fastapi import APIRouter, HTTPException

from backend.cache.manager import load_box_plot_data, load_eda_stats
from backend.models.responses import BoxPlotData, EDAResponse, StageStats

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
        raw=StageStats(**stats["raw"]),
        # Stage 2: After preprocessing
        vocab_before_filter=stats["vocab_before_filter"],
        tokenized=StageStats(**stats["tokenized"]),
        # Stage 3: After filter_extremes
        vocab_after_filter=stats["vocab_after_filter"],
        filtered=StageStats(**stats["filtered"]),
        vocab_reduction_pct=stats["vocab_reduction_pct"],
        # Stage 4: After document filtering
        final=StageStats(**stats["final"]),
        min_tokens_threshold=stats["min_tokens_threshold"],
        docs_removed=stats["docs_removed"],
        # Filter settings
        filter_no_below=stats["filter_no_below"],
        filter_no_above=stats["filter_no_above"],
    )


@router.get("/boxplot", response_model=BoxPlotData)
async def get_box_plot_data():
    """
    Get raw token count data for box plot visualizations.

    Returns token counts arrays for:
    - Each preprocessing stage (raw, tokenized, filtered, final)
    - Each newsgroup category (20 categories)

    This data is used by the frontend to render Plotly box plots.
    """
    data = load_box_plot_data()

    if data is None:
        raise HTTPException(
            status_code=503,
            detail="Box plot data not available. Run 'uv run python scripts/eda.py' first.",
        )

    return BoxPlotData(**data)
