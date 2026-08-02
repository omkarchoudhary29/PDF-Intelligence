from typing import List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Schema for hybrid document search request within a workspace."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language question, search query, or technical identifier keyword",
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of final reranked results to return",
    )
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of document UUIDs to filter search results within workspace",
    )
    chunk_types: Optional[List[str]] = Field(
        default=None,
        description="Optional list of chunk types to filter by (e.g., 'text', 'table')",
    )
    include_parent: bool = Field(
        default=True,
        description="Whether to recover and attach broader parent context text for each result",
    )
    include_historical: bool = Field(
        default=False,
        description="Whether to include historical (superseded/archived) document versions. Default False (current only)",
    )
    family_id: Optional[str] = Field(
        default=None,
        description="Optional filter by document family UUID",
    )
    version_status: Optional[str] = Field(
        default=None,
        description="Optional filter by version status ('current', 'superseded', 'archived')",
    )
    scope: Optional[str] = Field(
        default="company",
        description="Knowledge scope: 'company', 'personal', or 'combined'",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Authenticated user UUID for personal or combined scope filtering",
    )


class SearchResultItem(BaseModel):
    """Schema for a single retrieved, fused, and reranked chunk result with full provenance."""

    chunk_id: str
    parent_id: str
    document_id: str
    workspace_id: str
    original_filename: str
    page_start: int
    page_end: int
    section_title: str
    chunk_type: str
    text: str
    parent_text: Optional[str] = None
    visibility_scope: Optional[str] = "company"
    owner_user_id: Optional[str] = None
    family_id: Optional[str] = None
    family_title: Optional[str] = None
    version_number: int = 1
    version_status: str = "current"
    is_current: bool = True

    # Ranking and scoring diagnostics
    dense_rank: Optional[int] = None
    dense_distance: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    rrf_score: float
    rrf_rank: int
    cross_encoder_score: Optional[float] = None
    final_rank: int


class SearchResponse(BaseModel):
    """Schema for the final hybrid retrieval response."""

    query: str
    workspace_id: str
    total_results: int
    retrieval_method: str = "hybrid_rrf_cross_encoder"
    results: List[SearchResultItem]
