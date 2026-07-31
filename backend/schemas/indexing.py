from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ParentChunkRecord(BaseModel):
    """Schema for a persistent parent chunk preserving larger contextual sections."""

    parent_id: str
    document_id: str
    workspace_id: str
    section_title: str
    page_start: int
    page_end: int
    text: str
    content_type: str = "text"  # 'text', 'table', 'mixed'
    child_ids: List[str] = Field(default_factory=list)
    visibility_scope: str = "company"
    owner_user_id: Optional[str] = None


class ChildChunkRecord(BaseModel):
    """Schema for a child chunk used for vector indexing in ChromaDB."""

    chunk_id: str
    parent_id: str
    document_id: str
    workspace_id: str
    original_filename: str
    page_start: int
    page_end: int
    section_title: str
    text: str
    chunk_type: str = "text"  # 'text' or 'table'
    token_count: int
    file_hash: str
    embedding_model: str
    document_version: int = 1
    visibility_scope: str = "company"
    owner_user_id: Optional[str] = None


class DocumentIndexResponse(BaseModel):
    """API response model for document indexing metadata."""

    id: str
    document_id: str
    status: str  # 'pending', 'indexing', 'completed', 'failed'
    total_parents: int
    total_children: int
    embedding_model: str
    embedding_dimension: int
    parent_storage_path: Optional[str] = None
    indexed_file_hash: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentIndexStatusResponse(BaseModel):
    """Comprehensive status response combining extraction and vector indexing state."""

    document_id: str
    workspace_id: str
    original_filename: str
    extraction_status: str
    total_pages: Optional[int] = None
    index: Optional[DocumentIndexResponse] = None
    is_search_ready: bool
