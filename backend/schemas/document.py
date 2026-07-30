from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentResponse(BaseModel):
    """Public document metadata representation.

    Note: storage_key and internal absolute paths are explicitly excluded.
    """

    id: str
    workspace_id: str
    original_filename: str
    display_name: str
    sha256_hash: str
    file_size_bytes: int
    mime_type: str
    total_pages: Optional[int] = None
    processing_status: str
    error_message: Optional[str] = None
    family_id: Optional[str] = None
    version_number: int = 1
    previous_version_id: Optional[str] = None
    version_status: str = "current"
    approval_status: str = "unverified"
    issued_at: Optional[datetime] = None
    effective_from: Optional[datetime] = None
    effective_until: Optional[datetime] = None
    uploaded_by_user_id: Optional[str] = None
    visibility_scope: str = "company"
    owner_user_id: Optional[str] = None
    content_hash: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentUploadItemResult(BaseModel):
    """Result status for an individual file within a batch upload request."""

    filename: str
    status: str = Field(..., description="'uploaded', 'duplicate', or 'error'")
    document: Optional[DocumentResponse] = None
    message: str


class DocumentBatchUploadResponse(BaseModel):
    """Summary of batch upload operations."""

    total_submitted: int
    total_uploaded: int
    total_duplicates: int
    total_errors: int
    results: List[DocumentUploadItemResult]


class DocumentListResponse(BaseModel):
    """Paginated document listing."""

    total: int
    page: int
    page_size: int
    items: List[DocumentResponse]


class ExtractedTable(BaseModel):
    """Structured representation of an extracted table on a page."""

    table_id: str
    page_number: int
    row_count: int
    col_count: int
    headers: List[str]
    rows: List[List[str]]


class PageExtractionRecord(BaseModel):
    """Structured extraction record for an individual document page."""

    document_id: str
    filename: str
    page_number: int
    text: str
    headings: List[str]
    tables: List[Dict[str, Any]]
    extraction_method: str = Field(
        ...,
        description="'native_text', 'ocr', 'empty', or 'ocr_failed'",
    )
    ocr_used: bool


class DocumentExtractionDetailsResponse(BaseModel):
    """Summary response for inspecting extracted pages."""

    document_id: str
    workspace_id: str
    filename: str
    total_pages: int
    processing_status: str
    pages: List[PageExtractionRecord]
