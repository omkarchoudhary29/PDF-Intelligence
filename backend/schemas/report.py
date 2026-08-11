"""Pydantic schemas for Knowledge Dossier Report generation, requests, and responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ReportCreateRequest(BaseModel):
    report_name: Optional[str] = Field(None, max_length=255, description="Custom title for the dossier")
    report_type: str = Field(default="knowledge_dossier", description="Report format type")
    include_graph: bool = Field(default=True, description="Whether to embed knowledge graph visual")


class ReportResponse(BaseModel):
    id: str
    workspace_id: str
    report_name: str
    report_type: str
    status: str
    document_count: int = 0
    total_pages_covered: int = 0
    file_size_bytes: int = 0
    generation_started_at: Optional[datetime] = None
    generation_completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ReportListResponse(BaseModel):
    total: int
    items: List[ReportResponse]


class DocumentSummaryData(BaseModel):
    document_id: str
    original_filename: str
    total_pages: int
    purpose: str
    main_subjects: List[str] = Field(default_factory=list)
    important_findings: List[str] = Field(default_factory=list)
    key_sections: List[str] = Field(default_factory=list)
    page_references: List[int] = Field(default_factory=list)
    summary_text: str
    related_documents: List[str] = Field(default_factory=list)
    summary_status: str = "completed"  # 'completed', 'partial', 'unavailable'
    version_number: int = 1
    version_status: str = "current"
    family_title: Optional[str] = None


class ExecutiveSummaryData(BaseModel):
    overview_text: str
    key_themes: List[str] = Field(default_factory=list)
    documentation_gaps: List[str] = Field(default_factory=list)
    document_count: int = 0
    total_pages: int = 0
    contradiction_notices: List[Dict[str, Any]] = Field(default_factory=list)
    version_revisions_count: int = 0
