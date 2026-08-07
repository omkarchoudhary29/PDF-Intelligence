from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from backend.schemas.document import DocumentResponse


class DocumentFamilyResponse(BaseModel):
    """Public representation of a logical document family."""

    id: str
    workspace_id: str
    title: str
    canonical_name: str
    current_document_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    version_count: int = 0
    versions: List[DocumentResponse] = []

    model_config = ConfigDict(from_attributes=True)


class DocumentFamilyListResponse(BaseModel):
    """Listing response for document families in a workspace."""

    total: int
    items: List[DocumentFamilyResponse]


class CreateFamilyRequest(BaseModel):
    """Payload to create an explicit document family."""

    title: str = Field(..., min_length=1, max_length=255)
    canonical_name: Optional[str] = Field(None, max_length=255)


class SetCurrentVersionRequest(BaseModel):
    """Payload to designate a specific version as the authoritative current version."""

    document_id: str


class UpdateApprovalStatusRequest(BaseModel):
    """Payload to change the approval status of a document version."""

    approval_status: str = Field(
        ...,
        description="'unverified', 'pending_review', 'approved', or 'rejected'",
    )


class PageAlignmentItem(BaseModel):
    """Alignment mapping between source and target pages."""

    source_page: Optional[int] = None
    target_page: Optional[int] = None
    status: str = Field(
        ...,
        description="'unchanged', 'modified', 'added', or 'removed'",
    )
    similarity_score: float = 1.0


class TextDiffBlock(BaseModel):
    """Fine-grained text-level difference on aligned or modified pages."""

    source_page: Optional[int] = None
    target_page: Optional[int] = None
    diff_type: str = Field(..., description="'added', 'removed', or 'modified'")
    category: str = Field("Unclassified", description="Change category: Administrative, Formatting-only, Procedural, Technical specification, Safety-related, Regulatory, Responsibility change, Other, Unclassified")
    source_excerpt: Optional[str] = None
    target_excerpt: Optional[str] = None
    summary: Optional[str] = None


class TableDiffBlock(BaseModel):
    """Structured table comparison result."""

    source_page: Optional[int] = None
    target_page: Optional[int] = None
    table_id: Optional[str] = None
    change_type: str = Field(
        ...,
        description="'headers_changed', 'rows_added', 'rows_removed', 'cells_modified'",
    )
    details: Dict[str, Any] = {}


class NumericalChangeItem(BaseModel):
    """Detected numerical value modification between aligned text segments."""

    source_page: Optional[int] = None
    target_page: Optional[int] = None
    context: str
    old_value: str
    new_value: str
    units: Optional[str] = None


class VersionComparisonRequest(BaseModel):
    """Payload to trigger or retrieve comparison between two versions."""

    source_document_id: str
    target_document_id: str


class VersionComparisonResponse(BaseModel):
    """Comprehensive comparison output between two document versions."""

    id: str
    family_id: str
    source_document_id: str
    target_document_id: str
    source_version_number: int
    target_version_number: int
    page_alignments: List[PageAlignmentItem] = []
    text_diffs: List[TextDiffBlock] = []
    table_diffs: List[TableDiffBlock] = []
    numerical_changes: List[NumericalChangeItem] = []
    change_summary: Optional[str] = None
    ai_explanation: Optional[str] = None
    stats: Dict[str, Any] = {}
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ContradictionFindingResponse(BaseModel):
    """Public representation of a candidate or confirmed contradiction finding."""

    id: str
    workspace_id: str
    subject: str
    predicate: str
    finding_type: str = Field(
        ...,
        description="'numerical_discrepancy', 'instruction_conflict', 'policy_conflict', 'dated_revision', 'context_dependent_difference'",
    )
    explanation: str
    first_document_id: str
    first_page: int
    first_evidence_excerpt: str
    second_document_id: str
    second_page: int
    second_evidence_excerpt: str
    confidence_score: float = 1.0
    review_status: str = Field(
        ...,
        description="'candidate', 'confirmed', 'dismissed', or 'resolved'",
    )
    reviewer_user_id: Optional[str] = None
    review_reason: Optional[str] = None
    authoritative_document_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ContradictionFindingListResponse(BaseModel):
    """Paginated or filtered list of contradiction findings."""

    total: int
    items: List[ContradictionFindingResponse]


class ReviewFindingRequest(BaseModel):
    """Payload for an authorized reviewer to update a finding's status."""

    review_status: str = Field(
        ...,
        description="'confirmed', 'dismissed', 'resolved', 'under_review', 'pending_review', or 'not_a_contradiction'",
    )
    review_reason: Optional[str] = None
    authoritative_document_id: Optional[str] = None


class DetectVersionRequest(BaseModel):
    """Payload for multi-signal version detection on a file candidate."""

    filename: str
    file_hash: Optional[str] = None
    text_preview: Optional[str] = None


class DetectVersionResponse(BaseModel):
    """Multi-signal document version detection result."""

    classification: str = Field(
        ...,
        description="'exact_duplicate', 'possible_new_version', 'new_document', or 'uncertain_match'",
    )
    confidence: float
    matched_family_id: Optional[str] = None
    matched_family_title: Optional[str] = None
    matched_document_id: Optional[str] = None
    detected_document_code: Optional[str] = None
    detected_revision: Optional[str] = None
    similarity_score: float = 0.0
    explanation: str
    signals: List[str] = []


class AttachDocumentRequest(BaseModel):
    """Payload to attach an existing document to a document family."""

    document_id: str
    version_number: Optional[int] = None
