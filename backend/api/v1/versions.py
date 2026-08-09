import logging
from typing import List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from backend.core.dependencies import (
    require_workspace_admin,
    require_workspace_member,
)
from backend.database.connection import get_db
from backend.database.models import Document, WorkspaceMembership
from backend.schemas.document import DocumentResponse, DocumentUploadItemResult
from backend.schemas.version import (
    AttachDocumentRequest,
    ContradictionFindingListResponse,
    ContradictionFindingResponse,
    CreateFamilyRequest,
    DetectVersionRequest,
    DetectVersionResponse,
    DocumentFamilyListResponse,
    DocumentFamilyResponse,
    ReviewFindingRequest,
    SetCurrentVersionRequest,
    UpdateApprovalStatusRequest,
    VersionComparisonRequest,
    VersionComparisonResponse,
)
from backend.services.contradiction_service import contradiction_service
from backend.services.ingestion_service import ingestion_service
from backend.services.version_comparison_service import version_comparison_service
from backend.services.version_service import version_service
from backend.services.workspace_service import WorkspaceService

logger = logging.getLogger("ekn.versions_api")
router = APIRouter(prefix="/api/v1/workspaces/{workspace_id}", tags=["Document Version Intelligence"])


# -----------------------------------------------------------------------------
# Document Family & Version Management
# -----------------------------------------------------------------------------


@router.get(
    "/document-families",
    response_model=DocumentFamilyListResponse,
    summary="List document families in a workspace",
)
def list_document_families(
    workspace_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> DocumentFamilyListResponse:
    """List all logical document families with their active and historical versions."""
    workspace = WorkspaceService.get_by_id(db, workspace_id)
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' not found.",
        )

    families = version_service.list_families(db, workspace_id)
    items: List[DocumentFamilyResponse] = []
    for f in families:
        versions = [DocumentResponse.model_validate(v) for v in f.versions]
        items.append(
            DocumentFamilyResponse(
                id=f.id,
                workspace_id=f.workspace_id,
                title=f.title,
                canonical_name=f.canonical_name,
                current_document_id=f.current_document_id,
                created_at=f.created_at,
                updated_at=f.updated_at,
                version_count=len(versions),
                versions=versions,
            )
        )

    return DocumentFamilyListResponse(total=len(items), items=items)


@router.post(
    "/document-families",
    response_model=DocumentFamilyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an explicit document family",
)
def create_document_family(
    workspace_id: str,
    request: CreateFamilyRequest,
    membership: WorkspaceMembership = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> DocumentFamilyResponse:
    """Explicitly create a new logical document family for version management."""
    workspace = WorkspaceService.get_by_id(db, workspace_id)
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' not found.",
        )

    family = version_service.create_family(
        db=db,
        workspace_id=workspace_id,
        title=request.title,
        canonical_name=request.canonical_name,
        actor_user_id=membership.user_id,
    )
    versions = [DocumentResponse.model_validate(v) for v in family.versions]
    return DocumentFamilyResponse(
        id=family.id,
        workspace_id=family.workspace_id,
        title=family.title,
        canonical_name=family.canonical_name,
        current_document_id=family.current_document_id,
        created_at=family.created_at,
        updated_at=family.updated_at,
        version_count=len(versions),
        versions=versions,
    )


@router.get(
    "/document-families/{family_id}",
    response_model=DocumentFamilyResponse,
    summary="Get document family details and version history",
)
def get_document_family(
    workspace_id: str,
    family_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> DocumentFamilyResponse:
    """Retrieve details and version history for a specific document family."""
    family = version_service.get_family_by_id(db, workspace_id, family_id)
    if not family:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document family '{family_id}' not found in workspace '{workspace_id}'.",
        )

    versions = [DocumentResponse.model_validate(v) for v in family.versions]
    return DocumentFamilyResponse(
        id=family.id,
        workspace_id=family.workspace_id,
        title=family.title,
        canonical_name=family.canonical_name,
        current_document_id=family.current_document_id,
        created_at=family.created_at,
        updated_at=family.updated_at,
        version_count=len(versions),
        versions=versions,
    )


@router.post(
    "/document-families/{family_id}/versions",
    response_model=DocumentUploadItemResult,
    summary="Upload a new version to an existing document family",
)
async def upload_document_version(
    workspace_id: str,
    family_id: str,
    file: UploadFile = File(..., description="PDF file for the new version"),
    membership: WorkspaceMembership = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> DocumentUploadItemResult:
    """Upload a new PDF version to an existing document family with deduplication and sequential numbering."""
    workspace = WorkspaceService.get_by_id(db, workspace_id)
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' not found.",
        )

    family = version_service.get_family_by_id(db, workspace_id, family_id)
    if not family:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document family '{family_id}' not found in workspace '{workspace_id}'.",
        )

    result = await version_service.upload_new_version(
        db=db,
        workspace=workspace,
        family=family,
        upload_file=file,
        uploaded_by_user_id=membership.user_id,
    )
    return result


@router.get(
    "/documents/{document_id}/versions",
    response_model=DocumentFamilyResponse,
    summary="Get version family details for a document",
)
def get_document_version_lineage(
    workspace_id: str,
    document_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> DocumentFamilyResponse:
    """Retrieve full version history and family context for a specific document."""
    doc = (
        db.query(Document)
        .filter(Document.id == document_id, Document.workspace_id == workspace_id)
        .first()
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in workspace '{workspace_id}'.",
        )

    if not doc.family_id:
        version_service.assign_families_to_existing_documents(db, workspace_id)
        db.refresh(doc)

    family = version_service.get_family_by_id(db, workspace_id, doc.family_id)
    if not family:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document does not belong to a valid document family.",
        )

    versions = [DocumentResponse.model_validate(v) for v in family.versions]
    return DocumentFamilyResponse(
        id=family.id,
        workspace_id=family.workspace_id,
        title=family.title,
        canonical_name=family.canonical_name,
        current_document_id=family.current_document_id,
        created_at=family.created_at,
        updated_at=family.updated_at,
        version_count=len(versions),
        versions=versions,
    )


@router.patch(
    "/document-families/{family_id}/current-version",
    response_model=DocumentFamilyResponse,
    summary="Designate an explicit current version for a document family",
)
def set_current_version(
    workspace_id: str,
    family_id: str,
    request: SetCurrentVersionRequest,
    membership: WorkspaceMembership = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> DocumentFamilyResponse:
    """Switch the official current version of a document family."""
    try:
        family = version_service.set_current_version(
            db=db,
            workspace_id=workspace_id,
            family_id=family_id,
            document_id=request.document_id,
            actor_user_id=membership.user_id,
        )
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))

    versions = [DocumentResponse.model_validate(v) for v in family.versions]
    return DocumentFamilyResponse(
        id=family.id,
        workspace_id=family.workspace_id,
        title=family.title,
        canonical_name=family.canonical_name,
        current_document_id=family.current_document_id,
        created_at=family.created_at,
        updated_at=family.updated_at,
        version_count=len(versions),
        versions=versions,
    )


@router.patch(
    "/documents/{document_id}/approval-status",
    response_model=DocumentResponse,
    summary="Update approval status of a document version",
)
def update_approval_status(
    workspace_id: str,
    document_id: str,
    request: UpdateApprovalStatusRequest,
    membership: WorkspaceMembership = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """Set approval status for a document version ('unverified', 'pending_review', 'approved', 'rejected')."""
    try:
        doc = version_service.update_approval_status(
            db=db,
            workspace_id=workspace_id,
            document_id=document_id,
            approval_status=request.approval_status,
            actor_user_id=membership.user_id,
        )
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))

    return DocumentResponse.model_validate(doc)


@router.post(
    "/detect-version-candidate",
    response_model=DetectVersionResponse,
    summary="Multi-signal version detection on filename and metadata",
)
def detect_version_candidate(
    workspace_id: str,
    request: DetectVersionRequest,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> DetectVersionResponse:
    """Analyze filename, hash, codes, and text preview to classify version status."""
    res = version_service.detect_version_signals(
        db=db,
        workspace_id=workspace_id,
        filename=request.filename,
        file_hash=request.file_hash,
        text_content=request.text_preview,
    )
    return DetectVersionResponse(**res)


@router.post(
    "/documents/{document_id}/detect-version",
    response_model=DetectVersionResponse,
    summary="Multi-signal version detection on an existing uploaded document",
)
def detect_existing_document_version(
    workspace_id: str,
    document_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> DetectVersionResponse:
    """Analyze an existing document against all families to determine version relationship."""
    doc = (
        db.query(Document)
        .filter(Document.id == document_id, Document.workspace_id == workspace_id)
        .first()
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in workspace '{workspace_id}'.",
        )

    # Fetch text preview from extracted pages
    pages = ingestion_service.get_extracted_pages(db, workspace_id, document_id)
    text_content = ""
    if pages:
        text_content = "\n".join(
            (p.get("text") or p.get("raw_text") or "") if isinstance(p, dict)
            else (getattr(p, "text", "") or getattr(p, "raw_text", "") or "")
            for p in pages[:3]
        )

    res = version_service.detect_version_signals(
        db=db,
        workspace_id=workspace_id,
        filename=doc.original_filename,
        file_hash=doc.sha256_hash,
        text_content=text_content,
        doc_id=doc.id,
    )
    return DetectVersionResponse(**res)


@router.post(
    "/document-families/{family_id}/attach-document",
    response_model=DocumentResponse,
    summary="Attach an existing document to a document family",
)
def attach_document_to_family(
    workspace_id: str,
    family_id: str,
    request: AttachDocumentRequest,
    membership: WorkspaceMembership = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """Manually link an existing document into a family as a specific sequential version."""
    try:
        doc = version_service.attach_document_to_family(
            db=db,
            workspace_id=workspace_id,
            family_id=family_id,
            document_id=request.document_id,
            version_number=request.version_number,
            actor_user_id=membership.user_id,
        )
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))

    return DocumentResponse.model_validate(doc)


@router.post(
    "/document-families/{family_id}/detach-document/{document_id}",
    response_model=DocumentFamilyResponse,
    summary="Detach a document from a family into an independent family",
)
def detach_document_from_family(
    workspace_id: str,
    family_id: str,
    document_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> DocumentFamilyResponse:
    """Detach a document from its current family and create an independent family for it."""
    try:
        new_family = version_service.detach_document_from_family(
            db=db,
            workspace_id=workspace_id,
            document_id=document_id,
            actor_user_id=membership.user_id,
        )
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))

    versions = [DocumentResponse.model_validate(v) for v in new_family.versions]
    return DocumentFamilyResponse(
        id=new_family.id,
        workspace_id=new_family.workspace_id,
        title=new_family.title,
        canonical_name=new_family.canonical_name,
        current_document_id=new_family.current_document_id,
        created_at=new_family.created_at,
        updated_at=new_family.updated_at,
        version_count=len(versions),
        versions=versions,
    )


# -----------------------------------------------------------------------------
# Version Comparison
# -----------------------------------------------------------------------------


@router.post(
    "/document-families/{family_id}/compare",
    response_model=VersionComparisonResponse,
    summary="Compare two document versions within a family",
)
def compare_versions(
    workspace_id: str,
    family_id: str,
    request: VersionComparisonRequest,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> VersionComparisonResponse:
    """Align pages and detect text, numerical, and tabular differences between two versions."""
    try:
        comp = version_comparison_service.compare_versions(
            db=db,
            workspace_id=workspace_id,
            family_id=family_id,
            source_document_id=request.source_document_id,
            target_document_id=request.target_document_id,
        )
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))

    data = comp.comparison_data or {}
    return VersionComparisonResponse(
        id=comp.id,
        family_id=comp.family_id,
        source_document_id=comp.source_document_id,
        target_document_id=comp.target_document_id,
        source_version_number=comp.source_version_number,
        target_version_number=comp.target_version_number,
        page_alignments=data.get("page_alignments", []),
        text_diffs=data.get("text_diffs", []),
        table_diffs=data.get("table_diffs", []),
        numerical_changes=data.get("numerical_changes", []),
        change_summary=comp.change_summary,
        ai_explanation=comp.ai_explanation,
        stats=data.get("stats", {}),
        created_at=comp.created_at,
    )


@router.get(
    "/document-families/{family_id}/comparisons/{comparison_id}",
    response_model=VersionComparisonResponse,
    summary="Retrieve stored version comparison result",
)
def get_comparison(
    workspace_id: str,
    family_id: str,
    comparison_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> VersionComparisonResponse:
    """Retrieve existing version comparison result."""
    comp = version_comparison_service.get_comparison_by_id(
        db=db,
        workspace_id=workspace_id,
        family_id=family_id,
        comparison_id=comparison_id,
    )
    if not comp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version comparison '{comparison_id}' not found.",
        )

    data = comp.comparison_data or {}
    return VersionComparisonResponse(
        id=comp.id,
        family_id=comp.family_id,
        source_document_id=comp.source_document_id,
        target_document_id=comp.target_document_id,
        source_version_number=comp.source_version_number,
        target_version_number=comp.target_version_number,
        page_alignments=data.get("page_alignments", []),
        text_diffs=data.get("text_diffs", []),
        table_diffs=data.get("table_diffs", []),
        numerical_changes=data.get("numerical_changes", []),
        change_summary=comp.change_summary,
        ai_explanation=comp.ai_explanation,
        stats=data.get("stats", {}),
        created_at=comp.created_at,
    )


# -----------------------------------------------------------------------------
# Contradiction Findings
# -----------------------------------------------------------------------------


@router.get(
    "/findings",
    response_model=ContradictionFindingListResponse,
    summary="List contradiction findings with status and type filters",
)
def list_contradiction_findings(
    workspace_id: str,
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by review status"),
    finding_type: Optional[str] = Query(None, description="Filter by conflict category"),
    scan: bool = Query(False, description="Trigger fresh scan of promising document pairs before returning"),
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> ContradictionFindingListResponse:
    """List contradiction findings across documents in the workspace."""
    workspace = WorkspaceService.get_by_id(db, workspace_id)
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' not found.",
        )

    if scan:
        findings = contradiction_service.scan_workspace_contradictions(db, workspace_id)
        if status_filter:
            findings = [f for f in findings if f.review_status == status_filter]
        if finding_type:
            findings = [f for f in findings if f.finding_type == finding_type]
    else:
        findings = contradiction_service.list_findings(
            db=db,
            workspace_id=workspace_id,
            review_status=status_filter,
            finding_type=finding_type,
        )

    items = [ContradictionFindingResponse.model_validate(f) for f in findings]
    return ContradictionFindingListResponse(total=len(items), items=items)


@router.get(
    "/findings/{finding_id}",
    response_model=ContradictionFindingResponse,
    summary="Get contradiction finding details and supporting evidence",
)
def get_contradiction_finding(
    workspace_id: str,
    finding_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> ContradictionFindingResponse:
    """Retrieve detailed contradiction finding with both conflicting excerpts."""
    finding = contradiction_service.get_finding_by_id(db, workspace_id, finding_id)
    if not finding:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contradiction finding '{finding_id}' not found in workspace '{workspace_id}'.",
        )
    return ContradictionFindingResponse.model_validate(finding)


@router.patch(
    "/findings/{finding_id}/review",
    response_model=ContradictionFindingResponse,
    summary="Review contradiction finding (confirm, dismiss, resolve)",
)
def review_contradiction_finding(
    workspace_id: str,
    finding_id: str,
    request: ReviewFindingRequest,
    membership: WorkspaceMembership = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> ContradictionFindingResponse:
    """Authorized review action on a contradiction finding with audit logging."""
    try:
        finding = contradiction_service.review_finding(
            db=db,
            workspace_id=workspace_id,
            finding_id=finding_id,
            review_status=request.review_status,
            review_reason=request.review_reason,
            authoritative_document_id=request.authoritative_document_id,
            reviewer_user_id=membership.user_id,
        )
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))

    return ContradictionFindingResponse.model_validate(finding)
