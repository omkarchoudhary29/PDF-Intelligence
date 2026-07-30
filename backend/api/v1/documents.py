import logging
from typing import List, Optional
import fitz
from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.core.dependencies import (
    get_current_user,
    require_workspace_admin,
    require_workspace_member,
)
from backend.database.connection import get_db
from backend.database.models import Document, User, WorkspaceMembership
from backend.schemas.document import (
    DocumentBatchUploadResponse,
    DocumentExtractionDetailsResponse,
    DocumentListResponse,
    DocumentResponse,
)
from backend.schemas.indexing import (
    DocumentIndexResponse,
    DocumentIndexStatusResponse,
)
from backend.services.audit_service import audit_service
from backend.services.bm25_service import bm25_service
from backend.services.ingestion_service import ingestion_service
from backend.services.scope_resolver import scope_resolver
from backend.services.storage_service import storage_service
from backend.services.vector_index_service import vector_index_service
from backend.services.workspace_service import WorkspaceService
from config.settings import get_settings

logger = logging.getLogger("ekn.documents")
router = APIRouter(prefix="/api/v1/workspaces/{workspace_id}/documents", tags=["Documents"])


@router.post(
    "",
    response_model=DocumentBatchUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload multiple PDF documents into a workspace",
)
async def upload_documents(
    workspace_id: str,
    files: List[UploadFile] = File(..., description="PDF files to upload"),
    scope: str = Query("company", description="Ownership scope: 'company' or 'personal'"),
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentBatchUploadResponse:
    """Upload one or more PDF files with streaming write, SHA-256 deduplication, and ownership isolation."""
    settings = get_settings()

    # 1. Enforce max file count limit
    if len(files) > settings.max_files_per_upload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot upload more than {settings.max_files_per_upload} files in a single request.",
        )

    # 2. Scope and authorization validation
    norm_scope = scope.strip().lower() if scope else "company"
    if norm_scope == "company":
        if membership.role not in ("owner", "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrator permissions are required to upload company documents.",
            )
        vis_scope = "company"
        owner_id = None
    elif norm_scope == "personal":
        vis_scope = "personal"
        owner_id = current_user.id
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid upload scope. Allowed values: 'company', 'personal'.",
        )

    # 3. Workspace existence validation
    workspace = WorkspaceService.get_by_id(db, workspace_id)
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' was not found.",
        )

    # 4. Process batch upload
    response = await storage_service.process_batch_upload(
        db=db,
        workspace=workspace,
        files=files,
        visibility_scope=vis_scope,
        owner_user_id=owner_id,
        uploaded_by_user_id=current_user.id,
    )

    if vis_scope == "company":
        for item in response.results:
            if item.document:
                audit_service.log_event(
                    db=db,
                    action="document.uploaded",
                    outcome="success",
                    actor_user_id=current_user.id,
                    workspace_id=workspace_id,
                    resource_type="document",
                    resource_id=item.document.id,
                    details={
                        "filename": item.document.original_filename,
                        "visibility_scope": "company",
                        "file_size_bytes": item.document.file_size_bytes,
                    },
                )

    return response


@router.post(
    "/personal",
    response_model=DocumentBatchUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload personal documents owned by the authenticated employee",
)
async def upload_personal_documents(
    workspace_id: str,
    files: List[UploadFile] = File(..., description="PDF files to upload"),
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentBatchUploadResponse:
    """Upload personal PDF files strictly owned and accessible by the current authenticated user."""
    settings = get_settings()

    if len(files) > settings.max_files_per_upload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot upload more than {settings.max_files_per_upload} files in a single request.",
        )

    workspace = WorkspaceService.get_by_id(db, workspace_id)
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' was not found.",
        )

    response = await storage_service.process_batch_upload(
        db=db,
        workspace=workspace,
        files=files,
        visibility_scope="personal",
        owner_user_id=current_user.id,
        uploaded_by_user_id=current_user.id,
    )
    return response


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List documents within a workspace",
)
def list_documents(
    workspace_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by processing status"),
    scope: Optional[str] = Query(None, description="Scope filter: 'company', 'personal', 'all'"),
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """Paginated list of documents belonging to the specified workspace, filtered by scope."""
    workspace = WorkspaceService.get_by_id(db, workspace_id)
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' was not found.",
        )

    query = db.query(Document).filter(Document.workspace_id == workspace_id)

    # Apply strict ownership scope filtering
    norm_scope = scope.strip().lower() if scope else None
    if norm_scope == "personal":
        query = query.filter(
            Document.visibility_scope == "personal",
            Document.owner_user_id == current_user.id,
        )
    elif norm_scope == "all":
        query = query.filter(
            or_(
                Document.visibility_scope == "company",
                and_(
                    Document.visibility_scope == "personal",
                    Document.owner_user_id == current_user.id,
                ),
            )
        )
    else:
        # Default: Company documents only. Private documents are never returned in unscoped lists.
        query = query.filter(Document.visibility_scope == "company")

    if status_filter:
        query = query.filter(Document.processing_status == status_filter)

    total = query.count()
    items = (
        query.order_by(Document.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return DocumentListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[DocumentResponse.model_validate(item) for item in items],
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document details",
)
def get_document(
    workspace_id: str,
    document_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """Fetch metadata for a specific document inside a workspace with ownership validation."""
    doc = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
        .first()
    )
    if not doc or not scope_resolver.can_access_document(doc, current_user.id, membership.role):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' was not found in workspace '{workspace_id}'.",
        )

    return DocumentResponse.model_validate(doc)


@router.get(
    "/{document_id}/download",
    summary="Download original PDF document",
)
def download_document(
    workspace_id: str,
    document_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Download the original uploaded PDF file using its internal storage key."""
    doc = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
        .first()
    )
    if not doc or not scope_resolver.can_access_document(doc, current_user.id, membership.role):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' was not found in workspace '{workspace_id}'.",
        )

    try:
        physical_path = storage_service.get_document_file_path(doc)
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(err),
        )

    return FileResponse(
        path=str(physical_path),
        filename=doc.original_filename,
        media_type="application/pdf",
    )


@router.get(
    "/{document_id}/pages/{page_number}/preview",
    summary="Render a specific PDF page as a PNG image for preview",
)
def preview_document_page(
    workspace_id: str,
    document_id: str,
    page_number: int,
    dpi: int = Query(default=150, ge=72, le=300, description="DPI rendering resolution"),
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    """Render a specific page of an original PDF as PNG bytes for safe frontend preview."""
    doc = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
        .first()
    )
    if not doc or not scope_resolver.can_access_document(doc, current_user.id, membership.role):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' was not found in workspace '{workspace_id}'.",
        )

    try:
        physical_path = storage_service.get_document_file_path(doc)
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(err),
        )

    try:
        pdf_doc = fitz.open(physical_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to open PDF document: {exc}",
        )

    try:
        total_pages = len(pdf_doc)
        if page_number < 1 or page_number > total_pages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Page number {page_number} is out of bounds (1-{total_pages}).",
            )
        page = pdf_doc.load_page(page_number - 1)
        pix = page.get_pixmap(dpi=dpi)
        png_bytes = pix.tobytes("png")
        return Response(content=png_bytes, media_type="image/png")
    finally:
        pdf_doc.close()


@router.post(
    "/{document_id}/retry",
    response_model=DocumentResponse,
    summary="Retry a failed document ingestion job",
)
def retry_document(
    workspace_id: str,
    document_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """Queue a failed or completed document for re-ingestion."""
    doc = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
        .first()
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' was not found in workspace '{workspace_id}'.",
        )
    if not scope_resolver.can_manage_document(doc, current_user.id, membership.role):
        if doc.visibility_scope == "personal":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' was not found.")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator permissions required.")

    try:
        ingestion_service.retry_document_ingestion(db, workspace_id, document_id)
        db.refresh(doc)
        return DocumentResponse.model_validate(doc)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(err),
        )


@router.get(
    "/{document_id}/pages",
    response_model=DocumentExtractionDetailsResponse,
    summary="Retrieve extracted structured page records",
)
def get_extracted_pages(
    workspace_id: str,
    document_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentExtractionDetailsResponse:
    """Retrieve structured JSON Lines extraction records for a document."""
    doc = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
        .first()
    )
    if not doc or not scope_resolver.can_access_document(doc, current_user.id, membership.role):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' was not found in workspace '{workspace_id}'.",
        )

    pages = ingestion_service.get_extracted_pages(db, workspace_id, document_id)

    return DocumentExtractionDetailsResponse(
        document_id=doc.id,
        workspace_id=doc.workspace_id,
        filename=doc.original_filename,
        total_pages=doc.total_pages or len(pages),
        processing_status=doc.processing_status,
        pages=pages,
    )


@router.post(
    "/{document_id}/reindex",
    response_model=DocumentIndexResponse,
    summary="Trigger hierarchical chunking and vector re-indexing",
)
def reindex_document(
    workspace_id: str,
    document_id: str,
    force: bool = Query(default=True, description="Force re-embedding even if file hash is unchanged"),
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentIndexResponse:
    """Trigger chunking, embedding computation, and ChromaDB vector indexing."""
    doc = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
        .first()
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' was not found in workspace '{workspace_id}'.",
        )
    if not scope_resolver.can_manage_document(doc, current_user.id, membership.role):
        if doc.visibility_scope == "personal":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' was not found.")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator permissions required.")

    try:
        doc_index = ingestion_service.reindex_document(
            db=db,
            workspace_id=workspace_id,
            document_id=document_id,
            force=force,
        )
        return DocumentIndexResponse.model_validate(doc_index)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Re-indexing failed: {exc}",
        )


@router.get(
    "/{document_id}/index",
    response_model=DocumentIndexStatusResponse,
    summary="Retrieve document indexing metadata and vector status",
)
def get_document_index(
    workspace_id: str,
    document_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentIndexStatusResponse:
    """Retrieve indexing status, parent/child counts, and search readiness for a document."""
    doc = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
        .first()
    )
    if not doc or not scope_resolver.can_access_document(doc, current_user.id, membership.role):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' was not found in workspace '{workspace_id}'.",
        )

    try:
        return ingestion_service.get_document_index_status(
            db=db,
            workspace_id=workspace_id,
            document_id=document_id,
        )
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(err),
        )


@router.delete(
    "/{document_id}",
    summary="Delete a document, stored files, and ChromaDB vectors",
)
def delete_document(
    workspace_id: str,
    document_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Safely delete a document, its underlying storage files, and its ChromaDB vectors with ownership enforcement."""
    doc = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
        .first()
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' was not found in workspace '{workspace_id}'.",
        )

    # Permission check:
    if doc.visibility_scope == "company":
        if membership.role not in ("owner", "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can delete company documents.",
            )
    elif doc.visibility_scope == "personal":
        if doc.owner_user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' was not found in workspace '{workspace_id}'.",
            )

    doc_filename = doc.original_filename
    doc_scope = doc.visibility_scope

    deleted = storage_service.delete_document(db, workspace_id, document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' was not found in workspace '{workspace_id}'.",
        )

    # Clean up ChromaDB vectors for deleted document
    try:
        vector_index_service.delete_document_vectors(document_id)
    except Exception as exc:
        logger.warning("Could not delete ChromaDB vectors for document %s: %s", document_id, exc)

    # Clean up BM25 corpus records for deleted document
    try:
        bm25_service.remove_document_chunks(workspace_id=workspace_id, document_id=document_id)
    except Exception as exc:
        logger.warning("Could not delete BM25 chunks for document %s: %s", document_id, exc)

    if doc_scope == "company":
        audit_service.log_event(
            db=db,
            action="document.deleted",
            outcome="success",
            actor_user_id=current_user.id,
            workspace_id=workspace_id,
            resource_type="document",
            resource_id=document_id,
            details={
                "filename": doc_filename,
                "visibility_scope": "company",
            },
        )

    return {
        "message": "Document deleted successfully.",
        "document_id": document_id,
    }
