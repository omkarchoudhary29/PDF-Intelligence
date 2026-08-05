import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.core.dependencies import get_current_user, require_workspace_member
from backend.database.connection import get_db
from backend.database.models import User, Workspace, WorkspaceMembership
from backend.schemas.search import SearchRequest, SearchResponse
from backend.services.retrieval_service import retrieval_service

logger = logging.getLogger("ekn.search_api")

router = APIRouter(prefix="/api/v1/workspaces/{workspace_id}/search", tags=["Search"])


@router.post(
    "",
    response_model=SearchResponse,
    summary="Execute hybrid document retrieval within a workspace",
)
def search_workspace(
    workspace_id: str,
    request: SearchRequest,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SearchResponse:
    """Execute hybrid search pipeline combining dense vector, BM25, RRF, Cross-Encoder, and parent context recovery.

    - Validates workspace existence.
    - Applies strict ownership and knowledge scope constraints.
    - Applies optional document_ids and chunk_types filtering.
    - Returns full page-level provenance, parent passages, and fusion diagnostic scores.
    """
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' was not found.",
        )

    # Attach authenticated user identity for ownership scoping
    if not request.user_id:
        request.user_id = current_user.id

    try:
        return retrieval_service.search(workspace_id=workspace_id, request=request)
    except Exception as exc:
        logger.error("Hybrid search error in workspace %s: %s", workspace_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search execution failed: {str(exc)}",
        )
