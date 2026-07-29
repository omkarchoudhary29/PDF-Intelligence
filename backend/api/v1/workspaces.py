"""Workspace REST API Endpoints with Strict Authentication and Multi-Tenant RBAC."""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.core.dependencies import (
    get_current_user,
    require_workspace_admin,
    require_workspace_member,
    require_workspace_owner,
)
from backend.database.connection import get_db
from backend.database.models import SecurityAuditLog, User, WorkspaceMembership
from backend.schemas.auth import (
    InvitationCreateRequest,
    InvitationCreateResponse,
    InvitationItemResponse,
    InvitationListResponse,
)
from backend.schemas.workspace import (
    ActivityItemResponse,
    ActivityListResponse,
    WorkspaceCreate,
    WorkspaceResponse,
    WorkspaceStatsResponse,
    WorkspaceUpdate,
)
from backend.services.invitation_service import invitation_service
from backend.services.workspace_service import WorkspaceService

router = APIRouter(prefix="/api/v1/workspaces", tags=["Workspaces"])


@router.post(
    "",
    response_model=WorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new workspace",
)
def create_workspace(
    data: WorkspaceCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkspaceResponse:
    """Create an isolated workspace and automatically assign creating user as Owner."""
    try:
        workspace = WorkspaceService.create_workspace(db, data)
        # Automatically assign creating user as the initial Workspace Owner
        membership = WorkspaceMembership(
            user_id=current_user.id,
            workspace_id=workspace.id,
            role="owner",
            membership_status="active",
        )
        db.add(membership)
        db.commit()

        return WorkspaceResponse(
            id=workspace.id,
            name=workspace.name,
            slug=workspace.slug,
            description=workspace.description,
            is_active=workspace.is_active,
            document_count=0,
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
        )
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )


@router.get(
    "",
    response_model=List[WorkspaceResponse],
    summary="List workspaces accessible to the authenticated user",
)
def list_workspaces(
    active_only: bool = True,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[WorkspaceResponse]:
    """List only workspaces where the authenticated user holds an active membership."""
    results = WorkspaceService.list_workspaces_for_user(
        db,
        user_id=current_user.id,
        active_only=active_only,
    )
    return [
        WorkspaceResponse(
            id=ws.id,
            name=ws.name,
            slug=ws.slug,
            description=ws.description,
            is_active=ws.is_active,
            document_count=doc_count,
            created_at=ws.created_at,
            updated_at=ws.updated_at,
        )
        for ws, doc_count in results
    ]


@router.get(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    summary="Get workspace details",
)
def get_workspace(
    workspace_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> WorkspaceResponse:
    """Fetch workspace details and document count by ID with tenant isolation."""
    row = WorkspaceService.get_with_doc_count(db, workspace_id)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' was not found.",
        )
    ws, doc_count = row
    return WorkspaceResponse(
        id=ws.id,
        name=ws.name,
        slug=ws.slug,
        description=ws.description,
        is_active=ws.is_active,
        document_count=doc_count,
        created_at=ws.created_at,
        updated_at=ws.updated_at,
    )


@router.patch(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    summary="Update workspace details (Administrator/Owner only)",
)
def update_workspace(
    workspace_id: str,
    data: WorkspaceUpdate,
    membership: WorkspaceMembership = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> WorkspaceResponse:
    """Update company/workspace profile settings such as name and description."""
    updated = WorkspaceService.update_workspace(db, workspace_id, data)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' was not found.",
        )
    row = WorkspaceService.get_with_doc_count(db, workspace_id)
    doc_count = row[1] if row else 0
    return WorkspaceResponse(
        id=updated.id,
        name=updated.name,
        slug=updated.slug,
        description=updated.description,
        is_active=updated.is_active,
        document_count=doc_count,
        created_at=updated.created_at,
        updated_at=updated.updated_at,
    )


@router.delete(
    "/{workspace_id}",
    summary="Delete a workspace (Owner only)",
)
def delete_workspace(
    workspace_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_owner),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Delete workspace and all associated resources. Enforces Owner-only permission."""
    deleted = WorkspaceService.delete_workspace(db, workspace_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' was not found.",
        )
    return {"message": "Workspace deleted successfully.", "workspace_id": workspace_id}


@router.get(
    "/{workspace_id}/stats",
    response_model=WorkspaceStatsResponse,
    summary="Get repository statistics for a workspace",
)
def get_workspace_stats(
    workspace_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> WorkspaceStatsResponse:
    """Fetch repository counts with tenant isolation."""
    stats = WorkspaceService.get_workspace_stats(db, workspace_id)
    return WorkspaceStatsResponse.model_validate(stats)


# --- Workspace Invitations & Joining Codes (Phase 4) ---


@router.post(
    "/{workspace_id}/invitations",
    response_model=InvitationCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a new employee joining code (Administrator only)",
)
def create_workspace_invitation(
    workspace_id: str,
    request: InvitationCreateRequest,
    current_membership: WorkspaceMembership = Depends(require_workspace_admin),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> InvitationCreateResponse:
    """Generate a one-time raw joining code for employee onboarding."""
    invitation, raw_code = invitation_service.create_invitation(
        db=db,
        workspace_id=workspace_id,
        creator_user_id=current_user.id,
        max_usage_count=request.max_usage_count or 10,
        expires_in_days=request.expires_in_days or 7,
        intended_role=request.intended_role or "employee",
    )
    return InvitationCreateResponse(
        id=invitation.id,
        raw_code=raw_code,
        workspace_id=invitation.workspace_id,
        intended_role=invitation.intended_role,
        expires_at=invitation.expires_at,
        max_usage_count=invitation.max_usage_count,
        created_at=invitation.created_at,
    )


@router.get(
    "/{workspace_id}/invitations",
    response_model=InvitationListResponse,
    summary="List all joining codes for the workspace (Administrator only)",
)
def list_workspace_invitations(
    workspace_id: str,
    current_membership: WorkspaceMembership = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> InvitationListResponse:
    """List joining codes and usage statistics. Raw codes and hashes are omitted."""
    items_data = invitation_service.list_invitations(db, workspace_id)
    items = [InvitationItemResponse.model_validate(item) for item in items_data]
    return InvitationListResponse(total=len(items), items=items)


@router.delete(
    "/{workspace_id}/invitations/{invitation_id}",
    summary="Revoke a joining code (Administrator only)",
)
def revoke_workspace_invitation(
    workspace_id: str,
    invitation_id: str,
    current_membership: WorkspaceMembership = Depends(require_workspace_admin),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Revoke an active joining code so no further registrations can use it."""
    invitation_service.revoke_invitation(
        db=db,
        workspace_id=workspace_id,
        invitation_id=invitation_id,
        actor_user_id=current_user.id,
    )
    return {
        "message": "Invitation revoked successfully.",
        "workspace_id": workspace_id,
        "invitation_id": invitation_id,
    }


# --- Company Activity (Phase 6) ---


@router.get(
    "/{workspace_id}/activity",
    response_model=ActivityListResponse,
    summary="List company activity audit logs (Administrator only)",
)
def get_workspace_activity(
    workspace_id: str,
    limit: int = Query(50, ge=1, le=200),
    current_membership: WorkspaceMembership = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> ActivityListResponse:
    """Retrieve genuine company-level activity audit records with tenant isolation."""
    query = (
        db.query(SecurityAuditLog, User)
        .outerjoin(User, SecurityAuditLog.actor_user_id == User.id)
        .filter(SecurityAuditLog.workspace_id == workspace_id)
        .order_by(SecurityAuditLog.created_at.desc())
        .limit(limit)
    )
    rows = query.all()

    items = []
    for log, usr in rows:
        actor_name = usr.full_name if usr else None
        actor_email = usr.email if usr else None
        items.append(
            ActivityItemResponse(
                id=log.id,
                action=log.action,
                actor_user_id=log.actor_user_id,
                actor_name=actor_name,
                actor_email=actor_email,
                resource_type=log.resource_type,
                resource_id=log.resource_id,
                outcome=log.outcome,
                details=log.details_json,
                created_at=log.created_at,
            )
        )
    return ActivityListResponse(total=len(items), items=items)
