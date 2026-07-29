"""Workspace Member Administration REST API Routes."""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from backend.core.dependencies import (
    get_current_user,
    require_workspace_admin,
    require_workspace_member,
)
from backend.database.connection import get_db
from backend.database.models import User, WorkspaceMembership
from backend.schemas.auth import (
    MemberCreateRequest,
    MemberListResponse,
    MemberResponse,
    MemberUpdateRequest,
)
from backend.services.auth_service import auth_service

router = APIRouter(prefix="/api/v1/workspaces/{workspace_id}/members", tags=["Workspace Members"])


@router.get(
    "",
    response_model=MemberListResponse,
    summary="List members belonging to the workspace",
)
def list_workspace_members(
    workspace_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> MemberListResponse:
    """Retrieve list of authorized members belonging to the current workspace."""
    query = (
        db.query(WorkspaceMembership, User)
        .join(User, User.id == WorkspaceMembership.user_id)
        .filter(WorkspaceMembership.workspace_id == workspace_id)
        .order_by(WorkspaceMembership.created_at.asc())
    )
    total = query.count()
    rows = query.offset((page - 1) * page_size).limit(page_size).all()

    items = [
        MemberResponse(
            id=mem.id,
            user_id=usr.id,
            workspace_id=mem.workspace_id,
            full_name=usr.full_name,
            email=usr.email,
            role=mem.role,
            membership_status=mem.membership_status,
            created_at=mem.created_at,
            temporary_password=None,
        )
        for mem, usr in rows
    ]
    return MemberListResponse(total=total, items=items)


@router.post(
    "",
    response_model=MemberResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add an existing user or provision a new user into the workspace",
)
def add_workspace_member(
    workspace_id: str,
    request: MemberCreateRequest,
    req: Request,
    current_membership: WorkspaceMembership = Depends(require_workspace_admin),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemberResponse:
    """Add a member to the workspace with role validation and temporary credential generation."""
    client_ip = req.client.host if req.client else None
    mem, temp_pass = auth_service.add_workspace_member(
        db=db,
        actor_user=current_user,
        workspace_id=workspace_id,
        request=request,
        ip_address=client_ip,
    )
    user = db.query(User).filter(User.id == mem.user_id).first()
    return MemberResponse(
        id=mem.id,
        user_id=user.id,
        workspace_id=mem.workspace_id,
        full_name=user.full_name,
        email=user.email,
        role=mem.role,
        membership_status=mem.membership_status,
        created_at=mem.created_at,
        temporary_password=temp_pass,
    )


@router.patch(
    "/{user_id}",
    response_model=MemberResponse,
    summary="Update a workspace member's role or membership status",
)
def update_workspace_member(
    workspace_id: str,
    user_id: str,
    request: MemberUpdateRequest,
    req: Request,
    current_membership: WorkspaceMembership = Depends(require_workspace_admin),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemberResponse:
    """Modify role or membership status with strict owner protection."""
    client_ip = req.client.host if req.client else None
    mem = auth_service.update_workspace_member(
        db=db,
        actor_user=current_user,
        workspace_id=workspace_id,
        target_user_id=user_id,
        request=request,
        ip_address=client_ip,
    )
    user = db.query(User).filter(User.id == mem.user_id).first()
    return MemberResponse(
        id=mem.id,
        user_id=user.id,
        workspace_id=mem.workspace_id,
        full_name=user.full_name,
        email=user.email,
        role=mem.role,
        membership_status=mem.membership_status,
        created_at=mem.created_at,
        temporary_password=None,
    )


@router.delete(
    "/{user_id}",
    summary="Remove a member from the workspace",
)
def remove_workspace_member(
    workspace_id: str,
    user_id: str,
    req: Request,
    current_membership: WorkspaceMembership = Depends(require_workspace_admin),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Remove a member from the workspace with final owner protection."""
    client_ip = req.client.host if req.client else None
    auth_service.remove_workspace_member(
        db=db,
        actor_user=current_user,
        workspace_id=workspace_id,
        target_user_id=user_id,
        ip_address=client_ip,
    )
    return {"message": "Member removed successfully from workspace."}
