"""Authentication REST API Routes: Login, Refresh, Logout, User Profile, and Password Change."""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from backend.core.dependencies import get_current_user
from backend.database.connection import get_db
from backend.database.models import User, Workspace, WorkspaceMembership
from backend.schemas.auth import (
    AdminRegisterRequest,
    ChangePasswordRequest,
    EmployeeRegisterRequest,
    InvitationValidateRequest,
    InvitationValidateResponse,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserMeResponse,
    UserWorkspaceItem,
)
from backend.services.auth_service import auth_service
from backend.services.invitation_service import invitation_service

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.get(
    "/config",
    summary="Retrieve public authentication configuration",
)
def get_auth_config() -> Dict[str, Any]:
    """Return public authentication features including demo mode status."""
    from config.settings import get_settings
    settings = get_settings()
    return {
        "demo_mode": settings.ekn_demo_mode,
        "demo_workspace_id": settings.demo_workspace_id,
    }


@router.post(
    "/admin/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new company and primary administrator account",
)
def register_admin(
    request: AdminRegisterRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Atomic company onboarding and administrator provisioning."""
    client_ip = req.client.host if req.client else None
    _, token_resp = auth_service.register_admin(
        db=db,
        full_name=request.full_name,
        company_name=request.company_name,
        email=request.email,
        password=request.password,
        ip_address=client_ip,
    )
    return token_resp


@router.post(
    "/employee/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new employee using an authorized company joining code",
)
def register_employee(
    request: EmployeeRegisterRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Atomic employee onboarding with joining code verification and single-workspace association."""
    client_ip = req.client.host if req.client else None
    _, token_resp = auth_service.register_employee(
        db=db,
        full_name=request.full_name,
        email=request.email,
        password=request.password,
        code=request.code,
        ip_address=client_ip,
    )
    return token_resp


@router.post(
    "/invitations/validate",
    response_model=InvitationValidateResponse,
    summary="Verify company joining code validity before registration",
)
def validate_invitation(
    request: InvitationValidateRequest,
    db: Session = Depends(get_db),
) -> InvitationValidateResponse:
    """Check whether a joining code is active, unexpired, and has remaining usage without logging credentials."""
    is_valid, inv, ws, msg = invitation_service.validate_code(db, request.code)
    return InvitationValidateResponse(
        is_valid=is_valid,
        company_name=ws.name if ws else None,
        workspace_id=ws.id if ws else None,
        intended_role=inv.intended_role if inv else None,
        message=msg,
    )


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new enterprise user and issue initial tokens",
)
@router.post(
    "/signup",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
def register(
    request: RegisterRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Self-registration for new users, automatically provisioning a primary workspace."""
    client_ip = req.client.host if req.client else None
    _, token_resp = auth_service.register_user(
        db=db,
        full_name=request.full_name,
        email=request.email,
        password=request.password,
        workspace_name=request.workspace_name,
        ip_address=client_ip,
    )
    return token_resp


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate user and issue JWT access and refresh tokens",
)
async def login(
    req: Request,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Validate user credentials, handle account lockout, and issue authentication tokens."""
    client_ip = req.client.host if req.client else None
    content_type = req.headers.get("content-type", "").lower()

    identifier = ""
    password = ""
    portal = ""
    if "application/json" in content_type:
        try:
            body = await req.json()
            identifier = (body.get("email") or body.get("username_or_email") or body.get("username") or "").strip()
            password = body.get("password") or ""
            portal = (body.get("portal") or "").strip()
        except Exception:
            pass
    elif "form" in content_type:
        try:
            form = await req.form()
            identifier = (form.get("username") or form.get("email") or form.get("username_or_email") or "").strip()
            password = form.get("password") or ""
            portal = (form.get("portal") or "").strip()
        except Exception:
            pass
    else:
        try:
            body = await req.json()
            identifier = (body.get("email") or body.get("username_or_email") or body.get("username") or "").strip()
            password = body.get("password") or ""
            portal = (body.get("portal") or "").strip()
        except Exception:
            try:
                form = await req.form()
                identifier = (form.get("username") or form.get("email") or form.get("username_or_email") or "").strip()
                password = form.get("password") or ""
                portal = (form.get("portal") or "").strip()
            except Exception:
                pass

    if not identifier:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Email or username is required.",
        )
    if not password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password is required.",
        )
    _, token_resp = auth_service.authenticate_user(
        db=db,
        email=identifier,
        password=password,
        ip_address=client_ip,
        portal=portal if portal else None,
    )
    return token_resp


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate refresh token and issue a fresh access token",
)
def refresh_token(
    request: RefreshRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Validate raw refresh token, rotate session, and return new token pair."""
    client_ip = req.client.host if req.client else None
    return auth_service.refresh_token(
        db=db,
        raw_refresh_token=request.refresh_token,
        ip_address=client_ip,
    )


@router.post(
    "/logout",
    summary="Revoke active refresh sessions",
)
def logout(
    req: Request,
    request: RefreshRequest = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Revoke user refresh session and invalidate active credentials."""
    client_ip = req.client.host if req.client else None
    raw_token = request.refresh_token if request else None
    auth_service.logout(
        db=db,
        user_id=current_user.id,
        raw_refresh_token=raw_token,
        ip_address=client_ip,
    )
    return {"message": "Logged out successfully."}


@router.get(
    "/me",
    response_model=UserMeResponse,
    summary="Retrieve current user profile and authorized workspace memberships",
)
def get_current_user_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserMeResponse:
    """Return sanitized profile of the authenticated user along with their active workspace roles."""
    memberships = (
        db.query(WorkspaceMembership, Workspace)
        .join(Workspace, Workspace.id == WorkspaceMembership.workspace_id)
        .filter(
            WorkspaceMembership.user_id == current_user.id,
            WorkspaceMembership.membership_status == "active",
            Workspace.is_active.is_(True),
        )
        .all()
    )

    workspace_items = [
        UserWorkspaceItem(
            workspace_id=ws.id,
            workspace_name=ws.name,
            workspace_slug=ws.slug,
            role=mem.role,
            membership_status=mem.membership_status,
        )
        for mem, ws in memberships
    ]

    return UserMeResponse(
        id=current_user.id,
        full_name=current_user.full_name,
        email=current_user.email,
        is_active=current_user.is_active,
        must_change_password=current_user.must_change_password,
        last_login_at=current_user.last_login_at,
        created_at=current_user.created_at,
        workspaces=workspace_items,
    )


@router.post(
    "/change-password",
    summary="Change user password and revoke previous sessions",
)
def change_password(
    request: ChangePasswordRequest,
    req: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Update password, enforce complexity requirements, and reset first-login flag."""
    client_ip = req.client.host if req.client else None
    auth_service.change_password(
        db=db,
        user=current_user,
        request=request,
        ip_address=client_ip,
    )
    return {"message": "Password changed successfully. Active sessions updated."}
