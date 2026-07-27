"""Centralized FastAPI Authorization Dependencies for EKN.

Enforces:
- JWT Bearer token authentication
- Active account verification and lockout checks
- Strict workspace membership validation (tenant boundary isolation)
- Role-based access control (Owner, Admin, Employee)
- Deny-by-default security policies across all endpoints
"""

from datetime import datetime, timezone
import logging
from typing import Callable, List, Optional
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.core.security import (
    InvalidTokenError,
    SecurityError,
    TokenExpiredError,
    decode_access_token,
)
from backend.database.connection import get_db
from backend.database.models import User, Workspace, WorkspaceMembership
from backend.services.audit_service import audit_service

logger = logging.getLogger("ekn.security.dependencies")

# Reusable HTTP Bearer scheme
bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Validate JWT access token and return active authenticated user."""
    unauthorized_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication credentials were not provided or are invalid.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not credentials or not credentials.credentials:
        raise unauthorized_exc

    token = credentials.credentials
    try:
        payload = decode_access_token(token)
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token has expired. Please refresh your session.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (InvalidTokenError, SecurityError) as err:
        logger.warning("Invalid token received: %s", err)
        raise unauthorized_exc

    user_id = payload.get("sub")
    if not user_id:
        raise unauthorized_exc

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise unauthorized_exc

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated.",
        )

    # Account lockout check
    now = datetime.now(timezone.utc)
    if user.locked_until:
        locked_utc = user.locked_until
        if locked_utc.tzinfo is None:
            locked_utc = locked_utc.replace(tzinfo=timezone.utc)
        if locked_utc > now:
            remaining_mins = int((locked_utc - now).total_seconds() / 60) + 1
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Account is temporarily locked. Try again in {remaining_mins} minute(s).",
            )

    return user


def get_active_user(user: User = Depends(get_current_user)) -> User:
    """Ensure user is authenticated and active."""
    return user


def get_workspace_membership(
    workspace_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkspaceMembership:
    """Verify that the current user has active membership in the target workspace.

    Returns 404 if the workspace does not exist or if the user is not a member,
    preventing cross-tenant information leakage about whether foreign workspaces exist.
    """
    membership = (
        db.query(WorkspaceMembership)
        .filter(
            WorkspaceMembership.workspace_id == workspace_id,
            WorkspaceMembership.user_id == user.id,
            WorkspaceMembership.membership_status == "active",
        )
        .first()
    )

    if not membership:
        # Check if workspace actually exists to log cross-tenant probe
        ws_exists = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if ws_exists:
            audit_service.log_event(
                db=db,
                action="cross_tenant_access_denied",
                outcome="denied",
                actor_user_id=user.id,
                workspace_id=workspace_id,
                details={"target_workspace": workspace_id, "user_email": user.email},
            )
        # Deny-by-default: return 404 to avoid confirming existence of foreign tenant resources
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' was not found.",
        )

    return membership


def require_role(allowed_roles: List[str]) -> Callable[..., WorkspaceMembership]:
    """Dependency factory checking that the user holds an approved role in the workspace."""

    def role_checker(
        membership: WorkspaceMembership = Depends(get_workspace_membership),
        db: Session = Depends(get_db),
    ) -> WorkspaceMembership:
        if membership.role not in allowed_roles:
            audit_service.log_event(
                db=db,
                action="insufficient_role_denied",
                outcome="denied",
                actor_user_id=membership.user_id,
                workspace_id=membership.workspace_id,
                details={
                    "user_role": membership.role,
                    "required_roles": allowed_roles,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {', '.join(allowed_roles)} (current role: {membership.role}).",
            )
        return membership

    return role_checker


# Pre-configured role dependencies
require_workspace_owner = require_role(["owner"])
require_workspace_admin = require_role(["owner", "admin"])
require_workspace_member = require_role(["owner", "admin", "employee"])
