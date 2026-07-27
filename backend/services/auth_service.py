"""Enterprise Authentication, User Provisioning, and Membership Service.

Handles:
- Argon2id credential verification and database-backed account lockout
- Access and refresh token generation, rotation, and revocation
- Password changes and first-login password enforcement
- Initial owner provisioning CLI support
- Workspace membership administration (Owner, Admin, Employee) with final owner protection
"""

from datetime import datetime, timedelta, timezone
import logging
import re
import secrets
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.core.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_token,
    validate_password_strength,
    verify_password,
)
from backend.database.models import (
    RefreshSession,
    User,
    Workspace,
    WorkspaceInvitation,
    WorkspaceMembership,
)
from backend.schemas.auth import (
    ChangePasswordRequest,
    MemberCreateRequest,
    MemberUpdateRequest,
    TokenResponse,
    UserResponse,
)
from backend.services.audit_service import audit_service
from config.settings import get_settings

logger = logging.getLogger("ekn.security.auth_service")


class AuthService:
    """Manages secure authentication, refresh sessions, and workspace memberships."""

    def __init__(self) -> None:
        self._settings = None

    @property
    def settings(self):
        return self._settings or get_settings()

    @settings.setter
    def settings(self, val):
        self._settings = val

    def authenticate_user(
        self,
        db: Session,
        email: str,
        password: str,
        ip_address: Optional[str] = None,
        portal: Optional[str] = None,
    ) -> Tuple[User, TokenResponse]:
        """Authenticate user credentials with brute-force lockout protection."""
        normalized_email = email.lower().strip()
        now = datetime.now(timezone.utc)

        user = db.query(User).filter(User.email == normalized_email).first()

        # Brute-force lockout check if user exists
        if user and user.locked_until:
            # Ensure timezone-aware comparison
            locked_until_utc = user.locked_until
            if locked_until_utc.tzinfo is None:
                locked_until_utc = locked_until_utc.replace(tzinfo=timezone.utc)

            if locked_until_utc > now:
                remaining_mins = int((locked_until_utc - now).total_seconds() / 60) + 1
                audit_service.log_event(
                    db=db,
                    action="login_attempt_locked",
                    outcome="denied",
                    actor_user_id=user.id,
                    ip_address=ip_address,
                    details={"email": normalized_email, "remaining_minutes": remaining_mins},
                )
                raise HTTPException(
                    status_code=status.HTTP_423_LOCKED,
                    detail=f"Account is temporarily locked due to consecutive failed login attempts. Try again in {remaining_mins} minute(s).",
                )
            else:
                # Lockout period expired; reset lock
                user.locked_until = None
                user.failed_login_attempts = 0
                db.commit()

        # Generic error message to prevent account enumeration
        invalid_credentials_exc = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

        if not user:
            audit_service.log_event(
                db=db,
                action="login_failed_unknown_user",
                outcome="failure",
                ip_address=ip_address,
                details={"email": normalized_email},
            )
            raise invalid_credentials_exc

        if not user.is_active:
            audit_service.log_event(
                db=db,
                action="login_failed_inactive_user",
                outcome="denied",
                actor_user_id=user.id,
                ip_address=ip_address,
                details={"email": normalized_email},
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated. Please contact your workspace administrator.",
            )

        # Verify password hash
        if not verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= self.settings.max_failed_login_attempts:
                user.locked_until = now + timedelta(minutes=self.settings.account_lockout_minutes)
                logger.warning("User %s account locked due to %d failed attempts", user.email, user.failed_login_attempts)
                audit_service.log_event(
                    db=db,
                    action="account_locked",
                    outcome="failure",
                    actor_user_id=user.id,
                    ip_address=ip_address,
                    details={"failed_attempts": user.failed_login_attempts},
                )
            else:
                audit_service.log_event(
                    db=db,
                    action="login_failed_bad_password",
                    outcome="failure",
                    actor_user_id=user.id,
                    ip_address=ip_address,
                    details={"failed_attempts": user.failed_login_attempts},
                )
            db.commit()
            raise invalid_credentials_exc

        # Reset failed login counter upon successful authentication
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login_at = now

        # Server-side portal role validation if portal is specified
        if portal:
            clean_portal = str(portal).lower().strip()
            active_memberships = (
                db.query(WorkspaceMembership)
                .join(Workspace, Workspace.id == WorkspaceMembership.workspace_id)
                .filter(
                    WorkspaceMembership.user_id == user.id,
                    WorkspaceMembership.membership_status == "active",
                    Workspace.is_active.is_(True),
                )
                .all()
            )
            if clean_portal == "admin":
                has_admin = any(m.role in ("owner", "admin") for m in active_memberships)
                if not has_admin:
                    audit_service.log_event(
                        db=db,
                        action="login_denied_portal_role_mismatch",
                        outcome="denied",
                        actor_user_id=user.id,
                        ip_address=ip_address,
                        details={"portal": "admin", "reason": "non_admin_attempt"},
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="This account does not have administrator access. Please use the Employee Portal.",
                    )
            elif clean_portal == "employee":
                has_employee = any(m.role == "employee" for m in active_memberships)
                all_admin = bool(active_memberships) and all(m.role in ("owner", "admin") for m in active_memberships)
                if not active_memberships:
                    audit_service.log_event(
                        db=db,
                        action="login_denied_portal_no_membership",
                        outcome="denied",
                        actor_user_id=user.id,
                        ip_address=ip_address,
                        details={"portal": "employee", "reason": "no_active_membership"},
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="This account does not have an active company membership. Please contact your administrator.",
                    )
                if all_admin and not has_employee:
                    audit_service.log_event(
                        db=db,
                        action="login_denied_portal_role_mismatch",
                        outcome="denied",
                        actor_user_id=user.id,
                        ip_address=ip_address,
                        details={"portal": "employee", "reason": "admin_in_employee_portal"},
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="This account has administrator access. Please use the Administrator Portal.",
                    )

        # Generate tokens
        access_token = create_access_token(user_id=user.id, email=user.email)
        raw_refresh_token = generate_refresh_token()
        token_hash = hash_token(raw_refresh_token)

        refresh_session = RefreshSession(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=now + timedelta(days=self.settings.refresh_token_expire_days),
        )
        db.add(refresh_session)

        audit_service.log_event(
            db=db,
            action="login_success",
            outcome="success",
            actor_user_id=user.id,
            ip_address=ip_address,
            details={"email": user.email},
        )
        db.commit()

        token_response = TokenResponse(
            access_token=access_token,
            refresh_token=raw_refresh_token,
            token_type="bearer",
            expires_in=self.settings.access_token_expire_minutes * 60,
            must_change_password=user.must_change_password,
            user=UserResponse.model_validate(user),
        )
        return user, token_response

    def refresh_token(
        self,
        db: Session,
        raw_refresh_token: str,
        ip_address: Optional[str] = None,
    ) -> TokenResponse:
        """Rotate and issue a new access/refresh token pair."""
        tok_hash = hash_token(raw_refresh_token.strip())
        now = datetime.now(timezone.utc)

        session = db.query(RefreshSession).filter(RefreshSession.token_hash == tok_hash).first()
        if not session:
            audit_service.log_event(
                db=db,
                action="token_refresh_unknown_token",
                outcome="failure",
                ip_address=ip_address,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or unrecognized refresh token.",
            )

        # Detect reuse of revoked token (compromise mitigation)
        if session.revoked_at is not None:
            # Revoke all active sessions for this user as a precaution
            db.query(RefreshSession).filter(
                RefreshSession.user_id == session.user_id,
                RefreshSession.revoked_at.is_(None),
            ).update({"revoked_at": now})
            db.commit()

            audit_service.log_event(
                db=db,
                action="token_reuse_detected",
                outcome="denied",
                actor_user_id=session.user_id,
                ip_address=ip_address,
                details={"session_id": session.id},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has been revoked. All active sessions terminated.",
            )

        # Check expiration
        exp_utc = session.expires_at
        if exp_utc.tzinfo is None:
            exp_utc = exp_utc.replace(tzinfo=timezone.utc)

        if exp_utc < now:
            session.revoked_at = now
            db.commit()
            audit_service.log_event(
                db=db,
                action="token_refresh_expired",
                outcome="failure",
                actor_user_id=session.user_id,
                ip_address=ip_address,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired. Please log in again.",
            )

        user = db.query(User).filter(User.id == session.user_id).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account associated with this session is inactive or removed.",
            )

        # Rotate: revoke current session and issue replacement
        new_raw_refresh = generate_refresh_token()
        new_token_hash = hash_token(new_raw_refresh)

        new_session = RefreshSession(
            user_id=user.id,
            token_hash=new_token_hash,
            expires_at=now + timedelta(days=self.settings.refresh_token_expire_days),
        )
        db.add(new_session)
        db.flush()

        session.revoked_at = now
        session.replaced_by_session_id = new_session.id

        new_access_token = create_access_token(user_id=user.id, email=user.email)

        audit_service.log_event(
            db=db,
            action="token_refresh_success",
            outcome="success",
            actor_user_id=user.id,
            ip_address=ip_address,
        )
        db.commit()

        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_raw_refresh,
            token_type="bearer",
            expires_in=self.settings.access_token_expire_minutes * 60,
            must_change_password=user.must_change_password,
            user=UserResponse.model_validate(user),
        )

    def logout(
        self,
        db: Session,
        user_id: str,
        raw_refresh_token: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> bool:
        """Revoke active refresh sessions on logout."""
        now = datetime.now(timezone.utc)
        if raw_refresh_token:
            tok_hash = hash_token(raw_refresh_token.strip())
            session = db.query(RefreshSession).filter(RefreshSession.token_hash == tok_hash).first()
            if session:
                session.revoked_at = now
        else:
            # Revoke all active sessions for this user
            db.query(RefreshSession).filter(
                RefreshSession.user_id == user_id,
                RefreshSession.revoked_at.is_(None),
            ).update({"revoked_at": now})

        audit_service.log_event(
            db=db,
            action="logout",
            outcome="success",
            actor_user_id=user_id,
            ip_address=ip_address,
        )
        db.commit()
        return True

    def change_password(
        self,
        db: Session,
        user: User,
        request: ChangePasswordRequest,
        ip_address: Optional[str] = None,
    ) -> bool:
        """Update password, validate security policy, and invalidate previous sessions."""
        # Current password required unless must_change_password is true
        if not user.must_change_password:
            if not request.current_password:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is required to change your password.",
                )
            if not verify_password(request.current_password, user.password_hash):
                audit_service.log_event(
                    db=db,
                    action="password_change_bad_current",
                    outcome="failure",
                    actor_user_id=user.id,
                    ip_address=ip_address,
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is incorrect.",
                )

        # Validate new password strength
        valid, msg = validate_password_strength(request.new_password)
        if not valid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg or "Weak password.")

        # Hash and persist new password
        user.password_hash = hash_password(request.new_password)
        user.must_change_password = False

        # Invalidate existing refresh sessions
        now = datetime.now(timezone.utc)
        db.query(RefreshSession).filter(
            RefreshSession.user_id == user.id,
            RefreshSession.revoked_at.is_(None),
        ).update({"revoked_at": now})

        audit_service.log_event(
            db=db,
            action="password_change_success",
            outcome="success",
            actor_user_id=user.id,
            ip_address=ip_address,
        )
        db.commit()
        return True

    def create_owner_user(
        self,
        db: Session,
        full_name: str,
        email: str,
        password: str,
        assign_existing_workspaces: bool = False,
    ) -> Tuple[User, int]:
        """CLI provisioning helper for creating the initial workspace owner."""
        normalized_email = email.lower().strip()
        clean_name = full_name.strip()

        if not clean_name:
            raise ValueError("Owner full name is required.")
        if not normalized_email or "@" not in normalized_email:
            raise ValueError("A valid email address is required.")

        existing = db.query(User).filter(User.email == normalized_email).first()
        if existing:
            raise ValueError(f"User with email '{normalized_email}' already exists.")

        valid, msg = validate_password_strength(password)
        if not valid:
            raise ValueError(msg or "Password does not meet enterprise complexity requirements.")

        owner_user = User(
            full_name=clean_name,
            email=normalized_email,
            password_hash=hash_password(password),
            is_active=True,
            must_change_password=False,
        )
        db.add(owner_user)
        db.flush()

        assigned_count = 0
        if assign_existing_workspaces:
            # Assign all workspaces that currently do not have an owner
            all_ws = db.query(Workspace).all()
            for ws in all_ws:
                has_owner = (
                    db.query(WorkspaceMembership)
                    .filter(
                        WorkspaceMembership.workspace_id == ws.id,
                        WorkspaceMembership.role == "owner",
                        WorkspaceMembership.membership_status == "active",
                    )
                    .first()
                )
                if not has_owner:
                    membership = WorkspaceMembership(
                        user_id=owner_user.id,
                        workspace_id=ws.id,
                        role="owner",
                        membership_status="active",
                    )
                    db.add(membership)
                    assigned_count += 1

        audit_service.log_event(
            db=db,
            action="owner_user_provisioned",
            outcome="success",
            actor_user_id=owner_user.id,
            details={"email": normalized_email, "assigned_workspaces": assigned_count},
        )
        db.commit()
        db.refresh(owner_user)
        return owner_user, assigned_count

    def register_user(
        self,
        db: Session,
        full_name: str,
        email: str,
        password: str,
        workspace_name: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Tuple[User, TokenResponse]:
        """Self-registration for new enterprise users."""
        clean_name = full_name.strip()
        normalized_email = email.lower().strip()

        if not clean_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Full name is required.",
            )
        if not normalized_email or "@" not in normalized_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A valid email address is required.",
            )

        existing = db.query(User).filter(User.email == normalized_email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"An account with email '{normalized_email}' already exists. Please sign in.",
            )

        valid, msg = validate_password_strength(password)
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=msg or "Password does not meet enterprise complexity requirements.",
            )

        new_user = User(
            full_name=clean_name,
            email=normalized_email,
            password_hash=hash_password(password),
            is_active=True,
            must_change_password=False,
        )
        db.add(new_user)
        db.flush()

        # Create primary workspace for user
        ws_name = (workspace_name or "").strip() or f"{clean_name}'s Workspace"
        ws_slug = re.sub(r"[^a-z0-9]+", "-", ws_name.lower()).strip("-")
        if not ws_slug:
            ws_slug = "workspace"
        base_slug = ws_slug
        counter = 1
        while db.query(Workspace).filter(Workspace.slug == ws_slug).first():
            ws_slug = f"{base_slug}-{counter}"
            counter += 1

        new_ws = Workspace(
            name=ws_name,
            slug=ws_slug,
            description=f"Primary workspace for {clean_name}",
            is_active=True,
        )
        db.add(new_ws)
        db.flush()

        membership = WorkspaceMembership(
            user_id=new_user.id,
            workspace_id=new_ws.id,
            role="owner",
            membership_status="active",
        )
        db.add(membership)

        audit_service.log_event(
            db=db,
            action="user_registered",
            outcome="success",
            actor_user_id=new_user.id,
            ip_address=ip_address,
            details={"email": normalized_email, "workspace_id": new_ws.id},
        )
        db.commit()
        db.refresh(new_user)

        # Issue tokens directly
        return self.authenticate_user(
            db=db,
            email=normalized_email,
            password=password,
            ip_address=ip_address,
        )

    def register_admin(
        self,
        db: Session,
        full_name: str,
        company_name: str,
        email: str,
        password: str,
        ip_address: Optional[str] = None,
    ) -> Tuple[User, TokenResponse]:
        """Onboard a new company workspace and its primary administrator account atomically."""
        clean_name = (full_name or "").strip()
        clean_company = (company_name or "").strip()
        normalized_email = (email or "").lower().strip()

        if not clean_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Administrator full name is required.",
            )
        if not clean_company:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Company name is required.",
            )
        if not normalized_email or "@" not in normalized_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A valid work email address is required.",
            )

        existing = db.query(User).filter(User.email == normalized_email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"An account with email '{normalized_email}' already exists. Please sign in.",
            )

        valid, msg = validate_password_strength(password)
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=msg or "Password does not meet enterprise complexity requirements.",
            )

        try:
            # Generate deterministic, clean company slug
            base_slug = re.sub(r"[^a-z0-9]+", "-", clean_company.lower()).strip("-") or "company"
            slug = base_slug
            counter = 1
            while db.query(Workspace).filter(Workspace.slug == slug).first():
                slug = f"{base_slug}-{counter}"
                counter += 1

            new_ws = Workspace(
                name=clean_company,
                slug=slug,
                description=f"Company workspace for {clean_company}",
                is_active=True,
            )
            db.add(new_ws)
            db.flush()

            new_user = User(
                full_name=clean_name,
                email=normalized_email,
                password_hash=hash_password(password),
                is_active=True,
                must_change_password=False,
            )
            db.add(new_user)
            db.flush()

            membership = WorkspaceMembership(
                user_id=new_user.id,
                workspace_id=new_ws.id,
                role="owner",
                membership_status="active",
            )
            db.add(membership)

            audit_service.log_event(
                db=db,
                action="admin_company_registered",
                outcome="success",
                actor_user_id=new_user.id,
                ip_address=ip_address,
                workspace_id=new_ws.id,
                details={"email": normalized_email, "company_name": clean_company, "slug": slug},
            )
            db.commit()
            db.refresh(new_user)
        except Exception as exc:
            db.rollback()
            logger.error("Failed to register company and administrator atomically: %s", exc)
            raise

        return self.authenticate_user(
            db=db,
            email=normalized_email,
            password=password,
            ip_address=ip_address,
        )

    def register_employee(
        self,
        db: Session,
        full_name: str,
        email: str,
        password: str,
        code: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Tuple[User, TokenResponse]:
        """Onboard a new employee into an existing company using a validated joining code or demo workspace."""
        from backend.services.invitation_service import invitation_service

        clean_name = (full_name or "").strip()
        normalized_email = (email or "").lower().strip()
        clean_code = (code or "").strip().upper()

        if not clean_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Full name is required.",
            )
        if not normalized_email or "@" not in normalized_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A valid work email address is required.",
            )

        existing = db.query(User).filter(User.email == normalized_email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"An account with email '{normalized_email}' already exists. Please sign in.",
            )

        valid, msg = validate_password_strength(password)
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=msg or "Password does not meet enterprise complexity requirements.",
            )

        # Handle joining code or demo mode
        if not clean_code:
            if not self.settings.ekn_demo_mode:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Company joining code is required.",
                )

            # Local demonstration mode: assign employee to designated demo workspace
            demo_ws = None
            if self.settings.demo_workspace_id:
                demo_ws = db.query(Workspace).filter(Workspace.id == self.settings.demo_workspace_id).first()
            if not demo_ws:
                demo_ws = db.query(Workspace).filter(Workspace.slug == "aerospace-engineering").first()
            if not demo_ws:
                demo_ws = db.query(Workspace).filter(Workspace.name.ilike("%aerospace%")).first()
            if not demo_ws:
                demo_ws = db.query(Workspace).filter(Workspace.is_active.is_(True)).first()

            if not demo_ws:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No demonstration workspace is available. Please contact your administrator.",
                )

            try:
                new_user = User(
                    full_name=clean_name,
                    email=normalized_email,
                    password_hash=hash_password(password),
                    is_active=True,
                    must_change_password=False,
                )
                db.add(new_user)
                db.flush()

                membership = WorkspaceMembership(
                    user_id=new_user.id,
                    workspace_id=demo_ws.id,
                    role="employee",
                    membership_status="active",
                )
                db.add(membership)

                audit_service.log_event(
                    db=db,
                    action="employee_registered_demo_mode",
                    outcome="success",
                    actor_user_id=new_user.id,
                    ip_address=ip_address,
                    workspace_id=demo_ws.id,
                    details={"email": normalized_email, "workspace_name": demo_ws.name, "role": "employee"},
                )
                db.commit()
                db.refresh(new_user)
            except Exception as exc:
                db.rollback()
                logger.error("Failed to register employee in demo mode: %s", exc)
                raise

            return self.authenticate_user(
                db=db,
                email=normalized_email,
                password=password,
                ip_address=ip_address,
                portal="employee",
            )

        # Standard invitation-based registration
        is_valid, invitation, ws, inv_msg = invitation_service.validate_code(db, clean_code)
        if not is_valid or not invitation or not ws:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=inv_msg or "Invalid joining code.",
            )

        try:
            # Query invitation with row-level lock if supported to prevent race conditions
            if db.bind and db.bind.dialect.name != "sqlite":
                inv = db.query(WorkspaceInvitation).filter(WorkspaceInvitation.id == invitation.id).with_for_update().first()
            else:
                inv = invitation

            if inv.is_revoked or inv.current_usage_count >= inv.max_usage_count:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Joining code is no longer valid or has reached its usage limit.",
                )

            new_user = User(
                full_name=clean_name,
                email=normalized_email,
                password_hash=hash_password(password),
                is_active=True,
                must_change_password=False,
            )
            db.add(new_user)
            db.flush()

            role_to_assign = inv.intended_role if inv.intended_role in ("employee", "admin") else "employee"
            membership = WorkspaceMembership(
                user_id=new_user.id,
                workspace_id=inv.workspace_id,
                role=role_to_assign,
                membership_status="active",
            )
            db.add(membership)

            # Safely increment usage count
            inv.current_usage_count += 1

            audit_service.log_event(
                db=db,
                action="employee_registered_via_code",
                outcome="success",
                actor_user_id=new_user.id,
                ip_address=ip_address,
                workspace_id=inv.workspace_id,
                details={"email": normalized_email, "invitation_id": inv.id, "role": role_to_assign},
            )
            db.commit()
            db.refresh(new_user)
        except HTTPException:
            db.rollback()
            raise
        except Exception as exc:
            db.rollback()
            logger.error("Failed to register employee with joining code atomically: %s", exc)
            raise

        return self.authenticate_user(
            db=db,
            email=normalized_email,
            password=password,
            ip_address=ip_address,
        )

    def add_workspace_member(
        self,
        db: Session,
        actor_user: User,
        workspace_id: str,
        request: MemberCreateRequest,
        ip_address: Optional[str] = None,
    ) -> Tuple[WorkspaceMembership, Optional[str]]:
        """Add an existing or newly provisioned user to a workspace."""
        # Resolve actor's role in this workspace
        actor_membership = (
            db.query(WorkspaceMembership)
            .filter(
                WorkspaceMembership.workspace_id == workspace_id,
                WorkspaceMembership.user_id == actor_user.id,
                WorkspaceMembership.membership_status == "active",
            )
            .first()
        )
        if not actor_membership or actor_membership.role not in ("owner", "admin"):
            audit_service.log_event(
                db=db,
                action="add_member_denied",
                outcome="denied",
                actor_user_id=actor_user.id,
                workspace_id=workspace_id,
                ip_address=ip_address,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only workspace owners and administrators can add members.",
            )

        # Administrators can only add employees
        if actor_membership.role == "admin" and request.role in ("owner", "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrators can only add members with the 'employee' role.",
            )

        normalized_email = request.email.lower().strip()
        user = db.query(User).filter(User.email == normalized_email).first()
        temporary_password: Optional[str] = None

        if not user:
            # Provision new user with temporary credentials
            temporary_password = request.initial_password or (secrets.token_urlsafe(10) + "A1!")
            valid, msg = validate_password_strength(temporary_password)
            if not valid:
                temporary_password = secrets.token_urlsafe(12) + "A1!"

            clean_name = (request.full_name or "").strip() or request.email.split("@")[0].replace(".", " ").title()
            user = User(
                full_name=clean_name,
                email=normalized_email,
                password_hash=hash_password(temporary_password),
                is_active=True,
                must_change_password=True,
            )
            db.add(user)
            db.flush()

        # Check existing membership in this workspace
        existing_mem = (
            db.query(WorkspaceMembership)
            .filter(
                WorkspaceMembership.workspace_id == workspace_id,
                WorkspaceMembership.user_id == user.id,
            )
            .first()
        )
        if existing_mem:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"User '{user.email}' is already a member of this workspace (role: {existing_mem.role}).",
            )

        membership = WorkspaceMembership(
            user_id=user.id,
            workspace_id=workspace_id,
            role=request.role,
            membership_status="active",
        )
        db.add(membership)

        audit_service.log_event(
            db=db,
            action="member_added",
            outcome="success",
            actor_user_id=actor_user.id,
            workspace_id=workspace_id,
            resource_type="workspace_membership",
            resource_id=user.id,
            ip_address=ip_address,
            details={"target_email": user.email, "assigned_role": request.role},
        )
        db.commit()
        db.refresh(membership)
        return membership, temporary_password

    def update_workspace_member(
        self,
        db: Session,
        actor_user: User,
        workspace_id: str,
        target_user_id: str,
        request: MemberUpdateRequest,
        ip_address: Optional[str] = None,
    ) -> WorkspaceMembership:
        """Update a member's role or status with final owner protection."""
        actor_mem = (
            db.query(WorkspaceMembership)
            .filter(
                WorkspaceMembership.workspace_id == workspace_id,
                WorkspaceMembership.user_id == actor_user.id,
                WorkspaceMembership.membership_status == "active",
            )
            .first()
        )
        if not actor_mem or actor_mem.role not in ("owner", "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only workspace owners and administrators can modify member roles.",
            )

        target_mem = (
            db.query(WorkspaceMembership)
            .filter(
                WorkspaceMembership.workspace_id == workspace_id,
                WorkspaceMembership.user_id == target_user_id,
            )
            .first()
        )
        if not target_mem:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Member was not found in this workspace.",
            )

        # Protect Owner: Admins cannot modify Owner or promote members to Owner/Admin
        if actor_mem.role == "admin":
            if target_mem.role in ("owner", "admin"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Administrators cannot modify workspace owners or fellow administrators.",
                )
            if request.role in ("owner", "admin"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Administrators cannot promote members to owner or administrator.",
                )

        # Prevent leaving workspace without an owner
        if target_mem.role == "owner" and (request.role != "owner" or request.membership_status != "active"):
            owner_count = (
                db.query(func.count(WorkspaceMembership.id))
                .filter(
                    WorkspaceMembership.workspace_id == workspace_id,
                    WorkspaceMembership.role == "owner",
                    WorkspaceMembership.membership_status == "active",
                )
                .scalar()
            )
            if owner_count <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot modify or demote the final workspace owner. Designate a replacement owner first.",
                )

        if request.role:
            target_mem.role = request.role
        if request.membership_status:
            target_mem.membership_status = request.membership_status

        audit_service.log_event(
            db=db,
            action="member_updated",
            outcome="success",
            actor_user_id=actor_user.id,
            workspace_id=workspace_id,
            resource_type="workspace_membership",
            resource_id=target_user_id,
            ip_address=ip_address,
            details={"updated_role": request.role, "updated_status": request.membership_status},
        )
        db.commit()
        db.refresh(target_mem)
        return target_mem

    def remove_workspace_member(
        self,
        db: Session,
        actor_user: User,
        workspace_id: str,
        target_user_id: str,
        ip_address: Optional[str] = None,
    ) -> bool:
        """Remove a member from the workspace, preventing removal of the final owner."""
        actor_mem = (
            db.query(WorkspaceMembership)
            .filter(
                WorkspaceMembership.workspace_id == workspace_id,
                WorkspaceMembership.user_id == actor_user.id,
                WorkspaceMembership.membership_status == "active",
            )
            .first()
        )
        if not actor_mem or actor_mem.role not in ("owner", "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only workspace owners and administrators can remove members.",
            )

        target_mem = (
            db.query(WorkspaceMembership)
            .filter(
                WorkspaceMembership.workspace_id == workspace_id,
                WorkspaceMembership.user_id == target_user_id,
            )
            .first()
        )
        if not target_mem:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Member was not found in this workspace.",
            )

        # Protect Owner
        if actor_mem.role == "admin" and target_mem.role in ("owner", "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrators cannot remove workspace owners or fellow administrators.",
            )

        if target_mem.role == "owner":
            owner_count = (
                db.query(func.count(WorkspaceMembership.id))
                .filter(
                    WorkspaceMembership.workspace_id == workspace_id,
                    WorkspaceMembership.role == "owner",
                    WorkspaceMembership.membership_status == "active",
                )
                .scalar()
            )
            if owner_count <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot remove the final workspace owner. Transfer workspace ownership before removing this user.",
                )

        db.delete(target_mem)
        audit_service.log_event(
            db=db,
            action="member_removed",
            outcome="success",
            actor_user_id=actor_user.id,
            workspace_id=workspace_id,
            resource_type="workspace_membership",
            resource_id=target_user_id,
            ip_address=ip_address,
        )
        db.commit()
        return True


auth_service = AuthService()
