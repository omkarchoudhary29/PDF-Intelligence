"""Workspace Invitation and Joining-Code Management Service.

Provides cryptographically secure generation, validation, and usage tracking
for employee onboarding joining codes without exposing or storing raw codes in plaintext.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import logging
import secrets
from typing import List, Optional, Tuple
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from backend.database.models import User, Workspace, WorkspaceInvitation, WorkspaceMembership
from backend.services.audit_service import audit_service

logger = logging.getLogger("ekn.security.invitation_service")


class InvitationService:
    """Manages company joining codes and invitations."""

    @staticmethod
    def hash_code(raw_code: str) -> str:
        """Deterministically compute SHA-256 hash of normalized code."""
        normalized = raw_code.strip().upper()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def create_invitation(
        self,
        db: Session,
        workspace_id: str,
        creator_user_id: str,
        max_usage_count: int = 10,
        expires_in_days: int = 7,
        intended_role: str = "employee",
    ) -> Tuple[WorkspaceInvitation, str]:
        """Generate a new secure joining code and persist only its SHA-256 digest."""
        # Verify workspace exists and is active
        ws = db.query(Workspace).filter(Workspace.id == workspace_id, Workspace.is_active.is_(True)).first()
        if not ws:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace '{workspace_id}' was not found or is inactive.",
            )

        # Generate human-readable yet cryptographically secure code: JOIN-XXXX-XXXX
        part1 = secrets.token_hex(3).upper()
        part2 = secrets.token_hex(3).upper()
        raw_code = f"JOIN-{part1}-{part2}"
        code_hash = self.hash_code(raw_code)

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=expires_in_days)

        invitation = WorkspaceInvitation(
            workspace_id=workspace_id,
            code_hash=code_hash,
            created_by_user_id=creator_user_id,
            intended_role=intended_role,
            expires_at=expires_at,
            max_usage_count=max_usage_count,
            current_usage_count=0,
            is_revoked=False,
        )
        db.add(invitation)
        db.commit()
        db.refresh(invitation)

        audit_service.log_event(
            db=db,
            action="invitation_created",
            outcome="success",
            actor_user_id=creator_user_id,
            workspace_id=workspace_id,
            details={
                "invitation_id": invitation.id,
                "intended_role": intended_role,
                "max_usage": max_usage_count,
                "expires_at": expires_at.isoformat(),
            },
        )
        return invitation, raw_code

    def validate_code(
        self,
        db: Session,
        raw_code: str,
    ) -> Tuple[bool, Optional[WorkspaceInvitation], Optional[Workspace], str]:
        """Verify code validity without incrementing usage."""
        clean_code = (raw_code or "").strip().upper()
        if not clean_code or len(clean_code) < 4:
            return False, None, None, "Invalid joining code format."

        code_hash = self.hash_code(clean_code)
        invitation = db.query(WorkspaceInvitation).filter(WorkspaceInvitation.code_hash == code_hash).first()

        if not invitation:
            return False, None, None, "Invalid joining code."

        if invitation.is_revoked:
            return False, invitation, None, "This joining code has been revoked."

        now = datetime.now(timezone.utc)
        exp = invitation.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)

        if exp < now:
            return False, invitation, None, "This joining code has expired."

        if invitation.current_usage_count >= invitation.max_usage_count:
            return False, invitation, None, "This joining code has reached its maximum usage limit."

        ws = db.query(Workspace).filter(Workspace.id == invitation.workspace_id).first()
        if not ws or not ws.is_active:
            return False, invitation, None, "The associated company workspace is not active."

        return True, invitation, ws, "Joining code is valid."

    def list_invitations(
        self,
        db: Session,
        workspace_id: str,
    ) -> List[dict]:
        """List invitations for an authorized workspace administrator."""
        now = datetime.now(timezone.utc)
        invitations = (
            db.query(WorkspaceInvitation)
            .filter(WorkspaceInvitation.workspace_id == workspace_id)
            .order_by(WorkspaceInvitation.created_at.desc())
            .all()
        )
        items = []
        for inv in invitations:
            exp = inv.expires_at
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            is_valid = (
                not inv.is_revoked
                and exp >= now
                and inv.current_usage_count < inv.max_usage_count
            )
            items.append({
                "id": inv.id,
                "workspace_id": inv.workspace_id,
                "intended_role": inv.intended_role,
                "expires_at": inv.expires_at,
                "max_usage_count": inv.max_usage_count,
                "current_usage_count": inv.current_usage_count,
                "is_revoked": inv.is_revoked,
                "is_valid": is_valid,
                "created_at": inv.created_at,
            })
        return items

    def revoke_invitation(
        self,
        db: Session,
        workspace_id: str,
        invitation_id: str,
        actor_user_id: str,
    ) -> bool:
        """Revoke an active invitation so it can no longer be used."""
        invitation = (
            db.query(WorkspaceInvitation)
            .filter(
                WorkspaceInvitation.id == invitation_id,
                WorkspaceInvitation.workspace_id == workspace_id,
            )
            .first()
        )
        if not invitation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Invitation '{invitation_id}' was not found in this workspace.",
            )

        invitation.is_revoked = True
        db.commit()

        audit_service.log_event(
            db=db,
            action="invitation_revoked",
            outcome="success",
            actor_user_id=actor_user_id,
            workspace_id=workspace_id,
            details={"invitation_id": invitation_id},
        )
        return True


invitation_service = InvitationService()
