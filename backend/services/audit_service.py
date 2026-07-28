"""Enterprise Security Audit Logging Service.

Provides immutable audit trail tracking for security-sensitive actions:
- Authentications (login, logout, token refresh, password changes)
- Workspace membership modifications (add, remove, role changes)
- Document operations (upload, delete)
- Report downloads and deletions
- Access denied events

Strictly sanitizes all metadata to exclude passwords, tokens, API keys, and sensitive body text.
"""

from datetime import datetime
import logging
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session

from backend.database.models import SecurityAuditLog

logger = logging.getLogger("ekn.security.audit")

SENSITIVE_KEYS = {
    "password",
    "current_password",
    "new_password",
    "password_hash",
    "access_token",
    "refresh_token",
    "token",
    "token_hash",
    "api_key",
    "gemini_api_key",
    "authorization",
}


def sanitize_audit_details(details: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Recursively scrub credentials and sensitive fields from audit payload."""
    if not details or not isinstance(details, dict):
        return None

    sanitized = {}
    for k, v in details.items():
        if k.lower() in SENSITIVE_KEYS:
            sanitized[k] = "[REDACTED]"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_audit_details(v)
        elif isinstance(v, str) and len(v) > 500:
            # Truncate overly long text fields to avoid dumping entire documents into audit logs
            sanitized[k] = v[:500] + "... [TRUNCATED]"
        else:
            sanitized[k] = v
    return sanitized


class AuditService:
    """Records security-sensitive actions into the database audit log."""

    @staticmethod
    def log_event(
        db: Session,
        action: str,
        outcome: str,
        actor_user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        details_json: Optional[Dict[str, Any]] = None,
    ) -> SecurityAuditLog:
        """Create and persist a sanitized security audit log entry."""
        effective_details = details if details is not None else details_json
        sanitized = sanitize_audit_details(effective_details)
        log_entry = SecurityAuditLog(
            action=action,
            outcome=outcome,
            actor_user_id=actor_user_id,
            workspace_id=workspace_id,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            details_json=sanitized,
        )
        try:
            db.add(log_entry)
            db.commit()
            db.refresh(log_entry)
            logger.info(
                "SecurityAudit: action=%s outcome=%s actor=%s ws=%s resource=%s/%s",
                action,
                outcome,
                actor_user_id,
                workspace_id,
                resource_type,
                resource_id,
            )
            return log_entry
        except Exception as exc:
            db.rollback()
            logger.error("Failed to write security audit log: %s", exc)
            return log_entry

    log_action = log_event


audit_service = AuditService()
