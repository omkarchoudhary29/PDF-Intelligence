"""Pytest conftest providing backward-compatibility fixtures for legacy test suites."""

import pytest
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.core.dependencies import (
    get_active_user,
    get_current_user,
    get_workspace_membership,
    require_workspace_admin,
    require_workspace_member,
    require_workspace_owner,
)
from backend.database.connection import get_db
from backend.database.models import User, Workspace, WorkspaceMembership


MOCK_LEGACY_USER_ID = "legacy-test-owner-001"
MOCK_LEGACY_EMAIL = "legacy_owner@enterprise.com"


@pytest.fixture(autouse=True)
def legacy_auth_override(request):
    """Automatically mock authenticated Owner credentials for legacy test suites (Phases 1-9).

    This fixture is explicitly bypassed for tests in `test_security.py` so that
    actual token decoding, RBAC checks, account lockouts, and cross-tenant
    isolation can be strictly verified against real unauthenticated/unauthorized scenarios.
    """
    file_name = request.node.fspath.basename if hasattr(request.node, "fspath") else ""
    if (
        "test_security" in file_name
        or "test_auth" in file_name
        or "test_platform" in file_name
        or "test_e2e" in file_name
        or "test_phase1" in file_name
        or "test_phase3" in file_name
        or "test_phase4" in file_name
        or "test_phase5" in file_name
        or "test_phase6" in file_name
        or "test_final_phase" in file_name
        or "test_targeted" in file_name
    ):
        yield
        return

    from backend.main import app

    def _get_or_create_mock_user(db: Session) -> User:
        user = db.query(User).filter(User.id == MOCK_LEGACY_USER_ID).first()
        if not user:
            user = User(
                id=MOCK_LEGACY_USER_ID,
                email=MOCK_LEGACY_EMAIL,
                full_name="Legacy Test Owner",
                password_hash="mock_argon2_hash",
                is_active=True,
                must_change_password=False,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        return user

    def override_get_current_user(db: Session = Depends(get_db)) -> User:
        return _get_or_create_mock_user(db)

    def override_get_active_user(db: Session = Depends(get_db)) -> User:
        return _get_or_create_mock_user(db)

    def override_require_member(
        workspace_id: str,
        db: Session = Depends(get_db),
    ) -> WorkspaceMembership:
        ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if not ws:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace '{workspace_id}' was not found.",
            )
        user = _get_or_create_mock_user(db)
        membership = (
            db.query(WorkspaceMembership)
            .filter(
                WorkspaceMembership.workspace_id == workspace_id,
                WorkspaceMembership.user_id == user.id,
            )
            .first()
        )
        if not membership:
            membership = WorkspaceMembership(
                user_id=user.id,
                workspace_id=workspace_id,
                role="owner",
                membership_status="active",
            )
            db.add(membership)
            db.commit()
            db.refresh(membership)
        return membership

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_active_user] = override_get_active_user
    app.dependency_overrides[get_workspace_membership] = override_require_member
    app.dependency_overrides[require_workspace_member] = override_require_member
    app.dependency_overrides[require_workspace_admin] = override_require_member
    app.dependency_overrides[require_workspace_owner] = override_require_member

    try:
        yield
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_active_user, None)
        app.dependency_overrides.pop(get_workspace_membership, None)
        app.dependency_overrides.pop(require_workspace_member, None)
        app.dependency_overrides.pop(require_workspace_admin, None)
        app.dependency_overrides.pop(require_workspace_owner, None)
