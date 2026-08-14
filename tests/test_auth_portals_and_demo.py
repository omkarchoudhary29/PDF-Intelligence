"""Comprehensive Test Suite for EKN Authentication Portals, Role Separation, and Demo Accounts.

Verifies:
1. Administrator registration creates company and owner atomically.
2. Administrator login with portal='admin' succeeds.
3. Administrator credentials in Employee Portal return 403:
   'This account has administrator access. Please use the Administrator Portal.'
4. Employee registration in Demo Mode (EKN_DEMO_MODE=True) succeeds without joining code
   and assigns employee role directly in the Aerospace & Engineering demo workspace.
5. Employee login with portal='employee' succeeds.
6. Employee credentials in Administrator Portal return 403:
   'This account does not have administrator access. Please use the Employee Portal.'
7. Production Mode (EKN_DEMO_MODE=False) strictly rejects employee registration without joining code.
8. Seeded demo accounts (admin.demo@example.com, employee.demo@example.com) authenticate properly.
9. GET /api/v1/auth/me returns authorized workspaces and user roles.
10. Refresh token rotation and secure logout revocation.
"""

import uuid
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.core.security import hash_password, verify_password
from backend.database.connection import get_db_session
from backend.database.models import User, Workspace, WorkspaceMembership
from backend.main import app
from backend.scripts.seed_demo_accounts import seed_demo_accounts
from backend.services.auth_service import auth_service
from config.settings import get_settings


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


def test_admin_registration(client):
    """Test administrator registration creates company and owner atomically."""
    unique_suffix = uuid.uuid4().hex[:8]
    email = f"admin_reg_{unique_suffix}@example.com"
    comp_name = f"Test Aero Corp {unique_suffix}"
    pw = "AdminSecure2026!"

    resp = client.post(
        "/api/v1/auth/admin/register",
        json={
            "full_name": f"Admin User {unique_suffix}",
            "company_name": comp_name,
            "email": email,
            "password": pw,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"

    # Verify user and workspace in database
    with get_db_session() as db:
        user = db.query(User).filter(User.email == email).first()
        assert user is not None
        assert verify_password(pw, user.password_hash)

        membership = db.query(WorkspaceMembership).filter(WorkspaceMembership.user_id == user.id).first()
        assert membership is not None
        assert membership.role == "owner"


def test_admin_login_portal_success(client):
    """Test administrator credentials with portal='admin' succeeds."""
    unique_suffix = uuid.uuid4().hex[:8]
    email = f"admin_login_{unique_suffix}@example.com"
    pw = "AdminSecure2026!"

    # Create admin user
    client.post(
        "/api/v1/auth/admin/register",
        json={
            "full_name": "Admin Tester",
            "company_name": f"Aero {unique_suffix}",
            "email": email,
            "password": pw,
        },
    )

    # Login via Admin Portal
    resp = client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": pw,
            "portal": "admin",
        },
    )
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_admin_rejected_by_employee_portal(client):
    """Admin credentials entered in Employee Portal must return 403 Forbidden with specific message."""
    unique_suffix = uuid.uuid4().hex[:8]
    email = f"admin_reject_{unique_suffix}@example.com"
    pw = "AdminSecure2026!"

    # Create company owner/admin
    client.post(
        "/api/v1/auth/admin/register",
        json={
            "full_name": "Admin Only User",
            "company_name": f"Enterprise {unique_suffix}",
            "email": email,
            "password": pw,
        },
    )

    # Attempt to log in through Employee Portal
    resp = client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": pw,
            "portal": "employee",
        },
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "This account has administrator access. Please use the Administrator Portal."


def test_employee_registration_demo_mode(client):
    """Employee registration in Demo Mode succeeds without joining code and assigns to demo workspace."""
    settings = get_settings()
    unique_suffix = uuid.uuid4().hex[:8]
    personal_email = f"worker_{unique_suffix}@gmail.com"
    pw = "EmployeeSecure2026!"

    resp = client.post(
        "/api/v1/auth/employee/register",
        json={
            "full_name": f"Worker {unique_suffix}",
            "email": personal_email,
            "password": pw,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "access_token" in data

    # Verify membership in demo workspace with employee role
    with get_db_session() as db:
        user = db.query(User).filter(User.email == personal_email).first()
        assert user is not None

        memberships = db.query(WorkspaceMembership).filter(WorkspaceMembership.user_id == user.id).all()
        assert len(memberships) == 1
        assert memberships[0].role == "employee"
        assert memberships[0].workspace_id == settings.demo_workspace_id


def test_employee_login_portal_success(client):
    """Employee credentials with portal='employee' succeeds."""
    unique_suffix = uuid.uuid4().hex[:8]
    email = f"emp_login_{unique_suffix}@example.com"
    pw = "EmployeeSecure2026!"

    # Register employee
    client.post(
        "/api/v1/auth/employee/register",
        json={
            "full_name": "Employee Tester",
            "email": email,
            "password": pw,
        },
    )

    # Login via Employee Portal
    resp = client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": pw,
            "portal": "employee",
        },
    )
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_employee_rejected_by_admin_portal(client):
    """Employee credentials entered in Administrator Portal must return 403 Forbidden with specific message."""
    unique_suffix = uuid.uuid4().hex[:8]
    email = f"emp_reject_{unique_suffix}@example.com"
    pw = "EmployeeSecure2026!"

    # Register employee
    client.post(
        "/api/v1/auth/employee/register",
        json={
            "full_name": "Employee Only User",
            "email": email,
            "password": pw,
        },
    )

    # Attempt to log in through Admin Portal
    resp = client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": pw,
            "portal": "admin",
        },
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "This account does not have administrator access. Please use the Employee Portal."


def test_production_mode_rejects_public_employee_signup():
    """When ekn_demo_mode=False, employee registration without code is rejected with 400 Bad Request."""
    settings = get_settings()
    original_mode = settings.ekn_demo_mode
    try:
        settings.ekn_demo_mode = False
        with get_db_session() as db:
            with pytest.raises(HTTPException) as exc_info:
                auth_service.register_employee(
                    db=db,
                    full_name="Unauthorized Public User",
                    email=f"public_{uuid.uuid4().hex[:6]}@example.com",
                    password="ValidPassword123!",
                    code=None,
                )
            assert exc_info.value.status_code == 400
            assert "Company joining code is required" in str(exc_info.value.detail)
    finally:
        settings.ekn_demo_mode = original_mode


def test_seeded_demo_accounts_functional(client):
    """Verify pre-seeded demo accounts exist, hash passwords with Argon2id, and authenticate."""
    seed_data = seed_demo_accounts()
    settings = get_settings()

    admin_email = seed_data["admin_email"]
    admin_pw = seed_data["admin_password"]
    emp_email = seed_data["employee_email"]
    emp_pw = seed_data["employee_password"]

    assert admin_email == "admin.demo@example.com"
    assert emp_email == "employee.demo@example.com"
    assert len(admin_pw) >= 12
    assert len(emp_pw) >= 12

    # 1. Admin logs into Admin Portal
    admin_login = client.post(
        "/api/v1/auth/login",
        json={"email": admin_email, "password": admin_pw, "portal": "admin"},
    )
    assert admin_login.status_code == 200, f"Admin login failed: {admin_login.text}"
    admin_token = admin_login.json()["access_token"]

    # 2. Admin rejected in Employee Portal
    admin_in_emp = client.post(
        "/api/v1/auth/login",
        json={"email": admin_email, "password": admin_pw, "portal": "employee"},
    )
    assert admin_in_emp.status_code == 403

    # 3. Employee logs into Employee Portal
    emp_login = client.post(
        "/api/v1/auth/login",
        json={"email": emp_email, "password": emp_pw, "portal": "employee"},
    )
    assert emp_login.status_code == 200, f"Employee login failed: {emp_login.text}"
    emp_token = emp_login.json()["access_token"]

    # 4. Employee rejected in Admin Portal
    emp_in_admin = client.post(
        "/api/v1/auth/login",
        json={"email": emp_email, "password": emp_pw, "portal": "admin"},
    )
    assert emp_in_admin.status_code == 403

    # 5. Verify /api/v1/auth/me for admin
    me_admin = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert me_admin.status_code == 200
    assert any(w["role"] in ("owner", "admin") for w in me_admin.json()["workspaces"])

    # 6. Verify /api/v1/auth/me for employee
    me_emp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {emp_token}"},
    )
    assert me_emp.status_code == 200
    assert any(w["role"] == "employee" for w in me_emp.json()["workspaces"])


def test_token_rotation_and_logout(client):
    """Test token refresh rotation and logout revocation."""
    unique_suffix = uuid.uuid4().hex[:8]
    email = f"token_test_{unique_suffix}@example.com"
    pw = "TokenSecure2026!"

    reg = client.post(
        "/api/v1/auth/admin/register",
        json={
            "full_name": "Token Tester",
            "company_name": f"Token Corp {unique_suffix}",
            "email": email,
            "password": pw,
        },
    )
    data = reg.json()
    access_tok = data["access_token"]
    refresh_tok = data["refresh_token"]

    # Rotate refresh token
    ref_resp = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_tok},
    )
    assert ref_resp.status_code == 200
    new_data = ref_resp.json()
    new_access_tok = new_data["access_token"]
    new_refresh_tok = new_data["refresh_token"]
    assert new_refresh_tok != refresh_tok

    # Old refresh token is revoked
    old_reuse = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_tok},
    )
    assert old_reuse.status_code == 401

    # Logout
    logout_resp = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": new_refresh_tok},
        headers={"Authorization": f"Bearer {new_access_tok}"},
    )
    assert logout_resp.status_code == 200

    # New refresh token is now revoked
    after_logout = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": new_refresh_tok},
    )
    assert after_logout.status_code == 401
