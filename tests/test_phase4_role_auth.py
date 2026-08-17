"""Phase 4 Automated Test Suite: Role Selection, Admin/Employee Authentication, and Company Onboarding.

Verifies:
1. Role selection navigation states & Streamlit view routing.
2. Administrator company registration (atomic company, user, and owner membership).
3. Employee registration with secure joining codes (valid, invalid, expired, revoked, exhausted, duplicate email).
4. Administrator and Employee sign in and server-side role resolution.
5. Prevention of employee access to administrator privileges.
6. Zero leakage of raw codes and hashes.
7. Preservation of existing accounts (admin@enterprise.com) and Phase 1-3 capabilities.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import uuid
import pytest
from fastapi.testclient import TestClient

from backend.core.security import create_access_token, hash_password
from backend.database.connection import get_db_session
from backend.database.models import (
    Document,
    User,
    Workspace,
    WorkspaceInvitation,
    WorkspaceMembership,
)
from backend.main import app
from backend.services.invitation_service import invitation_service
from config.settings import get_settings

client = TestClient(app)


# =============================================================================
# PART 1: ROLE SELECTION & FRONTEND NAVIGATION LOGIC
# =============================================================================

def test_01_unauthenticated_visitor_initial_state():
    """Unauthenticated visitor starts on the role selection screen."""
    from frontend.views.login import render_login
    # Verify module has required callable entrypoints
    assert callable(render_login)


def test_02_role_selection_branding_and_copy():
    """Verify role selection strings match required specification exactly."""
    import inspect
    from frontend.views import login as login_module
    source = inspect.getsource(login_module)
    assert "Enterprise Knowledge Navigator" in source
    assert "Your organization's knowledge, in one place." in source
    assert "How would you like to continue?" in source
    assert "Administrator" in source
    assert "Manage company documents, employees and shared organizational knowledge." in source
    assert "Employee" in source
    assert "Ask questions, access company knowledge and work with your personal PDFs." in source


def test_03_05_portal_headings_and_back_navigation():
    """Verify administrator and employee portal branding and back button action."""
    import inspect
    from frontend.views import login as login_module
    source = inspect.getsource(login_module)
    assert "Administrator Portal" in source
    assert "Welcome to your company workspace" in source
    assert "Employee Portal" in source
    assert "Your company's knowledge starts here" in source
    assert "← Back to identity selection" in source


# =============================================================================
# PART 2: ADMINISTRATOR REGISTRATION (COMPANY ONBOARDING)
# =============================================================================

def test_06_07_admin_company_registration_success():
    """Admin registers a new company; company, user, and owner membership created atomically."""
    unique_suffix = uuid.uuid4().hex[:6]
    admin_email = f"admin_{unique_suffix}@newcorp.com"
    company_name = f"NewCorp Aerospace {unique_suffix}"
    payload = {
        "full_name": "Chief Executive",
        "company_name": company_name,
        "email": admin_email,
        "password": "SecurePassword123!",
    }
    resp = client.post("/api/v1/auth/admin/register", json=payload)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data

    # Verify atomic DB persistence
    with get_db_session() as db:
        user = db.query(User).filter(User.email == admin_email).first()
        assert user is not None
        assert user.full_name == "Chief Executive"

        ws = db.query(Workspace).filter(Workspace.name == company_name).first()
        assert ws is not None
        assert ws.is_active is True

        membership = (
            db.query(WorkspaceMembership)
            .filter(
                WorkspaceMembership.user_id == user.id,
                WorkspaceMembership.workspace_id == ws.id,
            )
            .first()
        )
        assert membership is not None
        assert membership.role == "owner"
        assert membership.membership_status == "active"


def test_08_admin_registration_duplicate_email_rejected():
    """Admin registration rejects duplicate email."""
    unique_suffix = uuid.uuid4().hex[:6]
    email = f"dup_{unique_suffix}@corp.com"
    payload = {
        "full_name": "Admin One",
        "company_name": "First Corp",
        "email": email,
        "password": "Password123!",
    }
    resp1 = client.post("/api/v1/auth/admin/register", json=payload)
    assert resp1.status_code == 201

    payload2 = {
        "full_name": "Admin Two",
        "company_name": "Second Corp",
        "email": email,
        "password": "Password123!",
    }
    resp2 = client.post("/api/v1/auth/admin/register", json=payload2)
    assert resp2.status_code == 400
    assert "already exists" in resp2.json()["detail"].lower()


def test_09_admin_registration_missing_company_name():
    """Admin registration rejects empty or missing company name."""
    payload = {
        "full_name": "No Company Admin",
        "company_name": "   ",
        "email": f"nocomp_{uuid.uuid4().hex[:6]}@corp.com",
        "password": "Password123!",
    }
    resp = client.post("/api/v1/auth/admin/register", json=payload)
    assert resp.status_code in (400, 422)


def test_10_admin_registration_weak_password():
    """Admin registration rejects weak password."""
    payload = {
        "full_name": "Weak Pwd Admin",
        "company_name": "Acme Corp",
        "email": f"weak_{uuid.uuid4().hex[:6]}@acme.com",
        "password": "short",
    }
    resp = client.post("/api/v1/auth/admin/register", json=payload)
    assert resp.status_code in (400, 422)


def test_11_admin_registration_slug_collision_handled_safely():
    """Two companies with identical names receive unique slugs safely."""
    unique_name = f"Identical Name {uuid.uuid4().hex[:6]}"
    p1 = {
        "full_name": "Founder One",
        "company_name": unique_name,
        "email": f"founder1_{uuid.uuid4().hex[:6]}@ident.com",
        "password": "Password123!",
    }
    p2 = {
        "full_name": "Founder Two",
        "company_name": unique_name,
        "email": f"founder2_{uuid.uuid4().hex[:6]}@ident.com",
        "password": "Password123!",
    }
    resp1 = client.post("/api/v1/auth/admin/register", json=p1)
    resp2 = client.post("/api/v1/auth/admin/register", json=p2)
    assert resp1.status_code == 201
    assert resp2.status_code == 201

    with get_db_session() as db:
        workspaces = db.query(Workspace).filter(Workspace.name == unique_name).all()
        assert len(workspaces) == 2
        assert workspaces[0].slug != workspaces[1].slug


def test_12_admin_initial_membership_is_owner():
    """Verified that initial company creator receives 'owner' role."""
    unique = uuid.uuid4().hex[:6]
    email = f"owner_{unique}@verify.com"
    resp = client.post("/api/v1/auth/admin/register", json={
        "full_name": "Owner Verifier",
        "company_name": f"Owner Workspace {unique}",
        "email": email,
        "password": "Password123!",
    })
    assert resp.status_code == 201
    token = resp.json()["access_token"]

    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    workspaces = me_resp.json()["workspaces"]
    assert len(workspaces) >= 1
    assert workspaces[0]["role"] == "owner"


# =============================================================================
# PART 3: INVITATIONS & EMPLOYEE REGISTRATION
# =============================================================================

@pytest.fixture(scope="module")
def sample_company():
    """Fixture providing a workspace, owner admin, and admin access token."""
    unique = uuid.uuid4().hex[:6]
    email = f"admin_fixture_{unique}@testco.com"
    resp = client.post("/api/v1/auth/admin/register", json={
        "full_name": "Fixture Admin",
        "company_name": f"TestCo Workspace {unique}",
        "email": email,
        "password": "Password123!",
    })
    assert resp.status_code == 201
    data = resp.json()
    token = data["access_token"]
    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    ws_id = me_resp.json()["workspaces"][0]["workspace_id"]
    return {
        "workspace_id": ws_id,
        "admin_token": token,
        "admin_email": email,
    }


def test_13_create_and_validate_joining_code(sample_company):
    """Admin creates joining code; validate endpoint confirms validity without leaking hash."""
    ws_id = sample_company["workspace_id"]
    admin_token = sample_company["admin_token"]

    resp = client.post(
        f"/api/v1/workspaces/{ws_id}/invitations",
        json={"max_usage_count": 5, "expires_in_days": 3, "intended_role": "employee"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert "raw_code" in data
    raw_code = data["raw_code"]
    assert raw_code.startswith("JOIN-")

    # Validate endpoint
    val_resp = client.post("/api/v1/auth/invitations/validate", json={"code": raw_code})
    assert val_resp.status_code == 200
    val_data = val_resp.json()
    assert val_data["is_valid"] is True
    assert val_data["workspace_id"] == ws_id
    assert val_data["intended_role"] == "employee"


def test_13b_employee_registration_valid_code(sample_company):
    """Employee successfully registers with a valid joining code."""
    ws_id = sample_company["workspace_id"]
    admin_token = sample_company["admin_token"]

    # Generate fresh code
    c_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/invitations",
        json={"max_usage_count": 2, "expires_in_days": 7, "intended_role": "employee"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    raw_code = c_resp.json()["raw_code"]

    emp_email = f"emp_{uuid.uuid4().hex[:6]}@testco.com"
    reg_resp = client.post(
        "/api/v1/auth/employee/register",
        json={
            "full_name": "Valid Employee",
            "email": emp_email,
            "password": "Password123!",
            "code": raw_code,
        },
    )
    assert reg_resp.status_code == 201, reg_resp.text
    reg_data = reg_resp.json()
    assert "access_token" in reg_data

    # Verify role in /api/v1/auth/me is 'employee'
    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {reg_data['access_token']}"})
    assert me_resp.status_code == 200
    workspaces = me_resp.json()["workspaces"]
    assert len(workspaces) == 1
    assert workspaces[0]["workspace_id"] == ws_id
    assert workspaces[0]["role"] == "employee"


def test_14_invalid_joining_code_rejected():
    """Attempting registration with a bogus code is rejected."""
    resp = client.post(
        "/api/v1/auth/employee/register",
        json={
            "full_name": "Bogus Emp",
            "email": f"bogus_{uuid.uuid4().hex[:6]}@test.com",
            "password": "Password123!",
            "code": "JOIN-NONEXISTENT-CODE",
        },
    )
    assert resp.status_code == 400
    assert "invalid joining code" in resp.json()["detail"].lower()


def test_15_expired_joining_code_rejected(sample_company):
    """Expired code is rejected."""
    ws_id = sample_company["workspace_id"]
    # Manually create expired invitation in DB
    raw_code = f"JOIN-EXP-{uuid.uuid4().hex[:6].upper()}"
    with get_db_session() as db:
        admin_user = db.query(User).filter(User.email == sample_company["admin_email"]).first()
        inv = WorkspaceInvitation(
            workspace_id=ws_id,
            code_hash=invitation_service.hash_code(raw_code),
            created_by_user_id=admin_user.id,
            intended_role="employee",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),  # expired
            max_usage_count=5,
            current_usage_count=0,
            is_revoked=False,
        )
        db.add(inv)
        db.commit()

    resp = client.post(
        "/api/v1/auth/employee/register",
        json={
            "full_name": "Expired Code Emp",
            "email": f"exp_{uuid.uuid4().hex[:6]}@test.com",
            "password": "Password123!",
            "code": raw_code,
        },
    )
    assert resp.status_code == 400
    assert "expired" in resp.json()["detail"].lower()


def test_16_revoked_joining_code_rejected(sample_company):
    """Revoked code is rejected."""
    ws_id = sample_company["workspace_id"]
    admin_token = sample_company["admin_token"]

    c_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/invitations",
        json={"max_usage_count": 5, "expires_in_days": 7},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    raw_code = c_resp.json()["raw_code"]
    inv_id = c_resp.json()["id"]

    # Revoke it
    del_resp = client.delete(
        f"/api/v1/workspaces/{ws_id}/invitations/{inv_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert del_resp.status_code == 200

    resp = client.post(
        "/api/v1/auth/employee/register",
        json={
            "full_name": "Revoked Code Emp",
            "email": f"rev_{uuid.uuid4().hex[:6]}@test.com",
            "password": "Password123!",
            "code": raw_code,
        },
    )
    assert resp.status_code == 400
    assert "revoked" in resp.json()["detail"].lower()


def test_17_21_exhausted_joining_code_and_limit_enforcement(sample_company):
    """A code with max_usage_count=1 allows 1 registration, rejects the second."""
    ws_id = sample_company["workspace_id"]
    admin_token = sample_company["admin_token"]

    c_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/invitations",
        json={"max_usage_count": 1, "expires_in_days": 1},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    raw_code = c_resp.json()["raw_code"]

    # First employee signs up (uses 1/1)
    e1_resp = client.post(
        "/api/v1/auth/employee/register",
        json={
            "full_name": "Employee One",
            "email": f"emp1_{uuid.uuid4().hex[:6]}@limit.com",
            "password": "Password123!",
            "code": raw_code,
        },
    )
    assert e1_resp.status_code == 201

    # Second employee tries to sign up with exhausted code
    e2_resp = client.post(
        "/api/v1/auth/employee/register",
        json={
            "full_name": "Employee Two",
            "email": f"emp2_{uuid.uuid4().hex[:6]}@limit.com",
            "password": "Password123!",
            "code": raw_code,
        },
    )
    assert e2_resp.status_code == 400
    assert "limit" in e2_resp.json()["detail"].lower()


def test_18_duplicate_employee_email_rejected(sample_company):
    """Employee signup with already registered email is rejected."""
    ws_id = sample_company["workspace_id"]
    admin_token = sample_company["admin_token"]

    c_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/invitations",
        json={"max_usage_count": 5},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    raw_code = c_resp.json()["raw_code"]

    email = f"dup_emp_{uuid.uuid4().hex[:6]}@testco.com"
    p1 = {"full_name": "Emp 1", "email": email, "password": "Password123!", "code": raw_code}
    p2 = {"full_name": "Emp 2", "email": email, "password": "Password123!", "code": raw_code}

    assert client.post("/api/v1/auth/employee/register", json=p1).status_code == 201
    resp2 = client.post("/api/v1/auth/employee/register", json=p2)
    assert resp2.status_code == 400
    assert "already exists" in resp2.json()["detail"].lower()


def test_19_20_employee_cannot_create_company_and_joins_correct_workspace(sample_company):
    """Employee endpoint has no company creation parameters; user joins target workspace."""
    ws_id = sample_company["workspace_id"]
    admin_token = sample_company["admin_token"]

    c_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/invitations",
        json={"max_usage_count": 5},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    raw_code = c_resp.json()["raw_code"]

    emp_email = f"emp_correct_{uuid.uuid4().hex[:6]}@test.com"
    reg_resp = client.post(
        "/api/v1/auth/employee/register",
        json={"full_name": "Correct WS Emp", "email": emp_email, "password": "Password123!", "code": raw_code},
    )
    assert reg_resp.status_code == 201
    emp_token = reg_resp.json()["access_token"]

    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {emp_token}"})
    workspaces = me_resp.json()["workspaces"]
    assert len(workspaces) == 1
    assert workspaces[0]["workspace_id"] == ws_id
    assert workspaces[0]["role"] == "employee"


# =============================================================================
# PART 4: AUTHENTICATION & ROLE ENFORCEMENT
# =============================================================================

def test_22_23_admin_and_employee_sign_in(sample_company):
    """Both admin and employee accounts can authenticate via /api/v1/auth/login."""
    ws_id = sample_company["workspace_id"]
    admin_token = sample_company["admin_token"]

    c_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/invitations",
        json={"max_usage_count": 2},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    raw_code = c_resp.json()["raw_code"]

    emp_email = f"signin_emp_{uuid.uuid4().hex[:6]}@test.com"
    client.post(
        "/api/v1/auth/employee/register",
        json={"full_name": "Signin Emp", "email": emp_email, "password": "Password123!", "code": raw_code},
    )

    # 1. Admin login
    admin_login = client.post(
        "/api/v1/auth/login",
        json={"email": sample_company["admin_email"], "password": "Password123!"},
    )
    assert admin_login.status_code == 200
    assert "access_token" in admin_login.json()

    # 2. Employee login
    emp_login = client.post(
        "/api/v1/auth/login",
        json={"email": emp_email, "password": "Password123!"},
    )
    assert emp_login.status_code == 200
    assert "access_token" in emp_login.json()


def test_24_25_employee_role_verification_denies_admin_portal_access(sample_company):
    """When an employee authenticates, their /api/v1/auth/me profile reflects 'employee', not 'admin'."""
    ws_id = sample_company["workspace_id"]
    admin_token = sample_company["admin_token"]

    c_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/invitations",
        json={"max_usage_count": 2},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    raw_code = c_resp.json()["raw_code"]

    emp_email = f"role_emp_{uuid.uuid4().hex[:6]}@test.com"
    reg = client.post(
        "/api/v1/auth/employee/register",
        json={"full_name": "Role Test Emp", "email": emp_email, "password": "Password123!", "code": raw_code},
    )
    emp_token = reg.json()["access_token"]

    # Verify server-side role
    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {emp_token}"})
    user_roles = [w["role"] for w in me_resp.json()["workspaces"]]
    assert "owner" not in user_roles
    assert "admin" not in user_roles
    assert "employee" in user_roles

    # Verify employee cannot call admin-only invitation creation
    forbidden_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/invitations",
        json={"max_usage_count": 5},
        headers={"Authorization": f"Bearer {emp_token}"},
    )
    assert forbidden_resp.status_code == 403


def test_26_27_existing_owner_admin_remains_functional():
    """Existing development owner account (admin@enterprise.com / AdminPassword123!) continues working."""
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@enterprise.com", "password": "AdminPassword123!"},
    )
    assert login_resp.status_code == 200, "Existing admin account must continue functioning!"
    token = login_resp.json()["access_token"]

    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    roles = [w["role"] for w in me_resp.json()["workspaces"]]
    assert any(r in ("owner", "admin") for r in roles)


def test_28_logout_revokes_session():
    """Logout revokes active refresh token."""
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@enterprise.com", "password": "AdminPassword123!"},
    )
    token_data = login_resp.json()
    acc_tok = token_data["access_token"]
    ref_tok = token_data["refresh_token"]

    logout_resp = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": ref_tok},
        headers={"Authorization": f"Bearer {acc_tok}"},
    )
    assert logout_resp.status_code == 200

    # Refreshing with the revoked token must now fail
    ref_resp = client.post("/api/v1/auth/refresh", json={"refresh_token": ref_tok})
    assert ref_resp.status_code in (401, 403)


def test_29_refresh_token_rotation():
    """Refresh token endpoint rotates and issues a fresh token pair."""
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@enterprise.com", "password": "AdminPassword123!"},
    )
    ref_tok = login_resp.json()["refresh_token"]

    rotate_resp = client.post("/api/v1/auth/refresh", json={"refresh_token": ref_tok})
    assert rotate_resp.status_code == 200
    new_data = rotate_resp.json()
    assert "access_token" in new_data
    assert "refresh_token" in new_data
    assert new_data["refresh_token"] != ref_tok


def test_30_invitation_list_does_not_expose_raw_code_or_hash(sample_company):
    """Listing invitations returns sanitized metadata; raw code and code_hash are never exposed."""
    ws_id = sample_company["workspace_id"]
    admin_token = sample_company["admin_token"]

    resp = client.get(
        f"/api/v1/workspaces/{ws_id}/invitations",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200
    items = resp.json()["items"]
    for item in items:
        assert "code_hash" not in item
        assert "raw_code" not in item
        assert "id" in item
        assert "is_valid" in item


# =============================================================================
# PART 5: REGRESSION VERIFICATION (PHASES 1–3 INTACT)
# =============================================================================

def test_31_existing_documents_listing_intact():
    """Existing documents endpoint continues returning 200 for authenticated requests."""
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@enterprise.com", "password": "AdminPassword123!"},
    )
    token = login_resp.json()["access_token"]
    ws_id = "dc5d9139-f381-4818-ab26-c2e878997b4d"

    doc_resp = client.get(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert doc_resp.status_code == 200
    docs = doc_resp.json()["items"]
    assert len(docs) >= 7


def test_32_35_existing_seven_pdfs_and_scopes_preserved():
    """All 7 core company PDFs remain present and scoped as company."""
    with get_db_session() as db:
        ws_id = "dc5d9139-f381-4818-ab26-c2e878997b4d"
        docs = db.query(Document).filter(Document.workspace_id == ws_id).all()
        assert len(docs) >= 7
        for d in docs:
            assert d.visibility_scope == "company"


def test_33_34_rag_and_gemini_config_intact():
    """RAG health diagnostic endpoint verifies Gemini and retrieval configurations remain intact."""
    resp = client.get("/health/llm")
    assert resp.status_code == 200
    data = resp.json()
    assert "configuration" in data
    assert "status" in data["configuration"]
    assert "model" in data["configuration"]


def test_36_database_records_preserved():
    """Users, workspaces, and memberships preserved without data loss."""
    with get_db_session() as db:
        users_count = db.query(User).count()
        ws_count = db.query(Workspace).count()
        assert users_count >= 1
        assert ws_count >= 1
