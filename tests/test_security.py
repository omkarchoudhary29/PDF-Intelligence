"""Comprehensive Security, Authentication, RBAC, and Multi-Tenant Isolation Test Suite.

Phase 10 Enterprise Security Validation:
- Argon2id password hashing and verification
- Enterprise password strength validation (12+ chars, mixed cases, digits, symbols)
- JWT access tokens (short-lived, signature verification, expiry enforcement)
- Database-backed rotating refresh tokens and revocation / reuse compromise detection
- Account lockout after 5 consecutive failed attempts (423 Locked)
- Role-Based Access Control (RBAC):
  * Owner: Full workspace management, member role updates, report deletion, workspace deletion
  * Admin: Document uploads/deletions, report generation, employee membership management
  * Employee: Document viewing/downloading, search, chat, report viewing/downloading
- Protection of sole owner (preventing workspace orphan state)
- Strict Multi-Tenant Isolation (Cross-tenant access blocked with 404 to avoid leaking resource existence)
- Private conversation creator isolation
- CLI administrative user provisioning
"""

from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.security import (
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_password,
    hash_token,
    validate_password_strength,
    verify_password,
)
from backend.database.base import Base
from backend.database.connection import get_db, set_test_engine
from backend.database.models import (
    Conversation,
    Document,
    Message,
    RefreshSession,
    Report,
    User,
    Workspace,
    WorkspaceMembership,
)
from backend.main import app
from backend.services.auth_service import auth_service
from config.settings import get_settings


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def test_db():
    """Create an isolated in-memory SQLite database for security tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    set_test_engine(engine)

    with session_factory() as session:
        yield session

    set_test_engine(None)
    engine.dispose()


@pytest.fixture
def client(test_db):
    """FastAPI TestClient with isolated DB and NO auth bypasses."""
    # Ensure any conftest legacy auth overrides are completely cleared
    app.dependency_overrides.clear()

    def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# Helper factory functions
def create_user(
    db: Session,
    email: str = "user@enterprise.com",
    password: str = "StrongPass123!#",
    full_name: str = "Enterprise User",
    is_active: bool = True,
    must_change_password: bool = False,
) -> User:
    pwd_hash = hash_password(password)
    user = User(
        email=email,
        full_name=full_name,
        password_hash=pwd_hash,
        is_active=is_active,
        must_change_password=must_change_password,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_workspace(db: Session, name: str = "Alpha Corp", slug: str = "alpha-corp") -> Workspace:
    ws = Workspace(name=name, slug=slug, is_active=True)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def add_membership(
    db: Session,
    user_id: str,
    workspace_id: str,
    role: str = "employee",
    status: str = "active",
) -> WorkspaceMembership:
    mem = WorkspaceMembership(
        user_id=user_id,
        workspace_id=workspace_id,
        role=role,
        membership_status=status,
    )
    db.add(mem)
    db.commit()
    db.refresh(mem)
    return mem


def get_auth_header(client: TestClient, email: str, password: str) -> dict:
    resp = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# =============================================================================
# 1. Password Hashing and Token Core Security Tests
# =============================================================================

def test_argon2id_password_hashing():
    """Verify Argon2id hashes are salted, non-reversible, and verifiable."""
    pwd = "EnterpriseSecret2026!#"
    hash1 = hash_password(pwd)
    hash2 = hash_password(pwd)

    # Salt uniqueness
    assert hash1 != hash2
    assert hash1.startswith("$argon2id$")

    # Correct password succeeds
    assert verify_password(pwd, hash1) is True
    # Incorrect password fails
    assert verify_password("WrongPassword123!", hash1) is False


def test_password_complexity_policy():
    """Verify password policy enforces length, mixed-case, digits, and special characters."""
    # Valid
    assert validate_password_strength("ValidPass123!#")[0] is True

    # Too short (< 12)
    valid, msg = validate_password_strength("Short1!Aa")
    assert valid is False
    assert "at least 12 characters" in msg

    # Missing uppercase
    valid, msg = validate_password_strength("lowercaseonly123!#")
    assert valid is False
    assert "uppercase" in msg

    # Missing lowercase
    valid, msg = validate_password_strength("UPPERCASEONLY123!#")
    assert valid is False
    assert "lowercase" in msg

    # Missing digit
    valid, msg = validate_password_strength("NoDigitsHereAtAll!#")
    assert valid is False
    assert ("number" in msg.lower() or "digit" in msg.lower())

    # Missing special char
    valid, msg = validate_password_strength("NoSpecialChar12345")
    assert valid is False
    assert "special character" in msg


def test_jwt_access_token_creation_and_expiration():
    """Verify JWT access token claims, signing, and expiration decoding."""
    token = create_access_token(
        subject="user-123",
        email="test@enterprise.com",
        expires_delta=timedelta(minutes=15),
    )
    payload = decode_access_token(token)
    assert payload["sub"] == "user-123"
    assert payload["email"] == "test@enterprise.com"
    assert payload["type"] == "access"

    # Expired token
    expired_token = create_access_token(
        subject="user-123",
        email="test@enterprise.com",
        expires_delta=timedelta(seconds=-1),
    )
    with pytest.raises(Exception) as exc:
        decode_access_token(expired_token)
    assert "expired" in str(exc.value).lower()


def test_refresh_token_generation_and_hashing():
    """Verify cryptographically secure refresh token generation and hashing."""
    raw_token = generate_refresh_token()
    assert len(raw_token) >= 64
    thash = hash_token(raw_token)
    assert len(thash) == 64
    assert thash == hash_token(raw_token)


# =============================================================================
# 2. Authentication API Endpoints
# =============================================================================

def test_login_success(client, test_db):
    """Verify successful login returns short-lived access token and database-backed refresh token."""
    user = create_user(test_db, email="alice@corp.com", password="SecurePassword123!#")
    resp = client.post(
        "/api/v1/auth/login",
        json={"email": "alice@corp.com", "password": "SecurePassword123!#"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "alice@corp.com"

    # Verify session recorded in DB
    session = test_db.query(RefreshSession).filter(RefreshSession.user_id == user.id).first()
    assert session is not None
    assert session.revoked_at is None


def test_login_invalid_password_returns_generic_error(client, test_db):
    """Verify login failure returns generic message without disclosing credential details."""
    create_user(test_db, email="bob@corp.com", password="SecurePassword123!#")
    resp = client.post(
        "/api/v1/auth/login",
        json={"email": "bob@corp.com", "password": "WrongPassword123!#"},
    )
    assert resp.status_code == 401
    assert "Invalid email or password" in resp.json()["detail"]


def test_login_nonexistent_user_returns_generic_error(client, test_db):
    """Verify nonexistent user login returns identical generic 401 error message."""
    resp = client.post(
        "/api/v1/auth/login",
        json={"email": "ghost@corp.com", "password": "AnyPassword123!#"},
    )
    assert resp.status_code == 401
    assert "Invalid email or password" in resp.json()["detail"]


def test_login_inactive_user_forbidden(client, test_db):
    """Verify deactivated user cannot log in."""
    create_user(test_db, email="inactive@corp.com", password="SecurePassword123!#", is_active=False)
    resp = client.post(
        "/api/v1/auth/login",
        json={"email": "inactive@corp.com", "password": "SecurePassword123!#"},
    )
    assert resp.status_code == 403
    detail = resp.json()["detail"].lower()
    assert "deactivated" in detail or "inactive" in detail


def test_account_lockout_after_five_failed_attempts(client, test_db):
    """Verify user is locked out for 15 minutes after 5 consecutive failed attempts (423 Locked)."""
    user = create_user(test_db, email="victim@corp.com", password="CorrectPassword123!#")

    for i in range(5):
        resp = client.post(
            "/api/v1/auth/login",
            json={"email": "victim@corp.com", "password": "BadPassword123!#"},
        )
        assert resp.status_code == 401

    # 6th attempt should be locked out even with correct password
    locked_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "victim@corp.com", "password": "CorrectPassword123!#"},
    )
    assert locked_resp.status_code == 423
    assert "locked" in locked_resp.json()["detail"].lower()


def test_refresh_token_rotation(client, test_db):
    """Verify refresh token rotation issues a new token and revokes the old one."""
    create_user(test_db, email="charlie@corp.com", password="SecurePassword123!#")
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "charlie@corp.com", "password": "SecurePassword123!#"},
    )
    orig_refresh = login_resp.json()["refresh_token"]

    # Rotate refresh token
    refresh_resp = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": orig_refresh},
    )
    assert refresh_resp.status_code == 200
    new_refresh = refresh_resp.json()["refresh_token"]
    assert new_refresh != orig_refresh

    # Original token is now revoked; replaying it fails
    replay_resp = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": orig_refresh},
    )
    assert replay_resp.status_code == 401


def test_revoked_refresh_token_reuse_compromise_detection(client, test_db):
    """Verify that replaying a revoked refresh token triggers compromise detection revoking all sessions."""
    user = create_user(test_db, email="dan@corp.com", password="SecurePassword123!#")
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "dan@corp.com", "password": "SecurePassword123!#"},
    )
    token_1 = login_resp.json()["refresh_token"]

    # Rotate token_1 -> token_2
    ref_resp = client.post("/api/v1/auth/refresh", json={"refresh_token": token_1})
    token_2 = ref_resp.json()["refresh_token"]

    # Replay revoked token_1 (Compromise attempt)
    client.post("/api/v1/auth/refresh", json={"refresh_token": token_1})

    # Token 2 should now also be invalidated because all sessions were revoked
    ref2_resp = client.post("/api/v1/auth/refresh", json={"refresh_token": token_2})
    assert ref2_resp.status_code == 401


def test_logout_revokes_session(client, test_db):
    """Verify logout revokes the refresh session."""
    user = create_user(test_db, email="elena@corp.com", password="SecurePassword123!#")
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "elena@corp.com", "password": "SecurePassword123!#"},
    )
    r_token = login_resp.json()["refresh_token"]
    headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}

    logout_resp = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": r_token},
        headers=headers,
    )
    assert logout_resp.status_code == 200

    # Refreshing with logged out token fails
    assert client.post("/api/v1/auth/refresh", json={"refresh_token": r_token}).status_code == 401


def test_me_endpoint_requires_auth(client, test_db):
    """Verify /api/v1/auth/me requires valid bearer token and returns profile."""
    # Unauthenticated
    assert client.get("/api/v1/auth/me").status_code == 401

    # Authenticated
    create_user(test_db, email="frank@corp.com", password="SecurePassword123!#", full_name="Frank Underwood")
    headers = get_auth_header(client, "frank@corp.com", "SecurePassword123!#")
    resp = client.get("/api/v1/auth/me", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["email"] == "frank@corp.com"
    assert resp.json()["full_name"] == "Frank Underwood"


def test_change_password(client, test_db):
    """Verify user can change their password and authenticate with the new password."""
    create_user(test_db, email="grace@corp.com", password="OldPassword123!#")
    headers = get_auth_header(client, "grace@corp.com", "OldPassword123!#")

    # Change password
    change_resp = client.post(
        "/api/v1/auth/change-password",
        json={"current_password": "OldPassword123!#", "new_password": "NewStrongPass456!#"},
        headers=headers,
    )
    assert change_resp.status_code == 200

    # Old password no longer works
    assert client.post("/api/v1/auth/login", json={"email": "grace@corp.com", "password": "OldPassword123!#"}).status_code == 401

    # New password works
    assert client.post("/api/v1/auth/login", json={"email": "grace@corp.com", "password": "NewStrongPass456!#"}).status_code == 200


def test_user_self_registration(client, test_db):
    """Verify self-registration creates user, provisions workspace, and issues access token."""
    resp = client.post(
        "/api/v1/auth/register",
        json={
            "full_name": "Alexander Hamilton",
            "email": "alexander@corp.com",
            "password": "HamiltonPassword123!",
            "workspace_name": "Treasury Division",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["user"]["email"] == "alexander@corp.com"
    assert "access_token" in data

    # Duplicate registration fails
    dup_resp = client.post(
        "/api/v1/auth/register",
        json={
            "full_name": "Alexander Hamilton",
            "email": "alexander@corp.com",
            "password": "HamiltonPassword123!",
        },
    )
    assert dup_resp.status_code == 400


# =============================================================================
# 3. RBAC & Workspace Authorization Tests
# =============================================================================

@pytest.fixture
def rbac_setup(test_db):
    """Setup workspace with Owner, Admin, and Employee users."""
    ws = create_workspace(test_db, name="Acme Corp", slug="acme-corp")

    owner = create_user(test_db, email="owner@acme.com", password="OwnerPass123!#", full_name="Alice Owner")
    admin = create_user(test_db, email="admin@acme.com", password="AdminPass123!#", full_name="Bob Admin")
    employee = create_user(test_db, email="emp@acme.com", password="EmployeePass123!#", full_name="Charlie Emp")

    add_membership(test_db, owner.id, ws.id, role="owner")
    add_membership(test_db, admin.id, ws.id, role="admin")
    add_membership(test_db, employee.id, ws.id, role="employee")

    return {
        "workspace": ws,
        "owner": owner,
        "admin": admin,
        "employee": employee,
    }


def test_owner_can_manage_members(client, rbac_setup):
    """Verify Owner can list, add, update, and remove workspace members."""
    ws = rbac_setup["workspace"]
    headers = get_auth_header(client, "owner@acme.com", "OwnerPass123!#")

    # List members
    list_resp = client.get(f"/api/v1/workspaces/{ws.id}/members", headers=headers)
    assert list_resp.status_code == 200
    assert len(list_resp.json()["items"]) == 3

    # Add member
    add_resp = client.post(
        f"/api/v1/workspaces/{ws.id}/members",
        json={"email": "newbie@acme.com", "role": "employee", "full_name": "New Employee"},
        headers=headers,
    )
    assert add_resp.status_code == 201
    newbie_user_id = add_resp.json()["user_id"]

    # Update member role to admin
    up_resp = client.patch(
        f"/api/v1/workspaces/{ws.id}/members/{newbie_user_id}",
        json={"role": "admin"},
        headers=headers,
    )
    assert up_resp.status_code == 200
    assert up_resp.json()["role"] == "admin"

    # Remove member
    del_resp = client.delete(f"/api/v1/workspaces/{ws.id}/members/{newbie_user_id}", headers=headers)
    assert del_resp.status_code == 200


def test_admin_member_management_permissions(client, rbac_setup):
    """Verify Admin can manage employees, but cannot modify/remove owners or promote to owner."""
    ws = rbac_setup["workspace"]
    owner = rbac_setup["owner"]
    headers = get_auth_header(client, "admin@acme.com", "AdminPass123!#")

    # Admin can list members
    assert client.get(f"/api/v1/workspaces/{ws.id}/members", headers=headers).status_code == 200

    # Admin can add employee
    add_emp = client.post(
        f"/api/v1/workspaces/{ws.id}/members",
        json={"email": "dev1@acme.com", "role": "employee"},
        headers=headers,
    )
    assert add_emp.status_code == 201

    # Admin CANNOT add an owner
    add_owner_fail = client.post(
        f"/api/v1/workspaces/{ws.id}/members",
        json={"email": "hacker@acme.com", "role": "owner"},
        headers=headers,
    )
    assert add_owner_fail.status_code == 403

    # Admin CANNOT demote or remove Owner
    demote_fail = client.patch(
        f"/api/v1/workspaces/{ws.id}/members/{owner.id}",
        json={"role": "employee"},
        headers=headers,
    )
    assert demote_fail.status_code == 403

    remove_owner_fail = client.delete(
        f"/api/v1/workspaces/{ws.id}/members/{owner.id}",
        headers=headers,
    )
    assert remove_owner_fail.status_code == 403


def test_employee_cannot_manage_members(client, rbac_setup):
    """Verify Employee cannot add, update, or remove workspace members."""
    ws = rbac_setup["workspace"]
    headers = get_auth_header(client, "emp@acme.com", "EmployeePass123!#")

    # Employee cannot add members
    resp_add = client.post(
        f"/api/v1/workspaces/{ws.id}/members",
        json={"email": "friend@acme.com", "role": "employee"},
        headers=headers,
    )
    assert resp_add.status_code == 403

    # Employee cannot delete members
    resp_del = client.delete(
        f"/api/v1/workspaces/{ws.id}/members/{rbac_setup['admin'].id}",
        headers=headers,
    )
    assert resp_del.status_code == 403


def test_protection_of_final_owner(client, rbac_setup):
    """Verify final owner protection prevents demoting or removing the sole owner."""
    ws = rbac_setup["workspace"]
    owner = rbac_setup["owner"]
    headers = get_auth_header(client, "owner@acme.com", "OwnerPass123!#")

    # Attempt to demote self
    demote_self = client.patch(
        f"/api/v1/workspaces/{ws.id}/members/{owner.id}",
        json={"role": "employee"},
        headers=headers,
    )
    assert demote_self.status_code in (400, 403)
    detail1 = demote_self.json()["detail"].lower()
    assert "final workspace owner" in detail1 or "sole active owner" in detail1

    # Attempt to delete self
    delete_self = client.delete(
        f"/api/v1/workspaces/{ws.id}/members/{owner.id}",
        headers=headers,
    )
    assert delete_self.status_code in (400, 403)
    detail2 = delete_self.json()["detail"].lower()
    assert "final workspace owner" in detail2 or "sole active owner" in detail2


def test_workspace_deletion_rbac(client, rbac_setup):
    """Verify only Owners can delete a workspace; Admin and Employee get 403."""
    ws = rbac_setup["workspace"]
    emp_headers = get_auth_header(client, "emp@acme.com", "EmployeePass123!#")
    admin_headers = get_auth_header(client, "admin@acme.com", "AdminPass123!#")
    owner_headers = get_auth_header(client, "owner@acme.com", "OwnerPass123!#")

    # Employee cannot delete workspace
    assert client.delete(f"/api/v1/workspaces/{ws.id}", headers=emp_headers).status_code == 403

    # Admin cannot delete workspace
    assert client.delete(f"/api/v1/workspaces/{ws.id}", headers=admin_headers).status_code == 403

    # Owner can delete workspace
    assert client.delete(f"/api/v1/workspaces/{ws.id}", headers=owner_headers).status_code == 200


def test_document_lifecycle_rbac(client, rbac_setup, test_db, tmp_path, monkeypatch):
    """Verify Admin and Owner can upload/delete docs; Employee is restricted to viewing/downloading."""
    from backend.services.storage_service import storage_service
    s_root = tmp_path / "sec_storage"
    monkeypatch.setattr(storage_service, "storage_root", s_root)
    monkeypatch.setattr(storage_service, "upload_root", s_root / "raw")
    monkeypatch.setattr(storage_service, "temp_root", s_root / "temp")
    (s_root / "raw").mkdir(parents=True, exist_ok=True)
    (s_root / "temp").mkdir(parents=True, exist_ok=True)

    ws = rbac_setup["workspace"]
    emp_headers = get_auth_header(client, "emp@acme.com", "EmployeePass123!#")
    admin_headers = get_auth_header(client, "admin@acme.com", "AdminPass123!#")

    # Create dummy PDF bytes
    pdf_data = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"
    files = [("files", ("contract.pdf", pdf_data, "application/pdf"))]

    # Employee cannot upload
    emp_upload = client.post(
        f"/api/v1/workspaces/{ws.id}/documents",
        files=files,
        headers=emp_headers,
    )
    assert emp_upload.status_code == 403

    # Admin can upload
    admin_upload = client.post(
        f"/api/v1/workspaces/{ws.id}/documents",
        files=files,
        headers=admin_headers,
    )
    assert admin_upload.status_code == 200
    item = admin_upload.json()["results"][0]
    assert item["status"] == "uploaded"
    doc_id = item["document"]["id"]

    # Employee CAN view/list documents
    assert client.get(f"/api/v1/workspaces/{ws.id}/documents", headers=emp_headers).status_code == 200

    # Employee CANNOT delete document
    assert client.delete(f"/api/v1/workspaces/{ws.id}/documents/{doc_id}", headers=emp_headers).status_code == 403

    # Admin CAN delete document
    assert client.delete(f"/api/v1/workspaces/{ws.id}/documents/{doc_id}", headers=admin_headers).status_code == 200


def test_report_lifecycle_rbac(client, rbac_setup, test_db):
    """Verify Admin & Owner can create reports; Employee cannot create; only Owner can delete reports."""
    ws = rbac_setup["workspace"]
    emp_headers = get_auth_header(client, "emp@acme.com", "EmployeePass123!#")
    admin_headers = get_auth_header(client, "admin@acme.com", "AdminPass123!#")
    owner_headers = get_auth_header(client, "owner@acme.com", "OwnerPass123!#")

    # Employee cannot create report
    emp_rep = client.post(
        f"/api/v1/workspaces/{ws.id}/reports",
        json={"include_graph": False},
        headers=emp_headers,
    )
    assert emp_rep.status_code == 403

    # Admin can create report (mock execution)
    with patch("backend.services.report_generation_service.report_generation_service.execute_report_generation") as mock_exec:
        mock_rep = Report(
            workspace_id=ws.id,
            report_name="Admin Dossier",
            status="completed",
            storage_key="reports/dummy.pdf",
            file_size_bytes=1000,
        )
        test_db.add(mock_rep)
        test_db.commit()
        test_db.refresh(mock_rep)
        mock_exec.return_value = mock_rep

        admin_rep = client.post(
            f"/api/v1/workspaces/{ws.id}/reports",
            json={"include_graph": False},
            headers=admin_headers,
        )
        assert admin_rep.status_code == 201
        report_id = admin_rep.json()["id"]

    # Employee CAN view reports
    assert client.get(f"/api/v1/workspaces/{ws.id}/reports", headers=emp_headers).status_code == 200

    # Employee CANNOT delete report
    assert client.delete(f"/api/v1/workspaces/{ws.id}/reports/{report_id}", headers=emp_headers).status_code == 403

    # Admin CANNOT delete shared enterprise report
    assert client.delete(f"/api/v1/workspaces/{ws.id}/reports/{report_id}", headers=admin_headers).status_code == 403

    # Owner CAN delete report
    with patch("backend.services.report_generation_service.report_generation_service.delete_report", return_value=True):
        assert client.delete(f"/api/v1/workspaces/{ws.id}/reports/{report_id}", headers=owner_headers).status_code == 200


# =============================================================================
# 4. Strict Multi-Tenant Isolation Tests
# =============================================================================

@pytest.fixture
def multi_tenant_setup(test_db):
    """Setup two isolated tenant workspaces: Company A and Company B."""
    ws_a = create_workspace(test_db, name="Company A", slug="company-a")
    ws_b = create_workspace(test_db, name="Company B", slug="company-b")

    user_a = create_user(test_db, email="alice@companya.com", password="Password123!#", full_name="Alice A")
    user_b = create_user(test_db, email="bob@companyb.com", password="Password123!#", full_name="Bob B")

    add_membership(test_db, user_a.id, ws_a.id, role="owner")
    add_membership(test_db, user_b.id, ws_b.id, role="owner")

    # Add a document in Company B
    doc_b = Document(
        workspace_id=ws_b.id,
        original_filename="secret_b.pdf",
        display_name="secret_b.pdf",
        storage_key=f"workspaces/{ws_b.id}/documents/secret_b.pdf",
        file_size_bytes=1024,
        sha256_hash="hash_b_12345",
        mime_type="application/pdf",
        processing_status="completed",
        total_pages=5,
    )
    test_db.add(doc_b)

    # Add a conversation in Company B
    conv_b = Conversation(
        workspace_id=ws_b.id,
        user_id=user_b.id,
        title="Company B Strategy",
    )
    test_db.add(conv_b)

    # Add a report in Company B
    rep_b = Report(
        workspace_id=ws_b.id,
        report_name="Company B Dossier",
        status="completed",
        storage_key=f"workspaces/{ws_b.id}/reports/dossier.pdf",
        file_size_bytes=2048,
    )
    test_db.add(rep_b)
    test_db.commit()

    return {
        "ws_a": ws_a,
        "ws_b": ws_b,
        "user_a": user_a,
        "user_b": user_b,
        "doc_b": doc_b,
        "conv_b": conv_b,
        "rep_b": rep_b,
    }


def test_tenant_isolation_workspaces_listing(client, multi_tenant_setup):
    """Verify User A only sees Company A and never sees Company B in workspace lists."""
    headers_a = get_auth_header(client, "alice@companya.com", "Password123!#")
    resp = client.get("/api/v1/workspaces", headers=headers_a)
    assert resp.status_code == 200
    workspaces = resp.json()
    assert len(workspaces) == 1
    assert workspaces[0]["id"] == multi_tenant_setup["ws_a"].id
    assert all(w["id"] != multi_tenant_setup["ws_b"].id for w in workspaces)


def test_tenant_isolation_direct_workspace_access_returns_404(client, multi_tenant_setup):
    """Verify attempting to access Company B returns 404 rather than 403 to prevent leaking tenant existence."""
    headers_a = get_auth_header(client, "alice@companya.com", "Password123!#")
    ws_b_id = multi_tenant_setup["ws_b"].id
    resp = client.get(f"/api/v1/workspaces/{ws_b_id}", headers=headers_a)
    assert resp.status_code == 404


def test_tenant_isolation_document_access_returns_404(client, multi_tenant_setup):
    """Verify User A cannot list, download, preview, or delete Company B documents."""
    headers_a = get_auth_header(client, "alice@companya.com", "Password123!#")
    ws_b_id = multi_tenant_setup["ws_b"].id
    doc_b_id = multi_tenant_setup["doc_b"].id

    # List docs in B
    assert client.get(f"/api/v1/workspaces/{ws_b_id}/documents", headers=headers_a).status_code == 404

    # Download doc in B
    assert client.get(f"/api/v1/workspaces/{ws_b_id}/documents/{doc_b_id}/download", headers=headers_a).status_code == 404

    # Preview page in B
    assert client.get(f"/api/v1/workspaces/{ws_b_id}/documents/{doc_b_id}/pages/1/preview", headers=headers_a).status_code == 404

    # Delete doc in B
    assert client.delete(f"/api/v1/workspaces/{ws_b_id}/documents/{doc_b_id}", headers=headers_a).status_code == 404


def test_tenant_isolation_search_returns_404(client, multi_tenant_setup):
    """Verify User A cannot execute search against Company B's workspace."""
    headers_a = get_auth_header(client, "alice@companya.com", "Password123!#")
    ws_b_id = multi_tenant_setup["ws_b"].id

    resp = client.post(
        f"/api/v1/workspaces/{ws_b_id}/search",
        json={"query": "financial confidential secrets"},
        headers=headers_a,
    )
    assert resp.status_code == 404


def test_tenant_isolation_conversations_returns_404(client, multi_tenant_setup):
    """Verify User A cannot access Company B's conversations."""
    headers_a = get_auth_header(client, "alice@companya.com", "Password123!#")
    ws_b_id = multi_tenant_setup["ws_b"].id
    conv_b_id = multi_tenant_setup["conv_b"].id

    assert client.get(f"/api/v1/workspaces/{ws_b_id}/conversations", headers=headers_a).status_code == 404
    assert client.get(f"/api/v1/workspaces/{ws_b_id}/conversations/{conv_b_id}", headers=headers_a).status_code == 404


def test_tenant_isolation_reports_returns_404(client, multi_tenant_setup):
    """Verify User A cannot view, generate, or download Company B's reports."""
    headers_a = get_auth_header(client, "alice@companya.com", "Password123!#")
    ws_b_id = multi_tenant_setup["ws_b"].id
    rep_b_id = multi_tenant_setup["rep_b"].id

    assert client.get(f"/api/v1/workspaces/{ws_b_id}/reports", headers=headers_a).status_code == 404
    assert client.get(f"/api/v1/workspaces/{ws_b_id}/reports/{rep_b_id}/download", headers=headers_a).status_code == 404


def test_unclaimed_workspace_inaccessible(client, test_db):
    """Verify an unowned/unclaimed workspace returns 404 to all users."""
    orphan_ws = create_workspace(test_db, name="Orphan Workspace", slug="orphan-ws")
    user = create_user(test_db, email="lonely@corp.com", password="Password123!#")
    headers = get_auth_header(client, "lonely@corp.com", "Password123!#")

    resp = client.get(f"/api/v1/workspaces/{orphan_ws.id}", headers=headers)
    assert resp.status_code == 404


# =============================================================================
# 5. CLI Provisioning Tests
# =============================================================================

def test_cli_create_owner(test_db):
    """Verify CLI create-owner logic provisions active owner and assigns workspaces."""
    from backend.cli import cmd_create_owner
    assert callable(cmd_create_owner)
    ws = create_workspace(test_db, name="CLI Corp", slug="cli-corp")

    # Run create_owner logic
    user, count = auth_service.create_owner_user(
        db=test_db,
        email="cli_owner@corp.com",
        password="CliStrongPass123!#",
        full_name="CLI Owner",
        assign_existing_workspaces=True,
    )
    assert user.email == "cli_owner@corp.com"
    assert user.is_active is True
    assert verify_password("CliStrongPass123!#", user.password_hash) is True
    assert count >= 1

    # Check membership
    mem = (
        test_db.query(WorkspaceMembership)
        .filter(WorkspaceMembership.user_id == user.id, WorkspaceMembership.workspace_id == ws.id)
        .first()
    )
    assert mem is not None
    assert mem.role == "owner"


def test_private_conversation_isolation_within_workspace(client, test_db):
    """Verify conversations within the same workspace are private to their creator."""
    ws = create_workspace(test_db, name="Secret Projects", slug="secret-projects")
    user_alice = create_user(test_db, email="alice_priv@corp.com", password="Password123!#", full_name="Alice")
    user_bob = create_user(test_db, email="bob_priv@corp.com", password="Password123!#", full_name="Bob")

    add_membership(test_db, user_alice.id, ws.id, role="owner")
    add_membership(test_db, user_bob.id, ws.id, role="employee")

    headers_alice = get_auth_header(client, "alice_priv@corp.com", "Password123!#")
    headers_bob = get_auth_header(client, "bob_priv@corp.com", "Password123!#")

    # Alice creates a conversation
    create_resp = client.post(
        f"/api/v1/workspaces/{ws.id}/conversations",
        json={"title": "Alice's Secret Strategy"},
        headers=headers_alice,
    )
    assert create_resp.status_code == 201
    conv_id = create_resp.json()["id"]

    # Bob lists conversations in the same workspace -> Should NOT see Alice's conversation
    bob_list = client.get(f"/api/v1/workspaces/{ws.id}/conversations", headers=headers_bob)
    assert bob_list.status_code == 200
    assert len(bob_list.json()["items"]) == 0

    # Bob attempts to access Alice's conversation directly -> Returns 404
    bob_get = client.get(f"/api/v1/workspaces/{ws.id}/conversations/{conv_id}", headers=headers_bob)
    assert bob_get.status_code == 404

    # Bob attempts to send a message to Alice's conversation -> Returns 404
    bob_send = client.post(
        f"/api/v1/workspaces/{ws.id}/conversations/{conv_id}/messages",
        json={"content": "Can I read this?"},
        headers=headers_bob,
    )
    assert bob_send.status_code == 404


def test_first_login_mandatory_password_change(client, test_db):
    """Verify must_change_password flag is surfaced on login and cleared after change."""
    user = create_user(
        test_db,
        email="new_hire@corp.com",
        password="TempPassword123!#",
        must_change_password=True,
    )

    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "new_hire@corp.com", "password": "TempPassword123!#"},
    )
    assert login_resp.status_code == 200
    assert login_resp.json()["must_change_password"] is True
    headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}

    # Change password
    chg_resp = client.post(
        "/api/v1/auth/change-password",
        json={"current_password": "TempPassword123!#", "new_password": "PermanentPass123!#"},
        headers=headers,
    )
    assert chg_resp.status_code == 200

    # Verify profile now reflects must_change_password = False
    me_resp = client.get("/api/v1/auth/me", headers=headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["must_change_password"] is False


def test_tampered_token_rejected(client, test_db):
    """Verify tampered JWT token signatures are rejected with 401."""
    user = create_user(test_db, email="target@corp.com", password="Password123!#")
    headers = get_auth_header(client, "target@corp.com", "Password123!#")
    real_token = headers["Authorization"].split("Bearer ")[-1]

    # Tamper with the token string
    tampered_token = real_token[:-4] + "abcd"
    tampered_headers = {"Authorization": f"Bearer {tampered_token}"}

    resp = client.get("/api/v1/auth/me", headers=tampered_headers)
    assert resp.status_code == 401


def test_security_audit_logging_persisted(client, test_db):
    """Verify security actions persist audit logs in security_audit_logs table."""
    from backend.database.models import SecurityAuditLog
    initial_count = test_db.query(SecurityAuditLog).count()

    create_user(test_db, email="audited@corp.com", password="Password123!#")
    # Failed login
    client.post("/api/v1/auth/login", json={"email": "audited@corp.com", "password": "WrongPassword123!#"})
    # Successful login
    client.post("/api/v1/auth/login", json={"email": "audited@corp.com", "password": "Password123!#"})

    new_count = test_db.query(SecurityAuditLog).count()
    assert new_count > initial_count

    logs = test_db.query(SecurityAuditLog).order_by(SecurityAuditLog.created_at.desc()).all()
    actions = [l.action for l in logs]
    assert any("login" in a for a in actions)


def test_knowledge_graph_tenant_isolation(client, test_db):
    """Verify User A cannot query or build Knowledge Graph for Company B."""
    ws_b = create_workspace(test_db, name="Company B Graph", slug="comp-b-graph")
    user_a = create_user(test_db, email="user_a_graph@corp.com", password="Password123!#")
    ws_a = create_workspace(test_db, name="Company A Graph", slug="comp-a-graph")
    add_membership(test_db, user_a.id, ws_a.id, role="owner")

    headers_a = get_auth_header(client, "user_a_graph@corp.com", "Password123!#")

    # Access graph summary in B -> 404
    assert client.get(f"/api/v1/workspaces/{ws_b.id}/graph/summary", headers=headers_a).status_code == 404

    # Access graph data in B -> 404
    assert client.get(f"/api/v1/workspaces/{ws_b.id}/graph", headers=headers_a).status_code == 404

    # Trigger graph build in B -> 404
    assert client.post(f"/api/v1/workspaces/{ws_b.id}/graph/build", json={"force": True}, headers=headers_a).status_code == 404
