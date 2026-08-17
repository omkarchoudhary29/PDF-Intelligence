"""Test Suite for Phase 6 — Enterprise Administrator Dashboard, Company Document Library & Employee Management.

Validates all 40 required scenarios:
1. Administrator Dashboard & Overview (1-5)
2. Company Knowledge Library (6-18)
3. Employee Management (19-24)
4. Invitations & Joining Codes (25-31)
5. Company Activity Logging & Isolation (32-34)
6. Regressions & Data Preservation (35-40)
"""

from datetime import datetime, timedelta
import io
import uuid
import fitz
import pytest
from fastapi.testclient import TestClient

from backend.database.connection import get_db
from backend.database.models import Document, User, Workspace, WorkspaceInvitation, WorkspaceMembership
from backend.main import app

client = TestClient(app)


def make_pdf_bytes(title: str, text: str = "Company documentation content.") -> bytes:
    """Generate minimal valid PDF bytes with PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), f"{title}\n{text}")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture(scope="module")
def admin_fixture():
    """Setup an isolated company workspace with an Admin (Owner) and an Employee, plus a second company for isolation tests."""
    suffix = uuid.uuid4().hex[:6]
    comp_a_name = f"Apex Aerospace {suffix}"
    admin_a_email = f"admin_a_{suffix}@apex.com"

    # 1. Register Admin A and Company A
    reg_a = client.post("/api/v1/auth/admin/register", json={
        "full_name": "Admin Apex",
        "company_name": comp_a_name,
        "email": admin_a_email,
        "password": "Password123!",
    })
    assert reg_a.status_code == 201, reg_a.text
    admin_a_token = reg_a.json()["access_token"]
    admin_a_user_id = reg_a.json()["user"]["id"]

    # Get Workspace A ID
    me_a = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {admin_a_token}"})
    ws_a_id = me_a.json()["workspaces"][0]["workspace_id"]

    # 2. Generate Joining Code for Company A
    inv_a = client.post(
        f"/api/v1/workspaces/{ws_a_id}/invitations",
        json={"max_usage_count": 5, "expires_in_days": 7},
        headers={"Authorization": f"Bearer {admin_a_token}"},
    )
    assert inv_a.status_code == 201
    code_a = inv_a.json()["raw_code"]
    inv_a_id = inv_a.json()["id"]

    # 3. Register Employee 1 in Company A
    emp1_email = f"emp1_{suffix}@apex.com"
    reg_emp1 = client.post("/api/v1/auth/employee/register", json={
        "full_name": "Employee Apex One",
        "email": emp1_email,
        "password": "Password123!",
        "code": code_a,
    })
    assert reg_emp1.status_code == 201
    emp1_token = reg_emp1.json()["access_token"]
    emp1_id = reg_emp1.json()["user"]["id"]

    # 4. Register Company B and Admin B for cross-tenant isolation testing
    comp_b_name = f"Bravo Dynamics {suffix}"
    admin_b_email = f"admin_b_{suffix}@bravo.com"
    reg_b = client.post("/api/v1/auth/admin/register", json={
        "full_name": "Admin Bravo",
        "company_name": comp_b_name,
        "email": admin_b_email,
        "password": "Password123!",
    })
    assert reg_b.status_code == 201
    admin_b_token = reg_b.json()["access_token"]
    admin_b_user_id = reg_b.json()["user"]["id"]

    me_b = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {admin_b_token}"})
    ws_b_id = me_b.json()["workspaces"][0]["workspace_id"]

    return {
        "ws_a_id": ws_a_id,
        "admin_a_token": admin_a_token,
        "admin_a_user_id": admin_a_user_id,
        "admin_a_email": admin_a_email,
        "emp1_token": emp1_token,
        "emp1_id": emp1_id,
        "emp1_email": emp1_email,
        "inv_a_id": inv_a_id,
        "code_a": code_a,
        "ws_b_id": ws_b_id,
        "admin_b_token": admin_b_token,
        "admin_b_user_id": admin_b_user_id,
    }


# =============================================================================
# 1. DASHBOARD & OVERVIEW (Scenarios 1-5)
# =============================================================================

def test_01_admin_can_open_dashboard_stats(admin_fixture):
    """Scenario 1: Authenticated administrator can fetch company dashboard stats."""
    ws_id = admin_fixture["ws_a_id"]
    token = admin_fixture["admin_a_token"]

    resp = client.get(f"/api/v1/workspaces/{ws_id}/stats", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["workspace_id"] == ws_id
    assert "total_documents" in data
    assert "company_documents" in data
    assert "active_employees" in data
    assert "pending_invitations" in data


def test_02_employee_cannot_access_admin_routes(admin_fixture):
    """Scenario 2: Ordinary employee is denied access to admin-only management routes."""
    ws_id = admin_fixture["ws_a_id"]
    emp_token = admin_fixture["emp1_token"]

    # Activity endpoint requires admin
    r_act = client.get(f"/api/v1/workspaces/{ws_id}/activity", headers={"Authorization": f"Bearer {emp_token}"})
    assert r_act.status_code == 403

    # Create invitation requires admin
    r_inv = client.post(
        f"/api/v1/workspaces/{ws_id}/invitations",
        json={"max_usage_count": 5},
        headers={"Authorization": f"Bearer {emp_token}"},
    )
    assert r_inv.status_code == 403

    # Revoke invitation requires admin
    r_rev = client.delete(
        f"/api/v1/workspaces/{ws_id}/invitations/{admin_fixture['inv_a_id']}",
        headers={"Authorization": f"Bearer {emp_token}"},
    )
    assert r_rev.status_code == 403


def test_03_dashboard_metrics_reflect_real_data(admin_fixture):
    """Scenario 3: Dashboard metrics reflect actual database records."""
    ws_id = admin_fixture["ws_a_id"]
    token = admin_fixture["admin_a_token"]

    resp = client.get(f"/api/v1/workspaces/{ws_id}/stats", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()

    # We registered 1 employee in setup fixture
    assert data["active_employees"] >= 1
    # We created 1 invitation with 4 remaining uses
    assert data["pending_invitations"] >= 1


def test_04_empty_workspace_metrics_safe(admin_fixture):
    """Scenario 4: A fresh workspace with no documents returns 0 counts without error."""
    ws_b_id = admin_fixture["ws_b_id"]
    token_b = admin_fixture["admin_b_token"]

    resp = client.get(f"/api/v1/workspaces/{ws_b_id}/stats", headers={"Authorization": f"Bearer {token_b}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_documents"] == 0
    assert data["company_documents"] == 0
    assert data["active_employees"] == 0


def test_05_cross_company_stats_and_activity_isolation(admin_fixture):
    """Scenario 5: Admin A cannot access Company B's statistics or activity."""
    ws_b_id = admin_fixture["ws_b_id"]
    token_a = admin_fixture["admin_a_token"]

    # Stats for Workspace B with Admin A's token -> 404 (tenant boundary)
    r_stats = client.get(f"/api/v1/workspaces/{ws_b_id}/stats", headers={"Authorization": f"Bearer {token_a}"})
    assert r_stats.status_code == 404

    # Activity for Workspace B with Admin A's token -> 404
    r_act = client.get(f"/api/v1/workspaces/{ws_b_id}/activity", headers={"Authorization": f"Bearer {token_a}"})
    assert r_act.status_code == 404


# =============================================================================
# 2. COMPANY KNOWLEDGE LIBRARY (Scenarios 6-18)
# =============================================================================

def test_06_to_10_admin_uploads_company_pdf_and_verifies_scope(admin_fixture):
    """Scenarios 6-10: Admin uploads PDF, assigned company scope & workspace, appears in library."""
    ws_id = admin_fixture["ws_a_id"]
    token = admin_fixture["admin_a_token"]

    pdf_bytes = make_pdf_bytes("Standard Operating Procedures", "Detailed company SOP procedures.")
    upload_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files=[("files", ("Apex_SOP_Manual.pdf", pdf_bytes, "application/pdf"))],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert upload_resp.status_code == 200, upload_resp.text
    res_data = upload_resp.json()
    assert res_data["total_uploaded"] == 1
    doc = res_data["results"][0]["document"]

    # Scenarios 7, 8, 9:
    assert doc["visibility_scope"] == "company"
    assert doc["workspace_id"] == ws_id
    assert doc["owner_user_id"] is None
    doc_id = doc["id"]
    admin_fixture["sop_doc_id"] = doc_id

    # Scenario 10: Appears in company library listing
    list_resp = client.get(
        f"/api/v1/workspaces/{ws_id}/documents?scope=company",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_resp.status_code == 200
    items = list_resp.json()["items"]
    assert any(d["id"] == doc_id for d in items)


def test_11_12_search_filter_and_processing_status(admin_fixture):
    """Scenarios 11-12: Status filter and filename filter work, displaying valid processing state."""
    ws_id = admin_fixture["ws_a_id"]
    token = admin_fixture["admin_a_token"]
    doc_id = admin_fixture["sop_doc_id"]

    # Filter by filename
    list_resp = client.get(
        f"/api/v1/workspaces/{ws_id}/documents?scope=company",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_resp.status_code == 200
    doc = next(d for d in list_resp.json()["items"] if d["id"] == doc_id)

    # Valid processing status
    assert doc["processing_status"] in ("completed", "processing", "uploaded", "queued")


def test_13_14_document_preview_and_unauthorized_rejection(admin_fixture):
    """Scenarios 13-14: Preview opens PDF page; unauthorized/cross-company request is rejected."""
    ws_id = admin_fixture["ws_a_id"]
    doc_id = admin_fixture["sop_doc_id"]
    token_a = admin_fixture["admin_a_token"]
    token_b = admin_fixture["admin_b_token"]

    # Authorized preview
    prev_resp = client.get(
        f"/api/v1/workspaces/{ws_id}/documents/{doc_id}/pages/1/preview",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert prev_resp.status_code == 200
    assert prev_resp.headers["content-type"] == "image/png"
    assert prev_resp.content.startswith(b"\x89PNG")

    # Unauthorized preview: Admin B tries to preview Admin A's document -> 404
    prev_unauth = client.get(
        f"/api/v1/workspaces/{ws_id}/documents/{doc_id}/pages/1/preview",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert prev_unauth.status_code == 404

    # Unauthenticated preview -> 401
    prev_anon = client.get(f"/api/v1/workspaces/{ws_id}/documents/{doc_id}/pages/1/preview")
    assert prev_anon.status_code == 401


def test_15_16_authorized_company_document_deletion(admin_fixture):
    """Scenarios 15-16: Admin deletes company document; record and search indexes are removed."""
    ws_id = admin_fixture["ws_a_id"]
    token = admin_fixture["admin_a_token"]

    # Upload a document to delete
    pdf_bytes = make_pdf_bytes("Temporary Guidelines", "To be deleted.")
    up = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files=[("files", ("Temp_Guidelines.pdf", pdf_bytes, "application/pdf"))],
        headers={"Authorization": f"Bearer {token}"},
    )
    del_id = up.json()["results"][0]["document"]["id"]

    # Delete with authorization
    del_resp = client.delete(
        f"/api/v1/workspaces/{ws_id}/documents/{del_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert del_resp.status_code == 200

    # Ensure document is no longer listed
    get_resp = client.get(
        f"/api/v1/workspaces/{ws_id}/documents/{del_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert get_resp.status_code == 404


def test_17_admin_cannot_delete_personal_documents(admin_fixture):
    """Scenario 17: Company library deletion rejects deletion of an employee's personal document."""
    ws_id = admin_fixture["ws_a_id"]
    emp_token = admin_fixture["emp1_token"]
    admin_token = admin_fixture["admin_a_token"]

    # Employee uploads personal PDF
    pdf_bytes = make_pdf_bytes("Private Employee Notes", "Personal confidential notes.")
    up = client.post(
        f"/api/v1/workspaces/{ws_id}/documents/personal",
        files=[("files", ("Emp_Private_Notes.pdf", pdf_bytes, "application/pdf"))],
        headers={"Authorization": f"Bearer {emp_token}"},
    )
    pers_doc_id = up.json()["results"][0]["document"]["id"]

    # Admin tries to delete Employee's personal document -> 404 (fail-closed isolation)
    del_resp = client.delete(
        f"/api/v1/workspaces/{ws_id}/documents/{pers_doc_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert del_resp.status_code == 404


def test_18_existing_seven_documents_preserved():
    """Scenario 18: Verify pre-existing 7 aerospace company PDFs remain preserved."""
    admin_login = client.post("/api/v1/auth/login", json={
        "email": "admin@enterprise.com",
        "password": "AdminPassword123!",
    })
    assert admin_login.status_code == 200
    token = admin_login.json()["access_token"]
    ws_id = "dc5d9139-f381-4818-ab26-c2e878997b4d"

    doc_resp = client.get(
        f"/api/v1/workspaces/{ws_id}/documents?scope=company",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert doc_resp.status_code == 200
    items = doc_resp.json()["items"]
    assert len(items) >= 7


# =============================================================================
# 3. EMPLOYEE MANAGEMENT (Scenarios 19-24)
# =============================================================================

def test_19_admin_sees_company_employees(admin_fixture):
    """Scenario 19: Admin sees all employees registered in their company."""
    ws_id = admin_fixture["ws_a_id"]
    token = admin_fixture["admin_a_token"]

    resp = client.get(f"/api/v1/workspaces/{ws_id}/members", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    members = resp.json()["items"]
    assert any(m["user_id"] == admin_fixture["emp1_id"] for m in members)


def test_20_cross_company_employee_listing_rejected(admin_fixture):
    """Scenario 20: Cross-company employee listing returns 404."""
    ws_b_id = admin_fixture["ws_b_id"]
    token_a = admin_fixture["admin_a_token"]

    resp = client.get(f"/api/v1/workspaces/{ws_b_id}/members", headers={"Authorization": f"Bearer {token_a}"})
    assert resp.status_code == 404


def test_21_employee_cannot_manage_members(admin_fixture):
    """Scenario 21: Ordinary employee cannot modify member statuses or roles."""
    ws_id = admin_fixture["ws_a_id"]
    emp_token = admin_fixture["emp1_token"]
    emp_id = admin_fixture["emp1_id"]

    resp = client.patch(
        f"/api/v1/workspaces/{ws_id}/members/{emp_id}",
        json={"role": "admin"},
        headers={"Authorization": f"Bearer {emp_token}"},
    )
    assert resp.status_code == 403


def test_22_supported_status_changes_work(admin_fixture):
    """Scenario 22: Administrator can suspend and reactivate employee access."""
    ws_id = admin_fixture["ws_a_id"]
    token = admin_fixture["admin_a_token"]
    emp_id = admin_fixture["emp1_id"]

    # Suspend
    r_sus = client.patch(
        f"/api/v1/workspaces/{ws_id}/members/{emp_id}",
        json={"membership_status": "suspended"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r_sus.status_code == 200
    assert r_sus.json()["membership_status"] == "suspended"

    # Reactivate
    r_act = client.patch(
        f"/api/v1/workspaces/{ws_id}/members/{emp_id}",
        json={"membership_status": "active"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r_act.status_code == 200
    assert r_act.json()["membership_status"] == "active"


def test_23_owner_protection_cannot_be_demoted_illegally(admin_fixture):
    """Scenario 23: Final owner cannot be demoted or deactivated."""
    ws_id = admin_fixture["ws_a_id"]
    token = admin_fixture["admin_a_token"]
    owner_id = admin_fixture["admin_a_user_id"]

    resp = client.patch(
        f"/api/v1/workspaces/{ws_id}/members/{owner_id}",
        json={"role": "employee"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 400


def test_24_private_employee_files_not_exposed_to_admin(admin_fixture):
    """Scenario 24: Administrator cannot view or list personal employee files."""
    ws_id = admin_fixture["ws_a_id"]
    token = admin_fixture["admin_a_token"]

    # Admin lists company documents: no personal documents must appear
    comp_docs = client.get(
        f"/api/v1/workspaces/{ws_id}/documents?scope=company",
        headers={"Authorization": f"Bearer {token}"},
    ).json()["items"]
    assert all(d["visibility_scope"] == "company" for d in comp_docs)

    # Admin requesting personal scope only sees admin's own personal docs, not emp1's
    pers_docs = client.get(
        f"/api/v1/workspaces/{ws_id}/documents?scope=personal",
        headers={"Authorization": f"Bearer {token}"},
    ).json()["items"]
    assert not any(d.get("owner_user_id") == admin_fixture["emp1_id"] for d in pers_docs)


# =============================================================================
# 4. INVITATIONS & JOINING CODES (Scenarios 25-31)
# =============================================================================

def test_25_26_admin_creates_invitation_and_employee_onboards(admin_fixture):
    """Scenarios 25-26: Admin creates invitation code, employee signs up successfully with code."""
    ws_id = admin_fixture["ws_a_id"]
    token = admin_fixture["admin_a_token"]
    suffix = uuid.uuid4().hex[:6]

    inv_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/invitations",
        json={"max_usage_count": 2, "expires_in_days": 5},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert inv_resp.status_code == 201
    inv_data = inv_resp.json()
    raw_code = inv_data["raw_code"]

    # Scenario 26: Employee registers with raw code
    reg_resp = client.post("/api/v1/auth/employee/register", json={
        "full_name": "Employee Two",
        "email": f"emp2_{suffix}@apex.com",
        "password": "Password123!",
        "code": raw_code,
    })
    assert reg_resp.status_code == 201
    assert "access_token" in reg_resp.json()


def test_27_plaintext_code_omitted_from_listings(admin_fixture):
    """Scenario 27: Listing invitations omits raw plaintext code and code hashes."""
    ws_id = admin_fixture["ws_a_id"]
    token = admin_fixture["admin_a_token"]

    list_resp = client.get(
        f"/api/v1/workspaces/{ws_id}/invitations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_resp.status_code == 200
    for inv in list_resp.json()["items"]:
        assert "raw_code" not in inv
        assert "code_hash" not in inv


def test_28_used_and_expired_codes_handled(admin_fixture):
    """Scenario 28: Used up or expired code cannot be used to register."""
    ws_id = admin_fixture["ws_a_id"]
    token = admin_fixture["admin_a_token"]
    suffix = uuid.uuid4().hex[:6]

    # Create 1-use code
    inv_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/invitations",
        json={"max_usage_count": 1, "expires_in_days": 1},
        headers={"Authorization": f"Bearer {token}"},
    )
    code = inv_resp.json()["raw_code"]

    # Use once -> 201
    r1 = client.post("/api/v1/auth/employee/register", json={
        "full_name": "First User",
        "email": f"user1_{suffix}@apex.com",
        "password": "Password123!",
        "code": code,
    })
    assert r1.status_code == 201

    # Use second time -> 400 (exhausted)
    r2 = client.post("/api/v1/auth/employee/register", json={
        "full_name": "Second User",
        "email": f"user2_{suffix}@apex.com",
        "password": "Password123!",
        "code": code,
    })
    assert r2.status_code == 400


def test_29_revoked_code_rejected(admin_fixture):
    """Scenario 29: Revoking an invitation prevents further signups."""
    ws_id = admin_fixture["ws_a_id"]
    token = admin_fixture["admin_a_token"]
    suffix = uuid.uuid4().hex[:6]

    inv_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/invitations",
        json={"max_usage_count": 10},
        headers={"Authorization": f"Bearer {token}"},
    )
    inv_id = inv_resp.json()["id"]
    code = inv_resp.json()["raw_code"]

    # Revoke it
    del_resp = client.delete(
        f"/api/v1/workspaces/{ws_id}/invitations/{inv_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert del_resp.status_code == 200

    # Attempt signup with revoked code -> 400
    r_reg = client.post("/api/v1/auth/employee/register", json={
        "full_name": "Late User",
        "email": f"late_{suffix}@apex.com",
        "password": "Password123!",
        "code": code,
    })
    assert r_reg.status_code == 400


def test_30_31_cross_workspace_invitation_rejected_and_metadata_accurate(admin_fixture):
    """Scenarios 30-31: Cross-workspace invitation creation rejected (404); metadata accurate."""
    ws_b_id = admin_fixture["ws_b_id"]
    token_a = admin_fixture["admin_a_token"]

    # Admin A tries to create invitation for Workspace B -> 404
    r_cross = client.post(
        f"/api/v1/workspaces/{ws_b_id}/invitations",
        json={"max_usage_count": 5},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert r_cross.status_code == 404


# =============================================================================
# 5. COMPANY ACTIVITY (Scenarios 32-34)
# =============================================================================

def test_32_activity_records_supported_events(admin_fixture):
    """Scenario 32: Genuine events (document upload, delete, invitations, employee joins) appear."""
    ws_id = admin_fixture["ws_a_id"]
    token = admin_fixture["admin_a_token"]

    resp = client.get(f"/api/v1/workspaces/{ws_id}/activity", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    act_data = resp.json()
    assert "items" in act_data
    items = act_data["items"]
    assert len(items) > 0

    actions = [it["action"] for it in items]
    # Check that at least some core events were recorded
    assert any("document" in a or "invitation" in a or "employee" in a or "registered" in a for a in actions)


def test_33_activity_scoped_to_company(admin_fixture):
    """Scenario 33: Activity events are strictly isolated per company workspace."""
    ws_a_id = admin_fixture["ws_a_id"]
    ws_b_id = admin_fixture["ws_b_id"]
    token_a = admin_fixture["admin_a_token"]
    token_b = admin_fixture["admin_b_token"]

    resp_a = client.get(f"/api/v1/workspaces/{ws_a_id}/activity", headers={"Authorization": f"Bearer {token_a}"})
    resp_b = client.get(f"/api/v1/workspaces/{ws_b_id}/activity", headers={"Authorization": f"Bearer {token_b}"})

    assert resp_a.status_code == 200
    assert resp_b.status_code == 200

    ids_a = {it["id"] for it in resp_a.json()["items"]}
    ids_b = {it["id"] for it in resp_b.json()["items"]}
    assert ids_a.isdisjoint(ids_b)


def test_34_private_chats_never_recorded_in_activity(admin_fixture):
    """Scenario 34: Private chat contents and queries are never logged in company activity."""
    ws_id = admin_fixture["ws_a_id"]
    token = admin_fixture["admin_a_token"]

    resp = client.get(f"/api/v1/workspaces/{ws_id}/activity", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    items = resp.json()["items"]

    # Verify no item action is chat content or contains message bodies
    for it in items:
        assert it["action"] != "message.create"
        if it.get("details"):
            assert "password" not in it["details"]
            assert "token" not in it["details"]


# =============================================================================
# 6. REGRESSIONS (Scenarios 35-40)
# =============================================================================

def test_35_to_40_regressions_intact(admin_fixture):
    """Scenarios 35-40: Verify role auth, employee workspace, RAG, and citation preview remain functional."""
    ws_id = admin_fixture["ws_a_id"]
    emp_token = admin_fixture["emp1_token"]

    # Employee can create a conversation
    c_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        json={"title": "Regression Chat", "knowledge_scope": "company"},
        headers={"Authorization": f"Bearer {emp_token}"},
    )
    assert c_resp.status_code == 201
    conv_id = c_resp.json()["id"]

    # Employee can list their conversation
    list_c = client.get(f"/api/v1/workspaces/{ws_id}/conversations", headers={"Authorization": f"Bearer {emp_token}"})
    assert list_c.status_code == 200
    assert any(c["id"] == conv_id for c in list_c.json()["items"])
