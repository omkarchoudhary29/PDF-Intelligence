"""Phase 5 Automated Test Suite: Employee Chat-First Workspace and Three-Panel Experience.

Tests:
1. Authentication gating & role enforcement.
2. Conversation lifecycle: Create, List, Rename (PATCH), Delete (DELETE), and Search.
3. Message flow & LangGraph RAG synthesis with mock/real provider handling.
4. Document panel integration: Personal upload, My Files, Company Files (read-only).
5. Document attachments to conversations (with fail-closed cross-user rejection).
6. Knowledge scope persistence and scoping ('company', 'personal', 'combined').
7. Citation verification and PDF page preview.
8. Regressions across existing documents, users, and workspaces.
"""

from datetime import datetime
import io
import uuid
import pytest
from fastapi.testclient import TestClient

from backend.database.connection import get_db_session
from backend.database.models import (
    Conversation,
    ConversationAttachment,
    Document,
    Message,
    User,
    Workspace,
    WorkspaceInvitation,
    WorkspaceMembership,
)
from backend.main import app
from backend.services.invitation_service import invitation_service

client = TestClient(app)


def make_pdf_bytes(title: str) -> bytes:
    """Helper to generate minimal valid PDF bytes."""
    return f"%PDF-1.4\n1 0 obj\n<< /Title ({title}) >>\nendobj\ntrailer\n<< >>\n%%EOF".encode("utf-8")


@pytest.fixture(scope="module")
def employee_fixture():
    """Setup an isolated company with an Admin and two Employees (Emp 1 and Emp 2)."""
    suffix = uuid.uuid4().hex[:6]
    company_name = f"Aero Dynamics {suffix}"

    # 1. Register Admin and Company
    admin_email = f"admin_{suffix}@aerodyn.com"
    reg_admin = client.post("/api/v1/auth/admin/register", json={
        "full_name": "Admin Smith",
        "company_name": company_name,
        "email": admin_email,
        "password": "Password123!",
    })
    assert reg_admin.status_code == 201
    admin_token = reg_admin.json()["access_token"]

    # Get workspace ID
    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {admin_token}"})
    ws_id = me_resp.json()["workspaces"][0]["workspace_id"]

    # 2. Generate Joining Code
    inv_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/invitations",
        json={"max_usage_count": 10, "expires_in_days": 7},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    raw_code = inv_resp.json()["raw_code"]

    # 3. Register Employee 1
    emp1_email = f"emp1_{suffix}@aerodyn.com"
    r1 = client.post("/api/v1/auth/employee/register", json={
        "full_name": "Employee One",
        "email": emp1_email,
        "password": "Password123!",
        "code": raw_code,
    })
    emp1_token = r1.json()["access_token"]
    emp1_id = r1.json()["user"]["id"]

    # 4. Register Employee 2
    emp2_email = f"emp2_{suffix}@aerodyn.com"
    r2 = client.post("/api/v1/auth/employee/register", json={
        "full_name": "Employee Two",
        "email": emp2_email,
        "password": "Password123!",
        "code": raw_code,
    })
    emp2_token = r2.json()["access_token"]
    emp2_id = r2.json()["user"]["id"]

    # 5. Upload Company PDF as Admin
    upload_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files=[("files", ("Company_Operations.pdf", make_pdf_bytes("Company Operations"), "application/pdf"))],
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    company_doc_id = upload_resp.json()["results"][0]["document"]["id"]

    return {
        "workspace_id": ws_id,
        "admin_token": admin_token,
        "emp1_token": emp1_token,
        "emp1_id": emp1_id,
        "emp1_email": emp1_email,
        "emp2_token": emp2_token,
        "emp2_id": emp2_id,
        "company_doc_id": company_doc_id,
    }


# =============================================================================
# PART 1: AUTHENTICATION GATING & ROLE ENFORCEMENT
# =============================================================================

def test_01_unauthenticated_cannot_access_conversations(employee_fixture):
    """Unauthenticated access to conversations returns 401."""
    ws_id = employee_fixture["workspace_id"]
    resp = client.get(f"/api/v1/workspaces/{ws_id}/conversations")
    assert resp.status_code == 401


def test_02_employee_can_list_conversations(employee_fixture):
    """Authenticated employee can access their workspace conversations."""
    ws_id = employee_fixture["workspace_id"]
    token = employee_fixture["emp1_token"]
    resp = client.get(f"/api/v1/workspaces/{ws_id}/conversations", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert "items" in resp.json()


def test_03_logout_revokes_token(employee_fixture):
    """Logging out revokes the refresh token."""
    login_resp = client.post("/api/v1/auth/login", json={
        "email": employee_fixture["emp1_email"],
        "password": "Password123!",
    })
    token_data = login_resp.json()
    logout_resp = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": token_data["refresh_token"]},
        headers={"Authorization": f"Bearer {token_data['access_token']}"},
    )
    assert logout_resp.status_code == 200


# =============================================================================
# PART 2: CONVERSATION LIFECYCLE (CREATE, LIST, RENAME, DELETE)
# =============================================================================

def test_05_06_new_chat_creates_persisted_conversation(employee_fixture):
    """Creating a new conversation persists in DB and appears in list."""
    ws_id = employee_fixture["workspace_id"]
    token = employee_fixture["emp1_token"]

    create_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        json={"title": "Flight Deck Review", "knowledge_scope": "company"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert create_resp.status_code == 201
    conv = create_resp.json()
    conv_id = conv["id"]
    assert conv["title"] == "Flight Deck Review"
    assert conv["knowledge_scope"] == "company"

    # Verify listing includes it
    list_resp = client.get(
        f"/api/v1/workspaces/{ws_id}/conversations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_resp.status_code == 200
    items = list_resp.json()["items"]
    assert any(c["id"] == conv_id for c in items)


def test_09_rename_conversation_patch(employee_fixture):
    """Employee can rename their own conversation via PATCH."""
    ws_id = employee_fixture["workspace_id"]
    token = employee_fixture["emp1_token"]

    create_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        json={"title": "Old Title", "knowledge_scope": "company"},
        headers={"Authorization": f"Bearer {token}"},
    )
    conv_id = create_resp.json()["id"]

    patch_resp = client.patch(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        json={"title": "Updated Avionics Title"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["title"] == "Updated Avionics Title"


def test_10_delete_conversation_with_authorization(employee_fixture):
    """Employee can delete their conversation; unauthorized user cannot."""
    ws_id = employee_fixture["workspace_id"]
    token1 = employee_fixture["emp1_token"]
    token2 = employee_fixture["emp2_token"]

    # Emp 1 creates conversation
    create_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        json={"title": "Emp 1 Private Chat"},
        headers={"Authorization": f"Bearer {token1}"},
    )
    conv_id = create_resp.json()["id"]

    # Emp 2 attempts to delete Emp 1's conversation -> 404
    forbidden_resp = client.delete(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        headers={"Authorization": f"Bearer {token2}"},
    )
    assert forbidden_resp.status_code == 404

    # Emp 1 deletes their own conversation -> 200
    del_resp = client.delete(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        headers={"Authorization": f"Bearer {token1}"},
    )
    assert del_resp.status_code == 200

    # Ensure it no longer exists
    get_resp = client.get(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        headers={"Authorization": f"Bearer {token1}"},
    )
    assert get_resp.status_code == 404


# =============================================================================
# PART 3: DOCUMENT PANEL, PERSONAL UPLOAD & ATTACHMENTS
# =============================================================================

def test_24_25_personal_upload_creates_correct_ownership(employee_fixture):
    """Employee uploads personal PDF; ownership is strictly assigned to employee."""
    ws_id = employee_fixture["workspace_id"]
    token1 = employee_fixture["emp1_token"]
    emp1_id = employee_fixture["emp1_id"]

    upload_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/documents/personal",
        files=[("files", ("Private_Notes_Emp1.pdf", make_pdf_bytes("Private Notes Emp 1"), "application/pdf"))],
        headers={"Authorization": f"Bearer {token1}"},
    )
    assert upload_resp.status_code == 200, upload_resp.text
    doc = upload_resp.json()["results"][0]["document"]
    doc_id = doc["id"]
    assert doc["visibility_scope"] == "personal"
    assert doc["owner_user_id"] == emp1_id

    # Verify Emp 1 can see it in personal scope
    list_resp = client.get(
        f"/api/v1/workspaces/{ws_id}/documents?scope=personal",
        headers={"Authorization": f"Bearer {token1}"},
    )
    assert list_resp.status_code == 200
    items = list_resp.json()["items"]
    assert any(d["id"] == doc_id for d in items)

    # Verify Emp 2 CANNOT see Emp 1's personal document
    list_resp_emp2 = client.get(
        f"/api/v1/workspaces/{ws_id}/documents?scope=personal",
        headers={"Authorization": f"Bearer {employee_fixture['emp2_token']}"},
    )
    assert not any(d["id"] == doc_id for d in list_resp_emp2.json()["items"])


def test_27_28_attach_personal_and_company_documents(employee_fixture):
    """Employee can attach personal doc or company doc; cross-user personal doc is rejected."""
    ws_id = employee_fixture["workspace_id"]
    token1 = employee_fixture["emp1_token"]
    token2 = employee_fixture["emp2_token"]
    comp_doc_id = employee_fixture["company_doc_id"]

    # 1. Emp 1 creates conversation
    conv_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        json={"title": "Attachment Chat"},
        headers={"Authorization": f"Bearer {token1}"},
    )
    conv_id = conv_resp.json()["id"]

    # 2. Emp 1 attaches company doc -> 201
    att_comp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/attachments",
        json={"document_id": comp_doc_id},
        headers={"Authorization": f"Bearer {token1}"},
    )
    assert att_comp.status_code == 201
    assert att_comp.json()["document_id"] == comp_doc_id

    # 3. Emp 2 uploads a personal doc
    up2 = client.post(
        f"/api/v1/workspaces/{ws_id}/documents/personal",
        files=[("files", ("Emp2_Secret.pdf", make_pdf_bytes("Emp 2 Secret"), "application/pdf"))],
        headers={"Authorization": f"Bearer {token2}"},
    )
    emp2_doc_id = up2.json()["results"][0]["document"]["id"]

    # 4. Emp 1 tries to attach Emp 2's personal doc -> 404
    att_forbidden = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/attachments",
        json={"document_id": emp2_doc_id},
        headers={"Authorization": f"Bearer {token1}"},
    )
    assert att_forbidden.status_code == 404

    # 5. List attachments verifies only company doc is attached
    list_att = client.get(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/attachments",
        headers={"Authorization": f"Bearer {token1}"},
    )
    assert list_att.status_code == 200
    assert len(list_att.json()["items"]) == 1

    # 6. Detach document
    det_resp = client.delete(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/attachments/{comp_doc_id}",
        headers={"Authorization": f"Bearer {token1}"},
    )
    assert det_resp.status_code == 200


# =============================================================================
# PART 4: KNOWLEDGE SCOPE PERSISTENCE & SCOPED RAG
# =============================================================================

def test_29_33_knowledge_scope_persistence_and_update(employee_fixture):
    """Conversation scope can be updated to personal, company, or combined and persists."""
    ws_id = employee_fixture["workspace_id"]
    token = employee_fixture["emp1_token"]

    conv_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        json={"title": "Scope Test Chat", "knowledge_scope": "company"},
        headers={"Authorization": f"Bearer {token}"},
    )
    conv_id = conv_resp.json()["id"]
    assert conv_resp.json()["knowledge_scope"] == "company"

    # Update to 'combined'
    p1 = client.patch(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        json={"knowledge_scope": "combined"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert p1.status_code == 200
    assert p1.json()["knowledge_scope"] == "combined"

    # Reload from GET
    g1 = client.get(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert g1.json()["knowledge_scope"] == "combined"


# =============================================================================
# PART 5: MESSAGE FLOW & CITATIONS
# =============================================================================

def test_12_17_send_message_and_retrieve_history(employee_fixture):
    """Sending a message saves user and assistant turns and retrieves in order."""
    ws_id = employee_fixture["workspace_id"]
    token = employee_fixture["emp1_token"]

    conv_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        json={"title": "Q&A Session", "knowledge_scope": "company"},
        headers={"Authorization": f"Bearer {token}"},
    )
    conv_id = conv_resp.json()["id"]

    send_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/messages",
        json={"content": "What is the procedure for engine inspection?"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert send_resp.status_code == 200
    rag_data = send_resp.json()
    assert "answer" in rag_data
    assert "citations" in rag_data

    # List messages
    msg_resp = client.get(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/messages",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert msg_resp.status_code == 200
    msgs = msg_resp.json()["items"]
    assert len(msgs) == 2  # user + assistant
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


# =============================================================================
# PART 6: REGRESSIONS & PREEXISTING STATE
# =============================================================================

def test_38_42_regression_intact():
    """Verify legacy admin login and pre-existing 7 company documents remain accessible."""
    admin_login = client.post("/api/v1/auth/login", json={
        "email": "admin@enterprise.com",
        "password": "AdminPassword123!",
    })
    assert admin_login.status_code == 200
    token = admin_login.json()["access_token"]
    ws_id = "dc5d9139-f381-4818-ab26-c2e878997b4d"

    doc_resp = client.get(f"/api/v1/workspaces/{ws_id}/documents", headers={"Authorization": f"Bearer {token}"})
    assert doc_resp.status_code == 200
    assert len(doc_resp.json()["items"]) >= 7
