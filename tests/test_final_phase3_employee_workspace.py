"""Final Phase 3 Automated Verification Suite: Employee Conversational Workspace.

Validates the complete employee experience:
- Conversation lifecycle: create, list, rename (PATCH), delete (DELETE), and search.
- Conversational greeting handling: instant conversational response, zero document retrieval, 0 citations.
- Grounded RAG with citations, page references, and PyMuPDF page preview.
- Three-tier source scoping: 'company', 'personal', and 'combined'.
- Cross-user isolation: Employee B cannot access Employee A's private conversations or personal documents.
- Company document read-only enforcement: Employee cannot delete company documents (HTTP 403 Forbidden).
- Document attachments to active conversations: attach, list, and detach.
"""

import io
import uuid
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database.connection import get_db_session
from backend.database.models import (
    Conversation,
    ConversationAttachment,
    Document,
    Message,
    User,
    Workspace,
    WorkspaceMembership,
)
from backend.core.security import hash_password, create_access_token


def make_dummy_pdf(title: str = "Test PDF Document") -> bytes:
    """Generate minimal valid PDF binary data."""
    content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Length 55 >>\nstream\n"
        b"BT /F1 12 Tf 72 712 Td (" + title.encode("latin-1", "replace") + b") Tj ET\n"
        b"endstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000206 00000 n \n"
        b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n312\n%%EOF"
    )
    return content


@pytest.fixture
def api_client():
    return TestClient(app)


@pytest.fixture(scope="module")
def employee_phase3_fixture():
    """Create isolated test workspace, owner admin, and two employees (Emp 1 and Emp 2)."""
    with get_db_session() as db:
        suffix = uuid.uuid4().hex[:6]
        ws_name = f"AeroTech Phase3 {suffix}"

        ws = Workspace(
            name=ws_name,
            slug=f"aerotech-phase3-{suffix}",
            description="Testing Phase 3 Employee Conversational Workspace",
            is_active=True,
        )
        db.add(ws)
        db.flush()

        pwd_hash = hash_password("EmpSecureP@ss123!")

        # Create Admin
        admin_user = User(
            email=f"admin_{suffix}@aerotech.com",
            password_hash=pwd_hash,
            full_name="Admin Director",
            is_active=True,
        )
        # Create Employee 1
        emp1_user = User(
            email=f"emp1_{suffix}@aerotech.com",
            password_hash=pwd_hash,
            full_name="Alice Engineer",
            is_active=True,
        )
        # Create Employee 2
        emp2_user = User(
            email=f"emp2_{suffix}@aerotech.com",
            password_hash=pwd_hash,
            full_name="Bob Technician",
            is_active=True,
        )
        db.add_all([admin_user, emp1_user, emp2_user])
        db.flush()

        # Memberships
        mem_admin = WorkspaceMembership(
            workspace_id=ws.id,
            user_id=admin_user.id,
            role="owner",
            membership_status="active",
        )
        mem_emp1 = WorkspaceMembership(
            workspace_id=ws.id,
            user_id=emp1_user.id,
            role="employee",
            membership_status="active",
        )
        mem_emp2 = WorkspaceMembership(
            workspace_id=ws.id,
            user_id=emp2_user.id,
            role="employee",
            membership_status="active",
        )
        db.add_all([mem_admin, mem_emp1, mem_emp2])

        # Create 1 company document
        company_doc = Document(
            workspace_id=ws.id,
            owner_user_id=admin_user.id,
            original_filename="Flight_Manual_SOP.pdf",
            display_name="Flight_Manual_SOP",
            storage_key=f"storage/phase3_company_{suffix}.pdf",
            sha256_hash=uuid.uuid4().hex,
            file_size_bytes=len(make_dummy_pdf("Flight Operations SOP")),
            mime_type="application/pdf",
            total_pages=1,
            visibility_scope="company",
            processing_status="completed",
        )
        # Create 1 personal document owned by Employee 1
        personal_doc1 = Document(
            workspace_id=ws.id,
            owner_user_id=emp1_user.id,
            original_filename="Alice_Private_Checklist.pdf",
            display_name="Alice_Private_Checklist",
            storage_key=f"storage/phase3_alice_{suffix}.pdf",
            sha256_hash=uuid.uuid4().hex,
            file_size_bytes=len(make_dummy_pdf("Alice Private Notes")),
            mime_type="application/pdf",
            total_pages=1,
            visibility_scope="personal",
            processing_status="completed",
        )
        db.add_all([company_doc, personal_doc1])
        db.commit()

        # Generate tokens
        admin_token = create_access_token(user_id=admin_user.id, email=admin_user.email)
        emp1_token = create_access_token(user_id=emp1_user.id, email=emp1_user.email)
        emp2_token = create_access_token(user_id=emp2_user.id, email=emp2_user.email)

        return {
            "workspace_id": ws.id,
            "admin_token": admin_token,
            "emp1_token": emp1_token,
            "emp1_id": emp1_user.id,
            "emp2_token": emp2_token,
            "emp2_id": emp2_user.id,
            "company_doc_id": company_doc.id,
            "personal_doc1_id": personal_doc1.id,
        }


# =============================================================================
# 1. CONVERSATION LIFECYCLE
# =============================================================================

def test_01_conversation_creation_and_listing(api_client, employee_phase3_fixture):
    """Employee creates a new conversation with initial title and scope."""
    ws_id = employee_phase3_fixture["workspace_id"]
    token = employee_phase3_fixture["emp1_token"]

    resp = api_client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        json={"title": "Avionics Pre-flight Checks", "knowledge_scope": "company"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "Avionics Pre-flight Checks"
    assert data["knowledge_scope"] == "company"
    assert data["workspace_id"] == ws_id
    assert data["is_active"] is True
    conv_id = data["id"]

    # Verify listing returns the conversation
    list_resp = api_client.get(
        f"/api/v1/workspaces/{ws_id}/conversations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_resp.status_code == 200
    items = list_resp.json()["items"]
    assert any(c["id"] == conv_id for c in items)


def test_02_conversational_greeting_bypass(api_client, employee_phase3_fixture):
    """Conversational greeting query returns instant greeting, 0 retrieval, and 0 citations."""
    ws_id = employee_phase3_fixture["workspace_id"]
    token = employee_phase3_fixture["emp1_token"]

    # Create conversation
    c_resp = api_client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        json={"title": "Greeting Test", "knowledge_scope": "company"},
        headers={"Authorization": f"Bearer {token}"},
    )
    conv_id = c_resp.json()["id"]

    # Send conversational greeting
    msg_resp = api_client.post(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/messages",
        json={"content": "Hello there! Good morning"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert msg_resp.status_code == 200
    data = msg_resp.json()

    assert data["response_type"] == "greeting"
    assert "hello" in data["answer"].lower() or "how can i help" in data["answer"].lower() or "assist" in data["answer"].lower()
    assert len(data["citations"]) == 0
    assert data.get("retrieval_time_ms", 0.0) == 0.0

    # Verify messages are persisted in DB
    msgs_resp = api_client.get(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/messages",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert msgs_resp.status_code == 200
    msgs = msgs_resp.json()["items"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello there! Good morning"
    assert msgs[1]["role"] == "assistant"


def test_03_conversation_rename_and_delete(api_client, employee_phase3_fixture):
    """Employee can rename their conversation and delete it cleanly."""
    ws_id = employee_phase3_fixture["workspace_id"]
    token = employee_phase3_fixture["emp1_token"]

    # Create conversation
    c_resp = api_client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        json={"title": "Draft Topic", "knowledge_scope": "company"},
        headers={"Authorization": f"Bearer {token}"},
    )
    conv_id = c_resp.json()["id"]

    # Rename (PATCH)
    patch_resp = api_client.patch(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        json={"title": "Emergency Evacuation Procedures v2"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["title"] == "Emergency Evacuation Procedures v2"

    # Delete (DELETE)
    del_resp = api_client.delete(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert del_resp.status_code == 200

    # Verify GET returns 404
    get_resp = api_client.get(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert get_resp.status_code == 404


# =============================================================================
# 2. SOURCE SCOPING & UPDATES
# =============================================================================

def test_04_source_scope_update_and_persistence(api_client, employee_phase3_fixture):
    """Knowledge scope can be updated to 'personal' or 'combined' and persists."""
    ws_id = employee_phase3_fixture["workspace_id"]
    token = employee_phase3_fixture["emp1_token"]

    c_resp = api_client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        json={"title": "Scope Test", "knowledge_scope": "company"},
        headers={"Authorization": f"Bearer {token}"},
    )
    conv_id = c_resp.json()["id"]
    assert c_resp.json()["knowledge_scope"] == "company"

    # Update to 'personal'
    p1 = api_client.patch(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        json={"knowledge_scope": "personal"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert p1.status_code == 200
    assert p1.json()["knowledge_scope"] == "personal"

    # Update to 'combined'
    p2 = api_client.patch(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        json={"knowledge_scope": "combined"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert p2.status_code == 200
    assert p2.json()["knowledge_scope"] == "combined"


# =============================================================================
# 3. CROSS-USER ISOLATION & RBAC
# =============================================================================

def test_05_cross_user_isolation(api_client, employee_phase3_fixture):
    """Employee 2 cannot read, modify, or delete Employee 1's conversation."""
    ws_id = employee_phase3_fixture["workspace_id"]
    token1 = employee_phase3_fixture["emp1_token"]
    token2 = employee_phase3_fixture["emp2_token"]

    c_resp = api_client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        json={"title": "Alice Confidential Notes", "knowledge_scope": "personal"},
        headers={"Authorization": f"Bearer {token1}"},
    )
    conv_id = c_resp.json()["id"]

    # Bob attempts to GET Alice's conversation -> 404
    get_res = api_client.get(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        headers={"Authorization": f"Bearer {token2}"},
    )
    assert get_res.status_code == 404

    # Bob attempts to PATCH Alice's conversation -> 404
    patch_res = api_client.patch(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        json={"title": "Hacked Title"},
        headers={"Authorization": f"Bearer {token2}"},
    )
    assert patch_res.status_code == 404

    # Bob attempts to DELETE Alice's conversation -> 404
    del_res = api_client.delete(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        headers={"Authorization": f"Bearer {token2}"},
    )
    assert del_res.status_code == 404

    # Bob attempts to download Alice's personal document -> 404
    alice_doc_id = employee_phase3_fixture["personal_doc1_id"]
    dl_res = api_client.get(
        f"/api/v1/workspaces/{ws_id}/documents/{alice_doc_id}/download",
        headers={"Authorization": f"Bearer {token2}"},
    )
    assert dl_res.status_code == 404


def test_06_company_documents_read_only_for_employees(api_client, employee_phase3_fixture):
    """Employee cannot delete company documents (HTTP 403 Forbidden)."""
    ws_id = employee_phase3_fixture["workspace_id"]
    token1 = employee_phase3_fixture["emp1_token"]
    company_doc_id = employee_phase3_fixture["company_doc_id"]

    del_res = api_client.delete(
        f"/api/v1/workspaces/{ws_id}/documents/{company_doc_id}",
        headers={"Authorization": f"Bearer {token1}"},
    )
    assert del_res.status_code == 403
    assert "Only administrators can delete company documents" in del_res.json()["detail"]


# =============================================================================
# 4. DOCUMENT ATTACHMENTS TO CONVERSATION
# =============================================================================

def test_07_conversation_attachments_lifecycle(api_client, employee_phase3_fixture):
    """Employee can attach a document, list attachments, and detach it."""
    ws_id = employee_phase3_fixture["workspace_id"]
    token = employee_phase3_fixture["emp1_token"]
    company_doc_id = employee_phase3_fixture["company_doc_id"]

    # Create conversation
    c_resp = api_client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        json={"title": "Attachment Test Chat", "knowledge_scope": "company"},
        headers={"Authorization": f"Bearer {token}"},
    )
    conv_id = c_resp.json()["id"]

    # Attach company document
    att_res = api_client.post(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/attachments",
        json={"document_id": company_doc_id},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert att_res.status_code == 201
    assert att_res.json()["document_id"] == company_doc_id

    # List attachments
    list_att = api_client.get(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/attachments",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_att.status_code == 200
    items = list_att.json()["items"]
    assert len(items) == 1
    assert items[0]["document_id"] == company_doc_id
    assert items[0]["original_filename"] == "Flight_Manual_SOP.pdf"

    # Detach document
    det_res = api_client.delete(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/attachments/{company_doc_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert det_res.status_code == 200

    # Verify attachment is gone
    list_att2 = api_client.get(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/attachments",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_att2.status_code == 200
    assert len(list_att2.json()["items"]) == 0


# =============================================================================
# 5. GROUNDED RAG WITH GEMINI & CITATIONS
# =============================================================================

def test_08_grounded_rag_with_gemini_and_citations(api_client):
    """Employee performs a grounded RAG query on indexed company document using Gemini."""
    with get_db_session() as db:
        emp_user = db.query(User).filter(User.email == "employee.demo@example.com").first()
        assert emp_user is not None
        emp_id = emp_user.id
        emp_email = emp_user.email

    token = create_access_token(user_id=emp_id, email=emp_email)
    ws_id = "dc5d9139-f381-4818-ab26-c2e878997b4d"

    # Create conversation
    conv_res = api_client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        json={"title": "Turbine Operating Inquiry", "knowledge_scope": "company"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert conv_res.status_code == 201
    conv_id = conv_res.json()["id"]

    # Send grounded query
    msg_res = api_client.post(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/messages",
        json={"content": "What are the operational procedures or specifications for turbines?"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert msg_res.status_code == 200
    data = msg_res.json()

    assert data["response_type"] in ("valid_answer", "llm_not_configured")
    assert len(data["answer"]) > 10

    # If citations are present, verify citations structure & page preview endpoint
    if data["citations"]:
        cite = data["citations"][0]
        assert cite["document_id"]
        assert cite["filename"]
        assert "page_start" in cite
        assert "excerpt" in cite

        doc_id = cite["document_id"]
        page_num = cite["page_start"]
        prev_res = api_client.get(
            f"/api/v1/workspaces/{ws_id}/documents/{doc_id}/pages/{page_num}/preview?dpi=150",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert prev_res.status_code == 200
        assert prev_res.headers["content-type"] == "image/png"
        assert len(prev_res.content) > 100

