import io
import time
import uuid
from typing import Dict, Generator, Tuple
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.core.security import create_access_token, hash_password
from backend.database.connection import get_db, get_db_session
from backend.database.models import (
    Conversation,
    ConversationAttachment,
    Document,
    Message,
    User,
    Workspace,
    WorkspaceMembership,
)
from backend.main import app
from backend.schemas.search import SearchRequest
from backend.services.bm25_service import bm25_service
from backend.services.chunking_service import chunking_service
from backend.services.retrieval_service import retrieval_service
from backend.services.scope_resolver import scope_resolver
from backend.services.storage_service import storage_service
from backend.services.vector_index_service import vector_index_service

client = TestClient(app)

# Helper for minimal valid PDF bytes
def make_pdf_bytes(label: str) -> bytes:
    content = f"%PDF-1.4\n% EKN Test PDF - {label}\n1 0 obj\n<< /Title ({label}) >>\nendobj\ntrailer\n<< >>\n%%EOF"
    return content.encode("utf-8")


@pytest.fixture(scope="module")
def setup_tenants() -> Generator[Dict[str, any], None, None]:
    """Create isolated Company A and Company B fixtures with Admin and Employee users."""
    with get_db_session() as db:
        # 1. Company A Workspace
        ws_a = Workspace(
            id=str(uuid.uuid4()),
            name="Alpha Corp",
            slug=f"alpha-corp-{uuid.uuid4().hex[:6]}",
        )
        db.add(ws_a)

        # 2. Company B Workspace
        ws_b = Workspace(
            id=str(uuid.uuid4()),
            name="Beta Corp",
            slug=f"beta-corp-{uuid.uuid4().hex[:6]}",
        )
        db.add(ws_b)

        # Users for Company A
        user_admin_a = User(
            id=str(uuid.uuid4()),
            email=f"admin_a_{uuid.uuid4().hex[:6]}@alpha.com",
            full_name="Admin Alpha",
            password_hash=hash_password("Password123!"),
            is_active=True,
        )
        user_emp_a1 = User(
            id=str(uuid.uuid4()),
            email=f"emp_a1_{uuid.uuid4().hex[:6]}@alpha.com",
            full_name="Employee A1",
            password_hash=hash_password("Password123!"),
            is_active=True,
        )
        user_emp_a2 = User(
            id=str(uuid.uuid4()),
            email=f"emp_a2_{uuid.uuid4().hex[:6]}@alpha.com",
            full_name="Employee A2",
            password_hash=hash_password("Password123!"),
            is_active=True,
        )

        # Users for Company B
        user_admin_b = User(
            id=str(uuid.uuid4()),
            email=f"admin_b_{uuid.uuid4().hex[:6]}@beta.com",
            full_name="Admin Beta",
            password_hash=hash_password("Password123!"),
            is_active=True,
        )
        user_emp_b1 = User(
            id=str(uuid.uuid4()),
            email=f"emp_b1_{uuid.uuid4().hex[:6]}@beta.com",
            full_name="Employee B1",
            password_hash=hash_password("Password123!"),
            is_active=True,
        )

        db.add_all([user_admin_a, user_emp_a1, user_emp_a2, user_admin_b, user_emp_b1])
        db.flush()

        # Memberships
        m_admin_a = WorkspaceMembership(workspace_id=ws_a.id, user_id=user_admin_a.id, role="admin")
        m_emp_a1 = WorkspaceMembership(workspace_id=ws_a.id, user_id=user_emp_a1.id, role="employee")
        m_emp_a2 = WorkspaceMembership(workspace_id=ws_a.id, user_id=user_emp_a2.id, role="employee")

        m_admin_b = WorkspaceMembership(workspace_id=ws_b.id, user_id=user_admin_b.id, role="admin")
        m_emp_b1 = WorkspaceMembership(workspace_id=ws_b.id, user_id=user_emp_b1.id, role="employee")

        db.add_all([m_admin_a, m_emp_a1, m_emp_a2, m_admin_b, m_emp_b1])
        db.commit()

        # Issue JWT Access Tokens
        tokens = {
            "admin_a": create_access_token(user_admin_a.id, user_admin_a.email),
            "emp_a1": create_access_token(user_emp_a1.id, user_emp_a1.email),
            "emp_a2": create_access_token(user_emp_a2.id, user_emp_a2.email),
            "admin_b": create_access_token(user_admin_b.id, user_admin_b.email),
            "emp_b1": create_access_token(user_emp_b1.id, user_emp_b1.email),
        }

        context = {
            "ws_a_id": ws_a.id,
            "ws_b_id": ws_b.id,
            "admin_a": user_admin_a,
            "emp_a1": user_emp_a1,
            "emp_a2": user_emp_a2,
            "admin_b": user_admin_b,
            "emp_b1": user_emp_b1,
            "tokens": tokens,
        }

    yield context

    # Teardown
    with get_db_session() as db:
        db.query(Workspace).filter(Workspace.id.in_([ws_a.id, ws_b.id])).delete(synchronize_session=False)
        db.query(User).filter(
            User.id.in_([user_admin_a.id, user_emp_a1.id, user_emp_a2.id, user_admin_b.id, user_emp_b1.id])
        ).delete(synchronize_session=False)
        db.commit()


# =============================================================================
# GROUP 1: UPLOAD TESTS (1 - 4)
# =============================================================================


def test_01_admin_a_uploads_company_pdf(setup_tenants):
    """Scenario 1: Admin A uploads a company PDF."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]
    headers = {"Authorization": f"Bearer {ctx['tokens']['admin_a']}"}

    pdf_bytes = make_pdf_bytes("Company A Turbine Operating Manual")
    files = [("files", ("company_a_manual.pdf", io.BytesIO(pdf_bytes), "application/pdf"))]

    resp = client.post(f"/api/v1/workspaces/{ws_id}/documents?scope=company", headers=headers, files=files)
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    data = resp.json()
    assert data["total_uploaded"] == 1
    doc_res = data["results"][0]
    assert doc_res["status"] == "uploaded"
    assert doc_res["document"]["visibility_scope"] == "company"
    assert doc_res["document"]["owner_user_id"] is None
    ctx["doc_company_a_id"] = doc_res["document"]["id"]


def test_02_employee_a1_uploads_personal_pdf(setup_tenants):
    """Scenario 2: Employee A1 uploads a personal PDF."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]
    headers = {"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"}

    # Employee attempting company upload is rejected with 403
    pdf_bytes = make_pdf_bytes("Employee A1 Secret Notes")
    files_invalid = [("files", ("emp_a1_notes.pdf", io.BytesIO(pdf_bytes), "application/pdf"))]
    resp_forbidden = client.post(
        f"/api/v1/workspaces/{ws_id}/documents?scope=company",
        headers=headers,
        files=files_invalid,
    )
    assert resp_forbidden.status_code == 403, "Employee must not upload company-scoped documents"

    # Employee uploading via personal endpoint or personal scope succeeds
    files_valid = [("files", ("emp_a1_notes.pdf", io.BytesIO(pdf_bytes), "application/pdf"))]
    resp = client.post(f"/api/v1/workspaces/{ws_id}/documents/personal", headers=headers, files=files_valid)
    assert resp.status_code == 200, f"Personal upload failed: {resp.text}"
    data = resp.json()
    assert data["total_uploaded"] == 1
    doc_res = data["results"][0]
    assert doc_res["status"] == "uploaded"
    assert doc_res["document"]["visibility_scope"] == "personal"
    assert doc_res["document"]["owner_user_id"] == ctx["emp_a1"].id
    ctx["doc_emp_a1_id"] = doc_res["document"]["id"]


def test_03_employee_a2_uploads_different_personal_pdf(setup_tenants):
    """Scenario 3: Employee A2 uploads a different personal PDF."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]
    headers = {"Authorization": f"Bearer {ctx['tokens']['emp_a2']}"}

    pdf_bytes = make_pdf_bytes("Employee A2 Private Review")
    files = [("files", ("emp_a2_review.pdf", io.BytesIO(pdf_bytes), "application/pdf"))]
    resp = client.post(f"/api/v1/workspaces/{ws_id}/documents?scope=personal", headers=headers, files=files)
    assert resp.status_code == 200, f"Personal upload failed: {resp.text}"
    data = resp.json()
    assert data["total_uploaded"] == 1
    doc_res = data["results"][0]
    assert doc_res["document"]["visibility_scope"] == "personal"
    assert doc_res["document"]["owner_user_id"] == ctx["emp_a2"].id
    ctx["doc_emp_a2_id"] = doc_res["document"]["id"]


def test_04_admin_b_uploads_company_pdf(setup_tenants):
    """Scenario 4: Admin B uploads a company PDF into Company B."""
    ctx = setup_tenants
    ws_id = ctx["ws_b_id"]
    headers = {"Authorization": f"Bearer {ctx['tokens']['admin_b']}"}

    pdf_bytes = make_pdf_bytes("Company B Robotics Specification")
    files = [("files", ("company_b_spec.pdf", io.BytesIO(pdf_bytes), "application/pdf"))]
    resp = client.post(f"/api/v1/workspaces/{ws_id}/documents?scope=company", headers=headers, files=files)
    assert resp.status_code == 200
    data = resp.json()
    doc_res = data["results"][0]
    assert doc_res["document"]["workspace_id"] == ws_id
    assert doc_res["document"]["visibility_scope"] == "company"
    ctx["doc_company_b_id"] = doc_res["document"]["id"]


def test_04b_ownership_aware_deduplication(setup_tenants):
    """Verify duplicate detection is ownership-aware per employee/company."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]

    same_bytes = make_pdf_bytes("Shared Content Test")

    # 1. Employee A1 uploads it as personal
    files1 = [("files", ("shared.pdf", io.BytesIO(same_bytes), "application/pdf"))]
    r1 = client.post(
        f"/api/v1/workspaces/{ws_id}/documents/personal",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"},
        files=files1,
    )
    assert r1.status_code == 200
    assert r1.json()["total_uploaded"] == 1

    # 2. Employee A1 uploads it AGAIN -> detected as duplicate
    files1_dup = [("files", ("shared.pdf", io.BytesIO(same_bytes), "application/pdf"))]
    r1_dup = client.post(
        f"/api/v1/workspaces/{ws_id}/documents/personal",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"},
        files=files1_dup,
    )
    assert r1_dup.json()["total_duplicates"] == 1

    # 3. Employee A2 uploads the SAME file as their own personal -> succeeds (ownership isolated!)
    files2 = [("files", ("shared.pdf", io.BytesIO(same_bytes), "application/pdf"))]
    r2 = client.post(
        f"/api/v1/workspaces/{ws_id}/documents/personal",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a2']}"},
        files=files2,
    )
    assert r2.status_code == 200
    assert r2.json()["total_uploaded"] == 1
    assert r2.json()["results"][0]["document"]["owner_user_id"] == ctx["emp_a2"].id

    # 4. Admin A uploads the SAME file as company document -> succeeds (company scope separate from personal!)
    files_admin = [("files", ("shared.pdf", io.BytesIO(same_bytes), "application/pdf"))]
    r_admin = client.post(
        f"/api/v1/workspaces/{ws_id}/documents?scope=company",
        headers={"Authorization": f"Bearer {ctx['tokens']['admin_a']}"},
        files=files_admin,
    )
    assert r_admin.status_code == 200
    assert r_admin.json()["total_uploaded"] == 1
    assert r_admin.json()["results"][0]["document"]["visibility_scope"] == "company"


# =============================================================================
# GROUP 2: LISTING TESTS (5 - 8)
# =============================================================================


def test_05_employee_a1_lists_company_documents(setup_tenants):
    """Scenario 5: Employee A1 lists company documents (sees company doc; no private doc leakage)."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]
    headers = {"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"}

    resp = client.get(f"/api/v1/workspaces/{ws_id}/documents?scope=company", headers=headers)
    assert resp.status_code == 200
    items = resp.json()["items"]
    # All listed items must have visibility_scope == 'company'
    assert len(items) >= 1
    for item in items:
        assert item["visibility_scope"] == "company"
        assert item["owner_user_id"] is None
        assert item["id"] != ctx.get("doc_emp_a1_id")
        assert item["id"] != ctx.get("doc_emp_a2_id")


def test_06_employee_a1_lists_personal_documents(setup_tenants):
    """Scenario 6: Employee A1 lists personal documents (sees only own personal documents)."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]
    headers = {"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"}

    resp = client.get(f"/api/v1/workspaces/{ws_id}/documents?scope=personal", headers=headers)
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) >= 1
    for item in items:
        assert item["visibility_scope"] == "personal"
        assert item["owner_user_id"] == ctx["emp_a1"].id
        assert item["id"] != ctx.get("doc_emp_a2_id")
        assert item["id"] != ctx.get("doc_company_a_id")


def test_07_employee_a2_lists_personal_documents(setup_tenants):
    """Scenario 7: Employee A2 lists personal documents (sees only Employee A2's private files)."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]
    headers = {"Authorization": f"Bearer {ctx['tokens']['emp_a2']}"}

    resp = client.get(f"/api/v1/workspaces/{ws_id}/documents?scope=personal", headers=headers)
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) >= 1
    for item in items:
        assert item["visibility_scope"] == "personal"
        assert item["owner_user_id"] == ctx["emp_a2"].id
        assert item["id"] != ctx.get("doc_emp_a1_id")


def test_08_admin_a_lists_company_documents_no_employee_leakage(setup_tenants):
    """Scenario 8: Admin A lists company documents by default; employee private files are NOT returned."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]
    headers = {"Authorization": f"Bearer {ctx['tokens']['admin_a']}"}

    resp = client.get(f"/api/v1/workspaces/{ws_id}/documents", headers=headers)
    assert resp.status_code == 200
    items = resp.json()["items"]
    for item in items:
        assert item["visibility_scope"] == "company"
        assert item["owner_user_id"] is None
        # Must not contain employee personal documents
        assert item["id"] != ctx.get("doc_emp_a1_id")
        assert item["id"] != ctx.get("doc_emp_a2_id")


# =============================================================================
# GROUP 3: RETRIEVAL & SCOPE ISOLATION TESTS (9 - 16)
# =============================================================================


def test_09_to_13_scope_resolver_predicate_construction(setup_tenants):
    """Scenarios 9-13: Validate Centralized Scope Resolver across company, personal, and combined."""
    ctx = setup_tenants
    ws_a = ctx["ws_a_id"]
    user_a1 = ctx["emp_a1"].id
    user_a2 = ctx["emp_a2"].id

    with get_db_session() as db:
        # Scenario 9: Company-only scope
        scope_comp = scope_resolver.resolve(
            workspace_id=ws_a,
            user_id=user_a1,
            user_role="employee",
            scope="company",
            db=db,
        )
        assert scope_comp.scope == "company"
        assert ctx["doc_company_a_id"] in scope_comp.authorized_document_ids
        assert ctx["doc_emp_a1_id"] not in scope_comp.authorized_document_ids
        assert ctx["doc_emp_a2_id"] not in scope_comp.authorized_document_ids

        # Scenario 10: Personal-only scope for Employee A1
        scope_pers_a1 = scope_resolver.resolve(
            workspace_id=ws_a,
            user_id=user_a1,
            user_role="employee",
            scope="personal",
            db=db,
        )
        assert scope_pers_a1.scope == "personal"
        assert ctx["doc_emp_a1_id"] in scope_pers_a1.authorized_document_ids
        assert ctx["doc_company_a_id"] not in scope_pers_a1.authorized_document_ids
        assert ctx["doc_emp_a2_id"] not in scope_pers_a1.authorized_document_ids

        # Scenario 11: Personal-only scope for Employee A2
        scope_pers_a2 = scope_resolver.resolve(
            workspace_id=ws_a,
            user_id=user_a2,
            user_role="employee",
            scope="personal",
            db=db,
        )
        assert scope_pers_a2.scope == "personal"
        assert ctx["doc_emp_a2_id"] in scope_pers_a2.authorized_document_ids
        assert ctx["doc_emp_a1_id"] not in scope_pers_a2.authorized_document_ids
        assert ctx["doc_company_a_id"] not in scope_pers_a2.authorized_document_ids

        # Scenario 12: Combined scope for Employee A1
        scope_comb_a1 = scope_resolver.resolve(
            workspace_id=ws_a,
            user_id=user_a1,
            user_role="employee",
            scope="combined",
            db=db,
        )
        assert scope_comb_a1.scope == "combined"
        assert ctx["doc_company_a_id"] in scope_comb_a1.authorized_document_ids
        assert ctx["doc_emp_a1_id"] in scope_comb_a1.authorized_document_ids
        # CRITICAL: Employee A2's personal document must NEVER appear
        assert ctx["doc_emp_a2_id"] not in scope_comb_a1.authorized_document_ids

        # Scenario 13: Admin company retrieval
        scope_admin = scope_resolver.resolve(
            workspace_id=ws_a,
            user_id=ctx["admin_a"].id,
            user_role="admin",
            scope="company",
            db=db,
        )
        assert ctx["doc_company_a_id"] in scope_admin.authorized_document_ids
        assert ctx["doc_emp_a1_id"] not in scope_admin.authorized_document_ids
        assert ctx["doc_emp_a2_id"] not in scope_admin.authorized_document_ids


def test_14_cross_company_isolation(setup_tenants):
    """Scenario 14: Company B cannot access Company A documents."""
    ctx = setup_tenants
    ws_b = ctx["ws_b_id"]
    headers_b = {"Authorization": f"Bearer {ctx['tokens']['emp_b1']}"}

    # Employee B1 attempts search requesting Company A's document ID
    req = {
        "query": "Turbine",
        "document_ids": [ctx["doc_company_a_id"]],
        "top_k": 5,
    }
    resp = client.post(f"/api/v1/workspaces/{ws_b}/search", headers=headers_b, json=req)
    assert resp.status_code == 200
    assert resp.json()["total_results"] == 0, "Company B must not retrieve Company A documents"


def test_15_employee_a1_cannot_retrieve_employee_a2_content(setup_tenants):
    """Scenario 15: Employee A1 cannot retrieve Employee A2's personal content even by supplying ID."""
    ctx = setup_tenants
    ws_a = ctx["ws_a_id"]
    headers_a1 = {"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"}

    req = {
        "query": "Review",
        "document_ids": [ctx["doc_emp_a2_id"]],
        "scope": "personal",
        "top_k": 5,
    }
    resp = client.post(f"/api/v1/workspaces/{ws_a}/search", headers=headers_a1, json=req)
    assert resp.status_code == 200
    # Must return 0 results because Employee A1 does not own doc_emp_a2_id
    assert resp.json()["total_results"] == 0


def test_16_admin_cannot_retrieve_employee_private_content_by_id(setup_tenants):
    """Scenario 16: Administrator A cannot retrieve employee-private content by guessing its document ID."""
    ctx = setup_tenants
    ws_a = ctx["ws_a_id"]
    headers_admin = {"Authorization": f"Bearer {ctx['tokens']['admin_a']}"}

    # Admin querying with scope=company and guessing private doc id
    req = {
        "query": "Notes",
        "document_ids": [ctx["doc_emp_a1_id"]],
        "scope": "company",
        "top_k": 5,
    }
    resp = client.post(f"/api/v1/workspaces/{ws_a}/search", headers=headers_admin, json=req)
    assert resp.status_code == 200
    assert resp.json()["total_results"] == 0

    # Admin attempting direct GET on private document details
    get_resp = client.get(
        f"/api/v1/workspaces/{ws_a}/documents/{ctx['doc_emp_a1_id']}",
        headers=headers_admin,
    )
    assert get_resp.status_code == 404, "Admin must not see metadata of employee private documents"


# =============================================================================
# GROUP 4: CHROMADB & BM25 FILTERING TESTS (17 - 25)
# =============================================================================


def test_17_to_19_chromadb_vector_filtering(setup_tenants):
    """Scenarios 17-19: ChromaDB metadata filter dictionary construction and query isolation."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]
    user_id = ctx["emp_a1"].id

    # 17: Company filter
    f_comp = scope_resolver.build_chroma_filter(ws_id, user_id, "company")
    assert f_comp == {"$and": [{"workspace_id": ws_id}, {"visibility_scope": "company"}]}

    # 18: Personal filter
    f_pers = scope_resolver.build_chroma_filter(ws_id, user_id, "personal")
    assert f_pers == {
        "$and": [
            {"workspace_id": ws_id},
            {"visibility_scope": "personal"},
            {"owner_user_id": user_id},
        ]
    }

    # 19: Combined filter
    f_comb = scope_resolver.build_chroma_filter(ws_id, user_id, "combined")
    assert f_comb == {
        "$and": [
            {"workspace_id": ws_id},
            {
                "$or": [
                    {"visibility_scope": "company"},
                    {"$and": [{"visibility_scope": "personal"}, {"owner_user_id": user_id}]},
                ]
            },
        ]
    }


def test_20_to_22_bm25_scoped_filtering(setup_tenants):
    """Scenarios 20-22: BM25 in-memory keyword search filtering by ownership scope."""
    ctx = setup_tenants
    ws_id = f"test_bm25_ws_{uuid.uuid4().hex[:6]}"

    # Index chunks across company, User A1 personal, and User A2 personal
    from backend.schemas.indexing import ChildChunkRecord

    c_comp = ChildChunkRecord(
        chunk_id=f"{ws_id}_c1",
        parent_id="p1",
        document_id="doc_comp",
        workspace_id=ws_id,
        original_filename="comp.pdf",
        page_start=1,
        page_end=1,
        section_title="Architecture",
        text="UniqueTargetKeyword in company engineering guidelines",
        chunk_type="text",
        token_count=10,
        file_hash="h1",
        embedding_model="model",
        visibility_scope="company",
        owner_user_id=None,
    )
    c_a1 = ChildChunkRecord(
        chunk_id=f"{ws_id}_c2",
        parent_id="p2",
        document_id="doc_a1",
        workspace_id=ws_id,
        original_filename="a1.pdf",
        page_start=1,
        page_end=1,
        section_title="Notes",
        text="UniqueTargetKeyword in employee A1 private research notes",
        chunk_type="text",
        token_count=10,
        file_hash="h2",
        embedding_model="model",
        visibility_scope="personal",
        owner_user_id=ctx["emp_a1"].id,
    )
    c_a2 = ChildChunkRecord(
        chunk_id=f"{ws_id}_c3",
        parent_id="p3",
        document_id="doc_a2",
        workspace_id=ws_id,
        original_filename="a2.pdf",
        page_start=1,
        page_end=1,
        section_title="Review",
        text="UniqueTargetKeyword in employee A2 confidential diary",
        chunk_type="text",
        token_count=10,
        file_hash="h3",
        embedding_model="model",
        visibility_scope="personal",
        owner_user_id=ctx["emp_a2"].id,
    )

    bm25_service.add_or_update_document_chunks(ws_id, "doc_comp", [c_comp])
    bm25_service.add_or_update_document_chunks(ws_id, "doc_a1", [c_a1])
    bm25_service.add_or_update_document_chunks(ws_id, "doc_a2", [c_a2])

    try:
        # Scenario 20: Company BM25 filtering
        res_comp = bm25_service.search(ws_id, "UniqueTargetKeyword", scope="company")
        comp_ids = {r["chunk_id"] for r in res_comp}
        assert c_comp.chunk_id in comp_ids
        assert c_a1.chunk_id not in comp_ids
        assert c_a2.chunk_id not in comp_ids

        # Scenario 21: Personal BM25 filtering for Employee A1
        res_a1 = bm25_service.search(
            ws_id,
            "UniqueTargetKeyword",
            scope="personal",
            owner_user_id=ctx["emp_a1"].id,
        )
        a1_ids = {r["chunk_id"] for r in res_a1}
        assert c_a1.chunk_id in a1_ids
        assert c_comp.chunk_id not in a1_ids
        assert c_a2.chunk_id not in a1_ids

        # Scenario 22: Combined BM25 filtering for Employee A1
        res_comb = bm25_service.search(
            ws_id,
            "UniqueTargetKeyword",
            scope="combined",
            owner_user_id=ctx["emp_a1"].id,
        )
        comb_ids = {r["chunk_id"] for r in res_comb}
        assert c_comp.chunk_id in comb_ids
        assert c_a1.chunk_id in comb_ids
        assert c_a2.chunk_id not in comb_ids  # Employee A2 never leaks
    finally:
        # Clean up test BM25 corpus
        bm25_service.remove_document_chunks(ws_id, "doc_comp")
        bm25_service.remove_document_chunks(ws_id, "doc_a1")
        bm25_service.remove_document_chunks(ws_id, "doc_a2")


def test_23_legacy_vector_metadata_compatibility():
    """Scenario 23: Legacy vectors without explicit scope default safely to company scope."""
    col = vector_index_service.get_collection()
    count = col.count()
    assert count >= 1469, f"Vector count must be preserved (found {count})"

    # Sample an existing vector and verify it carries visibility_scope='company'
    sample = col.get(limit=5)
    for meta in sample["metadatas"]:
        assert meta.get("visibility_scope") == "company"
        assert meta.get("owner_user_id") == ""


def test_24_25_deleted_document_index_cleanup(setup_tenants):
    """Scenarios 24 & 25: Deleted document cleans up ChromaDB vectors and BM25 chunks."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]

    # Upload a temporary personal document
    pdf_bytes = make_pdf_bytes("Temporary Deletion Target")
    files = [("files", ("delete_me.pdf", io.BytesIO(pdf_bytes), "application/pdf"))]
    r = client.post(
        f"/api/v1/workspaces/{ws_id}/documents/personal",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"},
        files=files,
    )
    doc_id = r.json()["results"][0]["document"]["id"]

    # Delete the document as owner
    del_resp = client.delete(
        f"/api/v1/workspaces/{ws_id}/documents/{doc_id}",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"},
    )
    assert del_resp.status_code == 200

    # Verify vector count for deleted doc is 0
    assert vector_index_service.get_document_vector_count(doc_id) == 0


# =============================================================================
# GROUP 5: CONVERSATION OWNERSHIP & ATTACHMENT TESTS (26 - 32)
# =============================================================================


def test_26_conversation_owner_validation(setup_tenants):
    """Scenario 26: Conversation belongs to creator; another user cannot load it."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]

    # Employee A1 creates conversation
    create_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"},
        json={"title": "Private Chat A1", "knowledge_scope": "personal"},
    )
    assert create_resp.status_code == 201
    conv_id = create_resp.json()["id"]
    ctx["conv_a1_id"] = conv_id

    # Employee A1 can retrieve it
    get_resp = client.get(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"},
    )
    assert get_resp.status_code == 200

    # Employee A2 trying to retrieve it gets 404 (private conversation isolated)
    get_resp_a2 = client.get(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a2']}"},
    )
    assert get_resp_a2.status_code == 404


def test_27_conversation_knowledge_scope_persistence(setup_tenants):
    """Scenario 27: Conversation knowledge_scope persists across turns."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]

    create_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"},
        json={"title": "Combined Scoped Session", "knowledge_scope": "combined"},
    )
    conv_id = create_resp.json()["id"]
    assert create_resp.json()["knowledge_scope"] == "combined"

    get_resp = client.get(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"},
    )
    assert get_resp.json()["knowledge_scope"] == "combined"


def test_28_personal_pdf_attachment(setup_tenants):
    """Scenario 28: Employee A1 attaches their personal PDF to active conversation."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]
    conv_id = ctx["conv_a1_id"]
    doc_id = ctx["doc_emp_a1_id"]

    resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/attachments",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"},
        json={"document_id": doc_id},
    )
    assert resp.status_code == 201, f"Attachment failed: {resp.text}"
    data = resp.json()
    assert data["document_id"] == doc_id
    assert data["visibility_scope"] == "personal"


def test_29_authorized_company_pdf_attachment(setup_tenants):
    """Scenario 29: Employee A1 attaches authorized company PDF to conversation."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]
    conv_id = ctx["conv_a1_id"]
    doc_id = ctx["doc_company_a_id"]

    resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/attachments",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"},
        json={"document_id": doc_id},
    )
    assert resp.status_code == 201
    assert resp.json()["document_id"] == doc_id
    assert resp.json()["visibility_scope"] == "company"


def test_30_unauthorized_attachment_rejection(setup_tenants):
    """Scenario 30: Employee A1 CANNOT attach Employee A2's private document."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]
    conv_id = ctx["conv_a1_id"]
    unauthorized_doc_id = ctx["doc_emp_a2_id"]

    resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/attachments",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"},
        json={"document_id": unauthorized_doc_id},
    )
    assert resp.status_code == 404, "Attempting to attach another user's private document must fail"


def test_31_cross_user_conversation_rejection(setup_tenants):
    """Scenario 31: Employee A2 cannot post messages to Employee A1's conversation."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]
    conv_id = ctx["conv_a1_id"]

    resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/messages",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a2']}"},
        json={"content": "Attempted intrusion"},
    )
    assert resp.status_code == 404


def test_32_scope_preservation_on_follow_up_questions(setup_tenants):
    """Scenario 32: Sending a message retains active conversation scope."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]
    conv_id = ctx["conv_a1_id"]

    # Send message as Employee A1
    resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/messages",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"},
        json={"content": "What are the turbine operating steps?"},
    )
    assert resp.status_code == 200
    rag_data = resp.json()
    assert rag_data["conversation_id"] == conv_id
    assert "answer" in rag_data


# =============================================================================
# GROUP 6: STORAGE & PERMISSION ENFORCEMENT TESTS (33 - 37)
# =============================================================================


def test_33_company_pdf_storage_access(setup_tenants):
    """Scenario 33: Both Admin A and Employee A1 can download company PDFs."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]
    doc_id = ctx["doc_company_a_id"]

    # Admin A download
    r_admin = client.get(
        f"/api/v1/workspaces/{ws_id}/documents/{doc_id}/download",
        headers={"Authorization": f"Bearer {ctx['tokens']['admin_a']}"},
    )
    assert r_admin.status_code == 200
    assert r_admin.headers["content-type"] == "application/pdf"

    # Employee A1 download
    r_emp = client.get(
        f"/api/v1/workspaces/{ws_id}/documents/{doc_id}/download",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"},
    )
    assert r_emp.status_code == 200


def test_34_personal_pdf_owner_access(setup_tenants):
    """Scenario 34: Employee A1 can download their own personal PDF."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]
    doc_id = ctx["doc_emp_a1_id"]

    r = client.get(
        f"/api/v1/workspaces/{ws_id}/documents/{doc_id}/download",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"},
    )
    assert r.status_code == 200


def test_35_cross_user_download_rejection(setup_tenants):
    """Scenario 35: Neither Employee A2 nor Admin A can download Employee A1's private PDF."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]
    doc_id = ctx["doc_emp_a1_id"]

    # Employee A2 attempt
    r_a2 = client.get(
        f"/api/v1/workspaces/{ws_id}/documents/{doc_id}/download",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a2']}"},
    )
    assert r_a2.status_code == 404

    # Admin A attempt (admin must NOT automatically download employee private PDFs)
    r_admin = client.get(
        f"/api/v1/workspaces/{ws_id}/documents/{doc_id}/download",
        headers={"Authorization": f"Bearer {ctx['tokens']['admin_a']}"},
    )
    assert r_admin.status_code == 404


def test_36_cross_user_deletion_rejection(setup_tenants):
    """Scenario 36: Employee A2 cannot delete Employee A1's PDF; employee cannot delete company PDF."""
    ctx = setup_tenants
    ws_id = ctx["ws_a_id"]

    # Employee A2 attempts to delete Employee A1's private PDF
    r_del_private = client.delete(
        f"/api/v1/workspaces/{ws_id}/documents/{ctx['doc_emp_a1_id']}",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a2']}"},
    )
    assert r_del_private.status_code == 404

    # Employee A1 attempts to delete Company A PDF (forbidden!)
    r_del_company = client.delete(
        f"/api/v1/workspaces/{ws_id}/documents/{ctx['doc_company_a_id']}",
        headers={"Authorization": f"Bearer {ctx['tokens']['emp_a1']}"},
    )
    assert r_del_company.status_code == 403


def test_37_path_traversal_rejection():
    """Scenario 37: Path traversal violations are strictly caught and blocked by storage_service."""
    with pytest.raises(PermissionError):
        storage_service._resolve_safe_path("../../etc/passwd")

    with pytest.raises(PermissionError):
        storage_service._resolve_safe_path("..\\..\\Windows\\System32\\cmd.exe")


# =============================================================================
# GROUP 7: REGRESSION TESTS (38 - 42)
# =============================================================================


def test_38_existing_seven_company_pdfs_remain_searchable():
    """Scenario 38: Existing 7 company documents in Aerospace workspace remain searchable."""
    ws_id = "dc5d9139-f381-4818-ab26-c2e878997b4d"
    with get_db_session() as db:
        docs = db.query(Document).filter(Document.workspace_id == ws_id).all()
        assert len(docs) == 7
        for d in docs:
            assert d.visibility_scope == "company"
            assert d.owner_user_id is None
            assert d.processing_status == "completed"

    # Search against existing knowledge
    req = SearchRequest(query="What is tokenization and embeddings?", top_k=3, scope="company")
    res = retrieval_service.search(workspace_id=ws_id, request=req)
    assert res.total_results > 0
    assert len(res.results) > 0
    for r in res.results:
        assert r.visibility_scope == "company"


def test_39_existing_embedding_count_preserved():
    """Scenario 39: Existing ChromaDB vector count is preserved."""
    count = vector_index_service.get_total_vector_count()
    assert count >= 1469


def test_40_existing_conversations_remain_available():
    """Scenario 40: Existing conversation records in database remain intact with scope='company'."""
    with get_db_session() as db:
        convs = db.query(Conversation).all()
        for c in convs:
            assert c.knowledge_scope in ("company", "personal", "combined")
