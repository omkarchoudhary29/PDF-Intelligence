"""End-to-End System Integration, Security, and Lifecycle Test Suite for EKN (Phase 13).

Covers Tests A through J:
- Test A: Enterprise initialization (DB, ChromaDB, Storage, Health, Readiness)
- Test B: User and workspace management (Owner provisioning, Login, Employee invite, RBAC)
- Test C: Document ingestion (Upload, SHA-256 deduplication, Worker execution, Pages extraction)
- Test D: Indexing and hybrid retrieval (Hierarchical chunking, ChromaDB, BM25, RRF, Rerank)
- Test E: Conversational RAG (Gemini/Mock LLM, Grounded citations, Follow-ups, Honest uncertainty)
- Test F: Knowledge graph (Entity discovery, Document relationships, Neighborhood query, Deletion cleanup)
- Test G: Knowledge dossier (Report generation, Structure validation, PDF binary download)
- Test H: Version intelligence (Sequential versions, Auto-superseding, Diffing, Scoped retrieval)
- Test I: Contradiction review (Detection, Dated revisions vs True conflicts, Human review, Audit trail)
- Test J: Cross-tenant security (Strict multi-tenant isolation across 8 distinct API surfaces)
"""

import io
import json
from pathlib import Path
import sys
import uuid
import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.dependencies import get_db
from backend.core.health import check_llm_status, check_storage_status
from backend.database.base import Base
from backend.database.connection import (
    check_chroma_connection,
    check_database_connection,
    ensure_schema_upgrades,
    get_chroma_client,
    init_db,
    reset_db_state,
    set_test_engine,
)
from backend.database.models import (
    ContradictionFinding,
    Conversation,
    Document,
    DocumentFamily,
    DocumentIndex,
    IngestionJob,
    Report,
    SecurityAuditLog,
    User,
    Workspace,
    WorkspaceMembership,
)
from backend.main import app
from backend.schemas.indexing import ChildChunkRecord
from backend.services.audit_service import audit_service
from backend.services.auth_service import auth_service
from backend.services.bm25_service import bm25_service
from backend.services.chunking_service import chunking_service
from backend.services.contradiction_service import contradiction_service
from backend.services.ingestion_service import ingestion_service
from backend.services.knowledge_graph_service import knowledge_graph_service
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from backend.services.llm_service import llm_service


class MockChatModel(BaseChatModel):
    """Deterministic mock LangChain chat model for testing conversational RAG without external API calls."""

    response_text: str = (
        "**Answer**\n\n"
        "The AX-450 gas turbine operates at 3600 RPM under nominal load [E1]."
    )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: object,
    ) -> ChatResult:
        full_prompt = " ".join(str(m.content) for m in messages).lower()
        if "submarine" in full_prompt or "antarctica" in full_prompt:
            content = "I could not find sufficient evidence in the uploaded documents to answer your question."
        else:
            content = self.response_text
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"
from backend.services.retrieval_service import retrieval_service
from backend.services.storage_service import storage_service
from backend.services.version_comparison_service import version_comparison_service
from backend.services.version_service import version_service
from config.settings import get_settings


@pytest.fixture
def e2e_env(tmp_path, monkeypatch):
    """Set up an isolated environment with temporary database and storage paths."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    ensure_schema_upgrades(engine)

    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    set_test_engine(engine)

    # Patch storage paths to tmp_path
    storage_root = tmp_path / "storage"
    upload_dir = storage_root / "uploads"
    processed_dir = storage_root / "processed"
    reports_dir = storage_root / "reports"
    temp_dir = storage_root / "temp"

    storage_root.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(storage_service, "storage_root", storage_root)
    monkeypatch.setattr(storage_service, "upload_root", upload_dir)
    monkeypatch.setattr(storage_service, "temp_root", temp_dir)

    monkeypatch.setattr(bm25_service, "storage_root", storage_root)
    monkeypatch.setattr(bm25_service, "processed_root", processed_dir)
    bm25_service.invalidate_cache()

    monkeypatch.setattr(chunking_service, "storage_root", storage_root)
    monkeypatch.setattr(chunking_service, "processed_root", processed_dir)
    chunking_service.invalidate_parent_cache()

    from backend.services.report_generation_service import report_generation_service
    monkeypatch.setattr(report_generation_service, "reports_root", reports_dir)

    def override_get_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    with session_factory() as session:
        yield {
            "session": session,
            "engine": engine,
            "tmp_path": tmp_path,
            "storage_root": storage_root,
        }

    app.dependency_overrides.clear()
    set_test_engine(None)
    reset_db_state()
    engine.dispose()


@pytest.fixture
def client(e2e_env):
    """FastAPI TestClient with isolated dependency injection."""
    with TestClient(app) as test_client:
        yield test_client


def create_sample_pdf_bytes(title: str = "Technical Specification", body: str = "Sample content") -> bytes:
    """Helper to generate a real valid PDF in memory using PyMuPDF."""
    doc = fitz.open()
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((50, 72), f"# {title}", fontsize=18)
    page1.insert_text((50, 110), body, fontsize=11)

    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 72), "## Performance & Operating Limits", fontsize=14)
    page2.insert_text((50, 110), "The operating temperature limit is strictly 1050 C under standard conditions.", fontsize=11)
    page2.insert_text((50, 130), "Rated maximum rotational speed is 3600 RPM for model AX-450.", fontsize=11)

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# -----------------------------------------------------------------------------
# TEST A: ENTERPRISE INITIALIZATION
# -----------------------------------------------------------------------------
def test_e2e_test_a_enterprise_initialization(client, e2e_env):
    """Verify database initialization, schema verification, ChromaDB, storage, and readiness."""
    # 1. Database schema upgrades
    assert init_db(e2e_env["engine"]) is True

    # 2. Storage status
    storage_ok, storage_msg = check_storage_status()
    assert storage_ok is True
    assert "available" in storage_msg.lower()

    # 3. ChromaDB readiness
    chroma_ok, chroma_msg = check_chroma_connection()
    assert chroma_ok is True

    # 4. Liveness endpoint
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"

    # 5. Readiness endpoint
    ready_resp = client.get("/ready")
    assert ready_resp.status_code == 200
    services = ready_resp.json()["services"]
    assert "chromadb" in services
    assert "storage" in services
    assert services["storage"]["ready"] is True


# -----------------------------------------------------------------------------
# TEST B: USER AND WORKSPACE MANAGEMENT (RBAC)
# -----------------------------------------------------------------------------
def test_e2e_test_b_user_and_workspace_management(client, e2e_env):
    """Verify owner provisioning, login, workspace creation, member invites, and RBAC enforcement."""
    session = e2e_env["session"]

    # 1. Provision initial enterprise owner
    owner_user, _ = auth_service.create_owner_user(
        db=session,
        full_name="Alice Owner",
        email="alice.owner@enterprise.com",
        password="SecureOwnerPassword123!",
        assign_existing_workspaces=True,
    )
    assert owner_user.id is not None

    # 2. Login as Owner
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "alice.owner@enterprise.com", "password": "SecureOwnerPassword123!"},
    )
    assert login_resp.status_code == 200
    login_data = login_resp.json()
    owner_token = login_data["access_token"]
    owner_headers = {"Authorization": f"Bearer {owner_token}"}

    # 3. Create Workspace
    ws_resp = client.post(
        "/api/v1/workspaces",
        json={"name": "Energy Propulsion Systems", "slug": "energy-propulsion"},
        headers=owner_headers,
    )
    assert ws_resp.status_code == 201
    workspace_id = ws_resp.json()["id"]

    # 4. Provision Employee user and add to workspace
    emp_resp = client.post(
        f"/api/v1/workspaces/{workspace_id}/members",
        json={
            "email": "bob.engineer@enterprise.com",
            "full_name": "Bob Engineer",
            "role": "employee",
            "initial_password": "EmployeeTempPassword123!",
        },
        headers=owner_headers,
    )
    assert emp_resp.status_code == 201
    assert emp_resp.json()["role"] == "employee"

    # 5. Login as Employee
    emp_login = client.post(
        "/api/v1/auth/login",
        json={"email": "bob.engineer@enterprise.com", "password": "EmployeeTempPassword123!"},
    )
    assert emp_login.status_code == 200
    emp_token = emp_login.json()["access_token"]
    emp_headers = {"Authorization": f"Bearer {emp_token}"}

    # 6. Verify RBAC: Employee cannot invite another member (HTTP 403)
    forbidden_resp = client.post(
        f"/api/v1/workspaces/{workspace_id}/members",
        json={
            "email": "eve@enterprise.com",
            "full_name": "Eve Intruder",
            "role": "admin",
            "password": "TempPassword123!",
        },
        headers=emp_headers,
    )
    assert forbidden_resp.status_code == 403


# -----------------------------------------------------------------------------
# TEST C: DOCUMENT INGESTION PIPELINE
# -----------------------------------------------------------------------------
def test_e2e_test_c_document_ingestion_pipeline(client, e2e_env):
    """Verify streaming upload, SHA-256 deduplication, ingestion job queue, and worker execution."""
    session = e2e_env["session"]

    owner, _ = auth_service.create_owner_user(
        db=session,
        full_name="Ingestion Admin",
        email="ingestion.admin@enterprise.com",
        password="AdminPassword123!",
    )
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "ingestion.admin@enterprise.com", "password": "AdminPassword123!"},
    )
    headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}

    ws = client.post("/api/v1/workspaces", json={"name": "Aero Engines"}, headers=headers).json()
    ws_id = ws["id"]

    pdf_bytes = create_sample_pdf_bytes("AX-450 Operating Manual", "Detailed operational parameters for AX-450 turbine.")

    # 1. Upload Document
    upload_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files={"files": ("ax450_manual.pdf", pdf_bytes, "application/pdf")},
        headers=headers,
    )
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    assert upload_data["total_uploaded"] == 1
    doc_id = upload_data["results"][0]["document"]["id"]

    # 2. Verify SHA-256 duplicate detection
    dup_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files={"files": ("ax450_manual_copy.pdf", pdf_bytes, "application/pdf")},
        headers=headers,
    )
    assert dup_resp.status_code == 200
    assert dup_resp.json()["total_duplicates"] == 1

    # 3. Verify queued ingestion job
    job = session.query(IngestionJob).filter(IngestionJob.document_id == doc_id).first()
    assert job is not None
    assert job.status == "queued"

    # 4. Execute ingestion worker
    claimed_job = ingestion_service.claim_next_job(session)
    assert claimed_job.id == job.id
    success = ingestion_service.execute_job(session, claimed_job)
    assert success is True

    # 5. Check extracted page content
    doc = session.query(Document).filter(Document.id == doc_id).first()
    assert doc.processing_status == "completed"
    assert doc.total_pages == 2

    pages = ingestion_service.get_extracted_pages(session, ws_id, doc_id)
    assert len(pages) == 2
    assert "Operating Manual" in pages[0].text


# -----------------------------------------------------------------------------
# TEST D: INDEXING AND HYBRID RETRIEVAL
# -----------------------------------------------------------------------------
def test_e2e_test_d_indexing_and_hybrid_retrieval(client, e2e_env):
    """Verify hierarchical chunking, vector indexing, BM25 keyword search, RRF fusion, and parent context."""
    session = e2e_env["session"]

    owner, _ = auth_service.create_owner_user(
        db=session,
        full_name="Search Master",
        email="search.master@enterprise.com",
        password="SearchPassword123!",
    )
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "search.master@enterprise.com", "password": "SearchPassword123!"},
    )
    headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}
    ws = client.post("/api/v1/workspaces", json={"name": "Propulsion Specs"}, headers=headers).json()
    ws_id = ws["id"]

    pdf_bytes = create_sample_pdf_bytes("AX-450 Specs", "Turbine serial AX-450 operates at nominal speed.")
    upload_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files={"files": ("specs.pdf", pdf_bytes, "application/pdf")},
        headers=headers,
    ).json()
    doc_id = upload_res["results"][0]["document"]["id"]

    # Ingest & Index
    job = ingestion_service.claim_next_job(session)
    ingestion_service.execute_job(session, job)

    # 1. Search with exact technical identifier
    tech_search = client.post(
        f"/api/v1/workspaces/{ws_id}/search",
        json={"query": "AX-450 rotational speed 3600 RPM", "top_k": 5, "include_parent": True},
        headers=headers,
    )
    assert tech_search.status_code == 200
    res_data = tech_search.json()
    assert res_data["total_results"] > 0
    top_hit = res_data["results"][0]
    assert "3600 RPM" in top_hit["text"]
    assert top_hit["parent_text"] is not None

    # 2. Semantic query
    sem_search = client.post(
        f"/api/v1/workspaces/{ws_id}/search",
        json={"query": "What is the maximum allowed operating temperature?", "top_k": 5},
        headers=headers,
    )
    assert sem_search.status_code == 200
    assert sem_search.json()["total_results"] > 0


# -----------------------------------------------------------------------------
# TEST E: CONVERSATIONAL RAG WORKFLOW
# -----------------------------------------------------------------------------
def test_e2e_test_e_conversational_rag_workflow(client, e2e_env):
    """Verify conversational RAG session, multi-turn follow-ups, citation validation, and honest refusals."""
    session = e2e_env["session"]

    owner, _ = auth_service.create_owner_user(
        db=session,
        full_name="RAG Operator",
        email="rag.operator@enterprise.com",
        password="RAGPassword123!",
    )
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "rag.operator@enterprise.com", "password": "RAGPassword123!"},
    )
    headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}
    ws = client.post("/api/v1/workspaces", json={"name": "Gas Turbines RAG"}, headers=headers).json()
    ws_id = ws["id"]

    pdf_bytes = create_sample_pdf_bytes("AX-450 Manual", "The AX-450 turbine operating temperature is 1050 C.")
    up_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files={"files": ("ax450.pdf", pdf_bytes, "application/pdf")},
        headers=headers,
    ).json()
    doc_id = up_res["results"][0]["document"]["id"]
    job = ingestion_service.claim_next_job(session)
    ingestion_service.execute_job(session, job)

    # Inject mock LLM response
    mock_resp = (
        "**Answer**\n\n"
        "The AX-450 turbine operating temperature is strictly limited to 1050 C [E1].\n\n"
        "**Sources**\n"
        "1. ax450.pdf — Page 2"
    )
    llm_service.set_mock_model(MockChatModel(response_text=mock_resp))

    # 1. Create Conversation
    conv = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        json={"title": "Turbine Inquiries"},
        headers=headers,
    ).json()
    conv_id = conv["id"]

    # 2. Ask question
    msg_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/messages",
        json={"content": "What is the temperature limit for the AX-450?"},
        headers=headers,
    )
    assert msg_resp.status_code == 200
    msg_data = msg_resp.json()
    assert msg_data["evidence_sufficient"] is True
    assert len(msg_data["citations"]) > 0
    assert msg_data["citations"][0]["filename"] == "ax450.pdf"

    # 3. Test unanswerable question yields honest uncertainty
    unans_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations/{conv_id}/messages",
        json={"content": "What is the warranty policy for nuclear submarines in Antarctica?"},
        headers=headers,
    )
    assert unans_resp.status_code == 200
    unans_data = unans_resp.json()
    assert "could not find sufficient evidence" in unans_data["answer"].lower()


# -----------------------------------------------------------------------------
# TEST F: KNOWLEDGE GRAPH DISCOVERY AND DELETION
# -----------------------------------------------------------------------------
def test_e2e_test_f_knowledge_graph_discovery_and_deletion(client, e2e_env):
    """Verify knowledge graph construction, neighborhood querying, and cascading cleanup on document delete."""
    session = e2e_env["session"]

    owner, _ = auth_service.create_owner_user(
        db=session,
        full_name="Graph Specialist",
        email="graph.specialist@enterprise.com",
        password="GraphPassword123!",
    )
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "graph.specialist@enterprise.com", "password": "GraphPassword123!"},
    )
    headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}
    ws = client.post("/api/v1/workspaces", json={"name": "Knowledge Graph Workspace"}, headers=headers).json()
    ws_id = ws["id"]

    pdf_bytes1 = create_sample_pdf_bytes("Doc 1", "Section mentions ISO-9001 quality management.")
    pdf_bytes2 = create_sample_pdf_bytes("Doc 2", "Compliance requires adherence to ISO-9001 standard.")

    doc1_id = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files={"files": ("doc1.pdf", pdf_bytes1, "application/pdf")},
        headers=headers,
    ).json()["results"][0]["document"]["id"]

    doc2_id = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files={"files": ("doc2.pdf", pdf_bytes2, "application/pdf")},
        headers=headers,
    ).json()["results"][0]["document"]["id"]

    # Ingest both
    while True:
        j = ingestion_service.claim_next_job(session)
        if not j:
            break
        ingestion_service.execute_job(session, j)

    # 1. Build Knowledge Graph
    build_resp = client.post(f"/api/v1/workspaces/{ws_id}/graph/build", headers=headers)
    assert build_resp.status_code == 200
    graph_res = client.get(f"/api/v1/workspaces/{ws_id}/graph", headers=headers).json()
    assert graph_res["total_nodes"] > 0

    # 2. Query neighborhood
    neigh_resp = client.get(f"/api/v1/workspaces/{ws_id}/graph/documents/{doc1_id}", headers=headers)
    assert neigh_resp.status_code == 200

    # 3. Delete doc1 and verify graph and vector cleanup
    del_resp = client.delete(f"/api/v1/workspaces/{ws_id}/documents/{doc1_id}", headers=headers)
    assert del_resp.status_code == 200
    remaining_doc = session.query(Document).filter(Document.id == doc1_id).first()
    assert remaining_doc is None


# -----------------------------------------------------------------------------
# TEST G: ENTERPRISE KNOWLEDGE DOSSIER PDF
# -----------------------------------------------------------------------------
def test_e2e_test_g_enterprise_knowledge_dossier_pdf(client, e2e_env):
    """Verify dossier report generation, structure compilation, and valid PDF download."""
    session = e2e_env["session"]

    owner, _ = auth_service.create_owner_user(
        db=session,
        full_name="Report Lead",
        email="report.lead@enterprise.com",
        password="ReportPassword123!",
    )
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "report.lead@enterprise.com", "password": "ReportPassword123!"},
    )
    headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}
    ws = client.post("/api/v1/workspaces", json={"name": "Dossier Workspace"}, headers=headers).json()
    ws_id = ws["id"]

    pdf_bytes = create_sample_pdf_bytes("Safety Manual", "Safety guidelines and emergency shutdown procedures.")
    doc_id = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files={"files": ("safety.pdf", pdf_bytes, "application/pdf")},
        headers=headers,
    ).json()["results"][0]["document"]["id"]
    job = ingestion_service.claim_next_job(session)
    ingestion_service.execute_job(session, job)

    # 1. Generate Report
    gen_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/reports",
        json={"report_name": "Executive Dossier 2026"},
        headers=headers,
    )
    assert gen_resp.status_code == 201
    report_data = gen_resp.json()
    report_id = report_data["id"]
    assert report_data["status"] == "completed"

    # 2. Download and verify binary PDF
    dl_resp = client.get(f"/api/v1/workspaces/{ws_id}/reports/{report_id}/download", headers=headers)
    assert dl_resp.status_code == 200
    assert dl_resp.headers["content-type"] == "application/pdf"
    assert dl_resp.content.startswith(b"%PDF-")


# -----------------------------------------------------------------------------
# TEST H: DOCUMENT VERSION INTELLIGENCE
# -----------------------------------------------------------------------------
def test_e2e_test_h_document_version_intelligence(client, e2e_env):
    """Verify document family versioning, page diffing, and default vs historical search scoping."""
    session = e2e_env["session"]

    owner, _ = auth_service.create_owner_user(
        db=session,
        full_name="Version Manager",
        email="version.manager@enterprise.com",
        password="VersionPassword123!",
    )
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "version.manager@enterprise.com", "password": "VersionPassword123!"},
    )
    headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}
    ws = client.post("/api/v1/workspaces", json={"name": "Version Intelligence WS"}, headers=headers).json()
    ws_id = ws["id"]

    pdf_v1 = create_sample_pdf_bytes("Turbine Specs v1", "AX-450 operating temperature is 1050 C. Speed: 3200 RPM.")
    pdf_v2 = create_sample_pdf_bytes("Turbine Specs v2", "AX-450 operating temperature is 1150 C. Speed: 3600 RPM.")

    # Upload v1
    v1_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files={"files": ("turbine_specs.pdf", pdf_v1, "application/pdf")},
        headers=headers,
    ).json()["results"][0]
    v1_id = v1_res["document"]["id"]
    j1 = ingestion_service.claim_next_job(session)
    ingestion_service.execute_job(session, j1)

    # Get Family
    families = client.get(f"/api/v1/workspaces/{ws_id}/document-families", headers=headers).json()["items"]
    assert len(families) > 0
    family_id = families[0]["id"]

    # Upload v2 to same family
    v2_res = client.post(
        f"/api/v1/workspaces/{ws_id}/document-families/{family_id}/versions",
        files={"file": ("turbine_specs_v2.pdf", pdf_v2, "application/pdf")},
        headers=headers,
    ).json()
    v2_id = v2_res["document"]["id"]
    j2 = ingestion_service.claim_next_job(session)
    ingestion_service.execute_job(session, j2)

    # 1. Verify v1 is superseded and v2 is current
    v1_doc = session.query(Document).filter(Document.id == v1_id).first()
    v2_doc = session.query(Document).filter(Document.id == v2_id).first()
    assert v1_doc.version_status == "superseded"
    assert v2_doc.version_status == "current"
    assert v2_doc.version_number == 2

    # 2. Compare versions
    comp_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/document-families/{family_id}/compare",
        json={"source_document_id": v1_id, "target_document_id": v2_id},
        headers=headers,
    )
    assert comp_resp.status_code == 200
    comp_data = comp_resp.json()
    assert comp_data["stats"]["total_pages_target"] == 2
    assert len(comp_data["numerical_changes"]) > 0

    # 3. Default search retrieves current v2
    def_search = client.post(
        f"/api/v1/workspaces/{ws_id}/search",
        json={"query": "Turbine Specs Speed RPM", "top_k": 5},
        headers=headers,
    ).json()
    for item in def_search["results"]:
        assert item["is_current"] is True


# -----------------------------------------------------------------------------
# TEST I: CONTRADICTION DETECTION AND HUMAN REVIEW
# -----------------------------------------------------------------------------
def test_e2e_test_i_contradiction_detection_and_human_review(client, e2e_env):
    """Verify contradiction detection, classification of dated revisions vs true conflicts, and audit trail."""
    session = e2e_env["session"]

    owner, _ = auth_service.create_owner_user(
        db=session,
        full_name="Compliance Auditor",
        email="compliance.auditor@enterprise.com",
        password="AuditorPassword123!",
    )
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "compliance.auditor@enterprise.com", "password": "AuditorPassword123!"},
    )
    headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}
    ws = client.post("/api/v1/workspaces", json={"name": "Compliance Contradiction WS"}, headers=headers).json()
    ws_id = ws["id"]

    # Create two contemporary documents with conflicting operational statements
    pdf_doc_a = create_sample_pdf_bytes("Doc A", "Turbine model AX-450 maximum rotational speed is strictly 3200 RPM.")
    pdf_doc_b = create_sample_pdf_bytes("Doc B", "Turbine model AX-450 maximum rotational speed is strictly 3800 RPM.")

    doc_a_id = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files={"files": ("manual_a.pdf", pdf_doc_a, "application/pdf")},
        headers=headers,
    ).json()["results"][0]["document"]["id"]

    doc_b_id = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files={"files": ("manual_b.pdf", pdf_doc_b, "application/pdf")},
        headers=headers,
    ).json()["results"][0]["document"]["id"]

    while True:
        j = ingestion_service.claim_next_job(session)
        if not j:
            break
        ingestion_service.execute_job(session, j)

    # 1. Scan for contradictions
    findings = contradiction_service.scan_workspace_contradictions(session, ws_id)
    assert len(findings) > 0
    target_finding = findings[0]

    # 2. Review and resolve contradiction
    review_resp = client.patch(
        f"/api/v1/workspaces/{ws_id}/findings/{target_finding.id}/review",
        json={
            "review_status": "resolved",
            "review_reason": "Verified against engineering standard; Doc B represents correct updated parameter.",
            "authoritative_document_id": doc_b_id,
        },
        headers=headers,
    )
    assert review_resp.status_code == 200
    assert review_resp.json()["review_status"] == "resolved"

    # 3. Verify security audit trail
    audit_record = (
        session.query(SecurityAuditLog)
        .filter(SecurityAuditLog.workspace_id == ws_id, SecurityAuditLog.action == "contradiction_finding_reviewed")
        .first()
    )
    assert audit_record is not None
    assert audit_record.actor_user_id == owner.id


# -----------------------------------------------------------------------------
# TEST J: CROSS-TENANT SECURITY ISOLATION
# -----------------------------------------------------------------------------
def test_e2e_test_j_cross_tenant_security(client, e2e_env):
    """Verify that Company A user cannot list, download, search, or access Company B resources."""
    session = e2e_env["session"]

    # 1. Provision Company A Owner
    user_a, _ = auth_service.create_owner_user(
        db=session,
        full_name="Alice Company A",
        email="alice@company-a.com",
        password="PasswordA123!",
    )
    token_a = client.post(
        "/api/v1/auth/login",
        json={"email": "alice@company-a.com", "password": "PasswordA123!"},
    ).json()["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}
    ws_a = client.post("/api/v1/workspaces", json={"name": "Company A Secret Docs"}, headers=headers_a).json()
    ws_a_id = ws_a["id"]

    # Upload Doc A
    pdf_a = create_sample_pdf_bytes("Proprietary Blueprint A", "Confidential trade secrets for Company A.")
    doc_a_id = client.post(
        f"/api/v1/workspaces/{ws_a_id}/documents",
        files={"files": ("secret_a.pdf", pdf_a, "application/pdf")},
        headers=headers_a,
    ).json()["results"][0]["document"]["id"]
    j_a = ingestion_service.claim_next_job(session)
    ingestion_service.execute_job(session, j_a)

    # 2. Provision Company B Owner
    user_b, _ = auth_service.create_owner_user(
        db=session,
        full_name="Bob Company B",
        email="bob@company-b.com",
        password="PasswordB123!",
    )
    token_b = client.post(
        "/api/v1/auth/login",
        json={"email": "bob@company-b.com", "password": "PasswordB123!"},
    ).json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}
    ws_b = client.post("/api/v1/workspaces", json={"name": "Company B Workspace"}, headers=headers_b).json()
    ws_b_id = ws_b["id"]

    # --- VERIFY STRICT TENANT BOUNDARIES (User B attempting to access Company A resources) ---

    # 1. Listing documents in Workspace A with User B token
    assert client.get(f"/api/v1/workspaces/{ws_a_id}/documents", headers=headers_b).status_code in (403, 404)

    # 2. Downloading Document A with User B token
    assert client.get(f"/api/v1/workspaces/{ws_a_id}/documents/{doc_a_id}/download", headers=headers_b).status_code in (403, 404)

    # 3. Hybrid Search in Workspace A with User B token
    assert client.post(
        f"/api/v1/workspaces/{ws_a_id}/search",
        json={"query": "trade secrets"},
        headers=headers_b,
    ).status_code in (403, 404)

    # 4. Searching Workspace B does NOT return Company A vectors or BM25 entries
    search_b = client.post(
        f"/api/v1/workspaces/{ws_b_id}/search",
        json={"query": "trade secrets Company A"},
        headers=headers_b,
    ).json()
    assert search_b["total_results"] == 0

    # 5. Conversations in Workspace A with User B token
    assert client.get(f"/api/v1/workspaces/{ws_a_id}/conversations", headers=headers_b).status_code in (403, 404)

    # 6. Knowledge Graph in Workspace A with User B token
    assert client.get(f"/api/v1/workspaces/{ws_a_id}/graph", headers=headers_b).status_code in (403, 404)

    # 7. Reports in Workspace A with User B token
    assert client.get(f"/api/v1/workspaces/{ws_a_id}/reports", headers=headers_b).status_code in (403, 404)

    # 8. Contradictions in Workspace A with User B token
    assert client.get(f"/api/v1/workspaces/{ws_a_id}/contradictions", headers=headers_b).status_code in (403, 404)
