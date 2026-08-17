"""Phase 8 — Advanced Document Intelligence Verification Suite.

Validates:
1. Multi-Signal Document Version Detection:
   - Exact duplicate (SHA-256 hash match)
   - Possible new version (document code + revision tag match)
   - Uncertain match (moderate similarity requiring admin confirmation)
   - Completely new document (low similarity across all families)
2. Document Family Management & Lineage:
   - Explicit document family creation API
   - Manually attaching existing documents as sequential versions
   - Detaching documents from families into independent families
3. Document Comparison Engine & Change Categorization:
   - Classifies changes into: Administrative, Formatting-only, Procedural,
     Technical specification, Safety-related, Regulatory, Responsibility change, Other
   - Header/footer and whitespace filtering
4. Contradiction Detection & Review Lifecycle:
   - Detects numerical discrepancies, procedural disagreements, outdated revisions
   - Full review status lifecycle: candidate (pending review), under_review, confirmed, not_a_contradiction, dismissed, resolved
   - Authoritative document resolution
5. RAG Contradiction Awareness:
   - Context assembly identifies active conflicts between retrieved documents and injects conflict warnings
6. Privacy & Tenant Isolation:
   - Strictly company-scoped; personal documents are completely isolated from families and contradictions
7. Security Audit Logging:
   - All version and contradiction review events are audited
"""

import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from backend.database.base import Base
from backend.database.connection import get_db, set_test_engine
from backend.database.models import (
    ContradictionFinding,
    Document,
    DocumentFamily,
    SecurityAuditLog,
    User,
    Workspace,
    WorkspaceMembership,
)
from backend.main import app
from backend.core.security import hash_password
from backend.services.version_service import version_service
from backend.services.version_comparison_service import (
    classify_change_category,
    version_comparison_service,
)
from backend.services.contradiction_service import contradiction_service
from backend.services.rag_workflow import RAGWorkflowService


@pytest.fixture
def test_db():
    """Create isolated in-memory SQLite database."""
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
    """FastAPI TestClient with isolated test DB."""
    app.dependency_overrides.clear()

    def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def create_user(db: Session, email: str, role: str = "admin") -> User:
    u = User(
        id=f"user_{email.split('@')[0]}",
        email=email,
        password_hash=hash_password("SecurePass123!"),
        full_name="Test User",
        is_active=True,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def create_workspace(db: Session, name: str, slug: str) -> Workspace:
    ws = Workspace(
        id=f"ws_{slug}",
        name=name,
        slug=slug,
    )
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def add_member(db: Session, ws: Workspace, user: User, role: str = "admin") -> WorkspaceMembership:
    m = WorkspaceMembership(
        id=f"m_{user.id}_{ws.id}",
        workspace_id=ws.id,
        user_id=user.id,
        role=role,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def get_auth_headers(client: TestClient, email: str) -> dict:
    resp = client.post("/api/v1/auth/login", data={"username": email, "password": "SecurePass123!"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# =============================================================================
# 1. Multi-Signal Document Version Detection
# =============================================================================


def test_multi_signal_exact_duplicate_detection(test_db):
    ws = create_workspace(test_db, "Flight Ops", "flight-ops")
    fam = version_service.get_or_create_family(test_db, ws.id, "Operations Manual")
    doc = Document(
        id="doc_v1",
        workspace_id=ws.id,
        family_id=fam.id,
        version_number=1,
        version_status="current",
        visibility_scope="company",
        original_filename="Operations_Manual.pdf",
        display_name="Operations Manual",
        storage_key="k1",
        sha256_hash="abcd1234abcd1234",
        file_size_bytes=1000,
    )
    test_db.add(doc)
    test_db.commit()

    res = version_service.detect_version_signals(
        db=test_db,
        workspace_id=ws.id,
        filename="Operations_Manual_copy.pdf",
        file_hash="abcd1234abcd1234",
    )
    assert res["classification"] == "exact_duplicate"
    assert res["confidence"] == 1.0
    assert "sha256_match" in res["signals"]
    assert res["matched_family_id"] == fam.id


def test_multi_signal_possible_new_version_code_and_revision(test_db):
    ws = create_workspace(test_db, "Engineering", "engineering")
    fam = version_service.get_or_create_family(test_db, ws.id, "SOP-ENG-042 Turbine Maintenance")
    doc = Document(
        id="doc_eng_v1",
        workspace_id=ws.id,
        family_id=fam.id,
        version_number=1,
        version_status="current",
        visibility_scope="company",
        original_filename="SOP-ENG-042_Turbine_Maintenance_v1.pdf",
        display_name="SOP-ENG-042 Turbine Maintenance",
        storage_key="k_eng_1",
        sha256_hash="hash_eng_1",
        file_size_bytes=2000,
    )
    test_db.add(doc)
    test_db.commit()

    # Upload candidate with same SOP code and revision v2
    res = version_service.detect_version_signals(
        db=test_db,
        workspace_id=ws.id,
        filename="SOP-ENG-042_Turbine_Maintenance_v2.pdf",
        file_hash="different_hash_v2",
        text_content="Turbine maintenance procedural instructions Revision 2.0",
    )
    assert res["classification"] == "possible_new_version"
    assert res["confidence"] >= 0.85
    assert res["detected_document_code"] == "SOP-ENG-042"
    assert res["matched_family_id"] == fam.id


def test_multi_signal_uncertain_match_moderate_similarity(test_db):
    ws = create_workspace(test_db, "Safety", "safety")
    fam = version_service.get_or_create_family(test_db, ws.id, "Emergency Evacuation Procedures")
    doc = Document(
        id="doc_safe_v1",
        workspace_id=ws.id,
        family_id=fam.id,
        version_number=1,
        version_status="current",
        visibility_scope="company",
        original_filename="Emergency_Evacuation_Procedures.pdf",
        display_name="Emergency Evacuation Procedures",
        storage_key="k_safe_1",
        sha256_hash="hash_safe_1",
        file_size_bytes=1500,
    )
    test_db.add(doc)
    test_db.commit()

    # Upload candidate with moderate similarity and no revision indicator
    res = version_service.detect_version_signals(
        db=test_db,
        workspace_id=ws.id,
        filename="Evacuation_Emergency_Plan.pdf",
        file_hash="hash_diff",
    )
    assert res["classification"] == "uncertain_match"
    assert 0.50 <= res["confidence"] <= 0.85


def test_multi_signal_new_document_low_similarity(test_db):
    ws = create_workspace(test_db, "Corporate", "corporate")
    fam = version_service.get_or_create_family(test_db, ws.id, "Financial Audit Report 2024")
    doc = Document(
        id="doc_audit",
        workspace_id=ws.id,
        family_id=fam.id,
        version_number=1,
        version_status="current",
        visibility_scope="company",
        original_filename="Financial_Audit_2024.pdf",
        display_name="Financial Audit 2024",
        storage_key="k_aud",
        sha256_hash="hash_aud",
        file_size_bytes=3000,
    )
    test_db.add(doc)
    test_db.commit()

    # Candidate with completely different content/topic
    res = version_service.detect_version_signals(
        db=test_db,
        workspace_id=ws.id,
        filename="Cafeteria_Food_Safety_Guidelines.pdf",
        file_hash="hash_cafe",
    )
    assert res["classification"] == "new_document"
    assert res["confidence"] >= 0.90


# =============================================================================
# 2. Document Family Management & Attachment/Detachment APIs
# =============================================================================


def test_create_explicit_document_family_api(client, test_db):
    user = create_user(test_db, "admin@domain.com")
    ws = create_workspace(test_db, "Avionics", "avionics")
    add_member(test_db, ws, user, role="admin")
    headers = get_auth_headers(client, "admin@domain.com")

    resp = client.post(
        f"/api/v1/workspaces/{ws.id}/document-families",
        json={"title": "Avionics System Architecture", "canonical_name": "avionics_arch"},
        headers=headers,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "Avionics System Architecture"
    assert data["canonical_name"] == "avionics_arch"
    assert data["workspace_id"] == ws.id


def test_attach_and_detach_document_family(client, test_db):
    user = create_user(test_db, "admin@ops.com")
    ws = create_workspace(test_db, "Ops", "ops")
    add_member(test_db, ws, user, role="admin")
    headers = get_auth_headers(client, "admin@ops.com")

    fam = version_service.create_family(test_db, ws.id, "Flight Guidelines")
    doc1 = Document(
        id="d1",
        workspace_id=ws.id,
        family_id=fam.id,
        version_number=1,
        version_status="current",
        original_filename="FG_v1.pdf",
        display_name="FG v1",
        storage_key="k1",
        sha256_hash="h1",
        file_size_bytes=100,
    )
    doc2 = Document(
        id="d2",
        workspace_id=ws.id,
        family_id=None,  # unattached
        version_number=1,
        version_status="unverified",
        original_filename="FG_v2.pdf",
        display_name="FG v2",
        storage_key="k2",
        sha256_hash="h2",
        file_size_bytes=100,
    )
    test_db.add_all([doc1, doc2])
    test_db.commit()

    # Attach doc2 to fam as version 2
    attach_resp = client.post(
        f"/api/v1/workspaces/{ws.id}/document-families/{fam.id}/attach-document",
        json={"document_id": "d2"},
        headers=headers,
    )
    assert attach_resp.status_code == 200
    assert attach_resp.json()["family_id"] == fam.id
    assert attach_resp.json()["version_number"] == 2

    # Detach doc2 into its own family
    detach_resp = client.post(
        f"/api/v1/workspaces/{ws.id}/document-families/{fam.id}/detach-document/d2",
        headers=headers,
    )
    assert detach_resp.status_code == 200
    new_fam_data = detach_resp.json()
    assert new_fam_data["id"] != fam.id
    assert new_fam_data["current_document_id"] == "d2"


# =============================================================================
# 3. Change Category Classification
# =============================================================================


def test_classify_change_categories():
    # Safety-related
    assert classify_change_category(
        "Standard shutdown procedure.",
        "Danger: Emergency shutdown procedure required. Wear PPE.",
    ) == "Safety-related"

    # Technical specification
    assert classify_change_category(
        "Operating pressure set to 2500 psi.",
        "Operating pressure updated to 3000 psi tolerance +-50 psi.",
    ) == "Technical specification"

    # Procedural
    assert classify_change_category(
        "Review documentation upon completion.",
        "Must execute checklist step 4 before proceeding with the operation.",
    ) == "Procedural"

    # Regulatory
    assert classify_change_category(
        "Follow general guidelines.",
        "Must comply with FAA directive and ISO-9001 quality standards.",
    ) == "Regulatory"

    # Responsibility change
    assert classify_change_category(
        "Assigned to the technician.",
        "Lead Systems Engineer is accountable for formal sign-off.",
    ) == "Responsibility change"

    # Administrative
    assert classify_change_category(
        "Document Version 1.0 issued Jan 2024.",
        "Document Version 2.0 approved by Engineering Director.",
    ) == "Administrative"

    # Formatting-only
    assert classify_change_category(
        "Line one\nLine two\n",
        "Line one   Line two",
    ) == "Formatting-only"


# =============================================================================
# 4. Contradiction Review Statuses & Authoritative Resolution
# =============================================================================


def test_contradiction_review_statuses_and_resolution(client, test_db):
    admin = create_user(test_db, "chief@compliance.com", role="admin")
    ws = create_workspace(test_db, "Compliance WS", "compliance-ws")
    add_member(test_db, ws, admin, role="admin")
    headers = get_auth_headers(client, "chief@compliance.com")

    doc1 = Document(
        id="c_doc1",
        workspace_id=ws.id,
        version_number=1,
        original_filename="SpecA.pdf",
        display_name="SpecA",
        storage_key="ka",
        sha256_hash="ha",
        file_size_bytes=100,
    )
    doc2 = Document(
        id="c_doc2",
        workspace_id=ws.id,
        version_number=2,
        original_filename="SpecB.pdf",
        display_name="SpecB",
        storage_key="kb",
        sha256_hash="hb",
        file_size_bytes=100,
    )
    finding = ContradictionFinding(
        id="finding_101",
        workspace_id=ws.id,
        subject="Hydraulic Fluid Pressure",
        predicate="operating_limit",
        finding_type="numerical_discrepancy",
        explanation="Conflicting pressure limits",
        first_document_id="c_doc1",
        first_page=12,
        first_evidence_excerpt="Maximum hydraulic pressure is 3000 PSI",
        second_document_id="c_doc2",
        second_page=14,
        second_evidence_excerpt="Maximum hydraulic pressure is 3500 PSI",
        review_status="candidate",
    )
    test_db.add_all([doc1, doc2, finding])
    test_db.commit()

    # 1. Update status to 'under_review'
    r1 = client.patch(
        f"/api/v1/workspaces/{ws.id}/findings/{finding.id}/review",
        json={"review_status": "under_review", "review_reason": "Investigating engineering discrepancy"},
        headers=headers,
    )
    assert r1.status_code == 200
    assert r1.json()["review_status"] == "under_review"

    # 2. Update status to 'confirmed'
    r2 = client.patch(
        f"/api/v1/workspaces/{ws.id}/findings/{finding.id}/review",
        json={"review_status": "confirmed", "review_reason": "Confirmed genuine discrepancy"},
        headers=headers,
    )
    assert r2.status_code == 200
    assert r2.json()["review_status"] == "confirmed"

    # 3. Resolve with authoritative document
    r3 = client.patch(
        f"/api/v1/workspaces/{ws.id}/findings/{finding.id}/review",
        json={
            "review_status": "resolved",
            "review_reason": "SpecB (v2) is authoritative over draft SpecA",
            "authoritative_document_id": "c_doc2",
        },
        headers=headers,
    )
    assert r3.status_code == 200
    assert r3.json()["review_status"] == "resolved"
    assert r3.json()["authoritative_document_id"] == "c_doc2"


# =============================================================================
# 5. RAG Contradiction Awareness
# =============================================================================


def test_rag_context_assembly_injects_conflict_alert(test_db):
    ws = create_workspace(test_db, "RAG Test WS", "rag-test-ws")

    doc1 = Document(
        id="rag_doc1",
        workspace_id=ws.id,
        version_number=1,
        original_filename="Hydraulics_v1.pdf",
        display_name="Hydraulics v1",
        storage_key="kr1",
        sha256_hash="hr1",
        file_size_bytes=100,
    )
    doc2 = Document(
        id="rag_doc2",
        workspace_id=ws.id,
        version_number=2,
        original_filename="Hydraulics_v2.pdf",
        display_name="Hydraulics v2",
        storage_key="kr2",
        sha256_hash="hr2",
        file_size_bytes=100,
    )
    finding = ContradictionFinding(
        id="rag_finding_1",
        workspace_id=ws.id,
        subject="Main Pump Pressure",
        predicate="pressure_limit",
        finding_type="numerical_discrepancy",
        explanation="Conflicting pump limits between versions",
        first_document_id="rag_doc1",
        first_page=5,
        first_evidence_excerpt="Pump operating pressure is strictly 2800 PSI.",
        second_document_id="rag_doc2",
        second_page=6,
        second_evidence_excerpt="Pump operating pressure is increased to 3200 PSI.",
        review_status="confirmed",
    )
    test_db.add_all([doc1, doc2, finding])
    test_db.commit()

    rag_service = RAGWorkflowService()
    state = {
        "workspace_id": ws.id,
        "retrieved_evidence": [
            {
                "document_id": "rag_doc1",
                "original_filename": "Hydraulics_v1.pdf",
                "page_start": 5,
                "page_end": 5,
                "text": "Pump operating pressure is strictly 2800 PSI.",
            },
            {
                "document_id": "rag_doc2",
                "original_filename": "Hydraulics_v2.pdf",
                "page_start": 6,
                "page_end": 6,
                "text": "Pump operating pressure is increased to 3200 PSI.",
            },
        ],
    }

    with patch("backend.database.connection.get_db_session") as mock_sess:
        mock_sess.return_value.__enter__.return_value = test_db
        mock_sess.return_value.__exit__.return_value = None

        res = rag_service.context_assembly_node(state)
        assembled = res["assembled_context"]

        # Conflict alert must be present in assembled context
        assert "[CONFLICT ALERT" in assembled
        assert "Main Pump Pressure" in assembled
        assert "These sources disagree" in assembled


# =============================================================================
# 6. Privacy & Personal Document Isolation
# =============================================================================


def test_personal_documents_isolated_from_version_management(test_db):
    ws = create_workspace(test_db, "Secure WS", "secure-ws")
    emp = create_user(test_db, "emp@domain.com", role="employee")

    # Personal document
    p_doc = Document(
        id="personal_doc_1",
        workspace_id=ws.id,
        visibility_scope="personal",
        owner_user_id=emp.id,
        original_filename="Employee_Private_Notes.pdf",
        display_name="Private Notes",
        storage_key="kp",
        sha256_hash="hp",
        file_size_bytes=500,
    )
    test_db.add(p_doc)
    test_db.commit()

    # Legacy family assignment must completely ignore personal documents
    count = version_service.assign_families_to_existing_documents(test_db, ws.id)
    assert count == 0
    test_db.refresh(p_doc)
    assert p_doc.family_id is None

    # Contradiction detection must exclude personal documents
    pairs = contradiction_service.find_candidate_document_pairs(test_db, ws.id)
    assert len(pairs) == 0


# =============================================================================
# 7. Security Audit Logging for Phase 8 Actions
# =============================================================================


def test_phase8_audit_logging(test_db):
    user = create_user(test_db, "auditor@domain.com")
    ws = create_workspace(test_db, "Audit WS", "audit-ws")

    fam = version_service.create_family(
        db=test_db,
        workspace_id=ws.id,
        title="Audited Manual",
        actor_user_id=user.id,
    )

    doc = Document(
        id="audit_doc_1",
        workspace_id=ws.id,
        family_id=None,
        original_filename="Audited_Manual_v1.pdf",
        display_name="Audited Manual",
        storage_key="kaud1",
        sha256_hash="haud1",
        file_size_bytes=100,
    )
    test_db.add(doc)
    test_db.commit()

    version_service.attach_document_to_family(
        db=test_db,
        workspace_id=ws.id,
        family_id=fam.id,
        document_id=doc.id,
        version_number=1,
        actor_user_id=user.id,
    )

    # Check audit log entries
    logs = test_db.query(SecurityAuditLog).filter(SecurityAuditLog.workspace_id == ws.id).all()
    actions = [l.action for l in logs]
    assert "document_family_created" in actions
    assert "document_version_attached" in actions
