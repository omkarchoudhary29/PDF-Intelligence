"""Unit and integration test suite for Phase 9 Automated Enterprise Knowledge Dossier PDF Generator.

Covers:
1. Report model attributes, defaults, and cascade deletion
2. Document summary and summary_status storage
3. SummarizationService (deterministic extraction, caching, and LLM mocking)
4. Executive repository overview generation
5. PDFComposerService (ReportLab document composition, two-pass NumberedCanvas,
   clickable hyperlinks, and PyMuPDF validation)
6. ReportGenerationService (lifecycle, file persistence, error handling, deletion)
7. FastAPI REST endpoints (create, list, get, download, delete, isolation)
8. Frontend BackendAPIClient report methods
"""

from contextlib import contextmanager
from datetime import datetime
import io
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, PropertyMock, patch
import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database.base import Base
from backend.database.connection import get_db, set_test_engine
from backend.database.models import (
    Document,
    DocumentRelationship,
    Report,
    Workspace,
)
from backend.main import app
from backend.schemas.document import PageExtractionRecord
from backend.schemas.report import (
    DocumentSummaryData,
    ExecutiveSummaryData,
    ReportCreateRequest,
)
from backend.services.llm_service import LLMService, llm_service
from backend.services.pdf_composer_service import PDFComposerService, pdf_composer_service
from backend.services.report_generation_service import (
    ReportGenerationService,
    report_generation_service,
)
from backend.services.summarization_service import (
    SummarizationService,
    summarization_service,
)
from frontend.api_client import BackendAPIClient


@contextmanager
def mock_llm_unconfigured():
    """Ensure LLMService reports unconfigured for deterministic execution."""
    with patch.object(LLMService, "is_configured", new_callable=PropertyMock, return_value=False):
        yield


@contextmanager
def mock_llm_response(payload: dict):
    """Inject a mock model producing structured JSON response."""
    fake_llm_response = MagicMock()
    fake_llm_response.content = json.dumps(payload)
    fake_chat_model = MagicMock()
    fake_chat_model.invoke.return_value = fake_llm_response
    llm_service.set_mock_model(fake_chat_model)
    try:
        yield fake_chat_model
    finally:
        llm_service.reset()


@pytest.fixture
def test_db():
    """Isolated in-memory SQLite database for report tests."""
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
def api_client(test_db):
    """FastAPI TestClient wired to isolated in-memory test database."""
    def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_workspace(test_db):
    """Create test workspace."""
    ws = Workspace(
        name="Aerospace Systems Repository",
        slug="aerospace-systems-repo",
        description="Workspace for testing report dossier generation",
        is_active=True,
    )
    test_db.add(ws)
    test_db.commit()
    test_db.refresh(ws)
    return ws


@pytest.fixture
def sample_documents_with_pages(test_db, sample_workspace, tmp_path, monkeypatch):
    """Create test documents with stored extraction records on disk."""
    from backend.services.ingestion_service import ingestion_service
    from backend.services.storage_service import storage_service

    storage_root = tmp_path / "storage"
    reports_root = tmp_path / "reports"
    monkeypatch.setattr(storage_service, "storage_root", storage_root)
    monkeypatch.setattr(ingestion_service, "storage_root", storage_root)
    monkeypatch.setattr(report_generation_service, "reports_root", reports_root)
    reports_root.mkdir(parents=True, exist_ok=True)

    # Document 1: Turbine Maintenance Manual
    pages1_dir = storage_root / f"workspaces/{sample_workspace.id}/extracted/doc1"
    pages1_dir.mkdir(parents=True, exist_ok=True)
    pages1_file = pages1_dir / "pages.jsonl"

    rec1_p1 = PageExtractionRecord(
        document_id="doc1",
        filename="Turbine_Manual_AX450.pdf",
        page_number=1,
        text="Turbine Operation Manual AX-450. General safety guidelines and maintenance protocols.",
        headings=["1. Safety Protocols"],
        tables=[],
        extraction_method="native_text",
        ocr_used=False,
    )
    rec1_p2 = PageExtractionRecord(
        document_id="doc1",
        filename="Turbine_Manual_AX450.pdf",
        page_number=2,
        text="Turbine Rotor Assembly. Maximum operating temperature is 1200 degrees Celsius.",
        headings=["2. Rotor Specifications"],
        tables=[],
        extraction_method="native_text",
        ocr_used=False,
    )

    with open(pages1_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec1_p1.model_dump()) + "\n")
        f.write(json.dumps(rec1_p2.model_dump()) + "\n")

    doc1 = Document(
        workspace_id=sample_workspace.id,
        original_filename="Turbine_Manual_AX450.pdf",
        display_name="Turbine Manual AX-450",
        storage_key=f"workspaces/{sample_workspace.id}/documents/Turbine_Manual_AX450.pdf",
        file_size_bytes=1024 * 50,
        sha256_hash="hash_doc1_reports",
        mime_type="application/pdf",
        processing_status="completed",
        total_pages=2,
        extracted_pages_path=str(pages1_file.relative_to(storage_root)),
    )
    test_db.add(doc1)

    # Document 2: Safety Policy 2024
    pages2_dir = storage_root / f"workspaces/{sample_workspace.id}/extracted/doc2"
    pages2_dir.mkdir(parents=True, exist_ok=True)
    pages2_file = pages2_dir / "pages.jsonl"

    rec2_p1 = PageExtractionRecord(
        document_id="doc2",
        filename="Safety_Policy_2024.pdf",
        page_number=1,
        text="Enterprise Safety Policy 2024. All personnel operating AX-450 turbines must adhere to OSHA standards.",
        headings=["1. Scope & Compliance"],
        tables=[],
        extraction_method="native_text",
        ocr_used=False,
    )

    with open(pages2_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec2_p1.model_dump()) + "\n")

    doc2 = Document(
        workspace_id=sample_workspace.id,
        original_filename="Safety_Policy_2024.pdf",
        display_name="Safety Policy 2024",
        storage_key=f"workspaces/{sample_workspace.id}/documents/Safety_Policy_2024.pdf",
        file_size_bytes=1024 * 30,
        sha256_hash="hash_doc2_reports",
        mime_type="application/pdf",
        processing_status="completed",
        total_pages=1,
        extracted_pages_path=str(pages2_file.relative_to(storage_root)),
    )
    test_db.add(doc2)
    test_db.commit()
    test_db.refresh(doc1)
    test_db.refresh(doc2)

    # Add a cross-document relationship
    rel = DocumentRelationship(
        workspace_id=sample_workspace.id,
        source_document_id=doc1.id,
        target_document_id=doc2.id,
        relationship_type="shared_entity",
        confidence_score=0.92,
        explanation="Both documents reference AX-450 and safety operating protocols.",
        supporting_evidence={"shared_entity": "AX-450"},
    )
    test_db.add(rel)
    test_db.commit()

    return doc1, doc2, storage_root, reports_root


# =============================================================================
# 1. Model & Database Tests
# =============================================================================

def test_report_model_defaults(test_db, sample_workspace):
    """Verify Report model field defaults and relationships."""
    report = Report(
        workspace_id=sample_workspace.id,
        report_name="Test Dossier",
    )
    test_db.add(report)
    test_db.commit()
    test_db.refresh(report)

    assert report.id is not None
    assert report.report_type in ("knowledge_dossier", "dossier")
    assert report.status == "queued"
    assert report.storage_key is None
    assert report.document_count == 0
    assert report.total_pages_covered == 0
    assert report.file_size_bytes == 0
    assert report.error_message is None
    assert report.metadata_json is None or report.metadata_json == {}
    assert isinstance(report.created_at, datetime)


def test_workspace_cascade_deletes_reports(test_db, sample_workspace):
    """Verify deleting workspace cascades to associated reports."""
    report = Report(
        workspace_id=sample_workspace.id,
        report_name="To be deleted with workspace",
    )
    test_db.add(report)
    test_db.commit()

    report_id = report.id
    test_db.delete(sample_workspace)
    test_db.commit()

    deleted = test_db.query(Report).filter(Report.id == report_id).first()
    assert deleted is None


def test_document_summary_columns(test_db, sample_workspace):
    """Verify Document model supports summary and summary_status columns."""
    doc = Document(
        workspace_id=sample_workspace.id,
        original_filename="manual.pdf",
        display_name="Manual",
        storage_key="manual.pdf",
        file_size_bytes=1000,
        sha256_hash="hash_summary_test",
        mime_type="application/pdf",
        summary=json.dumps({"purpose": "Test Purpose", "summary_text": "Sample summary."}),
        summary_status="completed",
    )
    test_db.add(doc)
    test_db.commit()
    test_db.refresh(doc)

    assert doc.summary_status == "completed"
    parsed = json.loads(doc.summary)
    assert parsed["purpose"] == "Test Purpose"


# =============================================================================
# 2. Summarization Service Tests
# =============================================================================

def test_summarization_service_deterministic(test_db, sample_workspace, sample_documents_with_pages):
    """Verify deterministic document summarization when LLM is unavailable."""
    doc1, _, _, _ = sample_documents_with_pages

    with mock_llm_unconfigured():
        summary = summarization_service.summarize_document(
            db=test_db,
            workspace_id=sample_workspace.id,
            document=doc1,
            force=True,
        )

    assert isinstance(summary, DocumentSummaryData)
    assert summary.document_id == doc1.id
    assert summary.original_filename == "Turbine_Manual_AX450.pdf"
    assert summary.total_pages == 2
    assert "Turbine Manual AX-450" in summary.purpose
    assert summary.summary_status == "completed"
    assert len(summary.key_sections) >= 1
    assert "1. Safety Protocols" in summary.key_sections
    assert 1 in summary.page_references

    # Verify summary was cached in database
    test_db.refresh(doc1)
    assert doc1.summary_status == "completed"
    assert doc1.summary is not None
    stored_data = json.loads(doc1.summary)
    assert stored_data["document_id"] == doc1.id


def test_summarization_service_caching(test_db, sample_workspace, sample_documents_with_pages):
    """Verify summarization reuses cached summary without reading pages again."""
    doc1, _, _, _ = sample_documents_with_pages

    # Populate cache
    cached_payload = {
        "document_id": doc1.id,
        "original_filename": doc1.original_filename,
        "total_pages": 2,
        "purpose": "Cached purpose statement",
        "main_subjects": ["Cached Subject A"],
        "important_findings": ["Finding 1"],
        "key_sections": ["Section 1"],
        "page_references": [1, 2],
        "summary_text": "Cached summary paragraph.",
        "related_documents": ["Safety_Policy_2024.pdf"],
        "summary_status": "completed",
    }
    doc1.summary = json.dumps(cached_payload)
    doc1.summary_status = "completed"
    test_db.commit()

    with patch("backend.services.summarization_service.ingestion_service.get_extracted_pages") as mock_get_pages:
        summary = summarization_service.summarize_document(
            db=test_db,
            workspace_id=sample_workspace.id,
            document=doc1,
            force=False,
        )
        mock_get_pages.assert_not_called()

    assert summary.purpose == "Cached purpose statement"
    assert summary.summary_text == "Cached summary paragraph."
    assert summary.summary_status == "completed"


def test_summarization_service_llm_mode(test_db, sample_workspace, sample_documents_with_pages):
    """Verify LLM summarization with mocked chat model output."""
    doc1, _, _, _ = sample_documents_with_pages

    fake_response = {
        "purpose": "Comprehensive engineering manual for the AX-450 gas turbine.",
        "main_subjects": ["Turbine Assembly", "Thermal Limits"],
        "important_findings": ["Max continuous operating temperature is 1200 C."],
        "summary_text": "This manual outlines operating procedures, thermal limits, and inspection intervals for the AX-450 turbine.",
    }

    with mock_llm_response(fake_response):
        summary = summarization_service.summarize_document(
            db=test_db,
            workspace_id=sample_workspace.id,
            document=doc1,
            force=True,
        )

    assert summary.purpose == "Comprehensive engineering manual for the AX-450 gas turbine."
    assert "Thermal Limits" in summary.main_subjects
    assert len(summary.important_findings) == 1
    assert "1200 C" in summary.important_findings[0]


def test_generate_executive_overview_deterministic(test_db, sample_workspace):
    """Verify deterministic executive repository synthesis."""
    doc_summaries = [
        DocumentSummaryData(
            document_id="doc-1",
            original_filename="Turbine_Manual.pdf",
            total_pages=15,
            purpose="Operation protocols for turbines.",
            main_subjects=["Turbine", "Thermal Limits"],
            important_findings=["Inspect every 500 hours."],
            key_sections=["1. Overview", "2. Maintenance"],
            page_references=[1, 5],
            summary_text="Turbine manual overview.",
            summary_status="completed",
        ),
        DocumentSummaryData(
            document_id="doc-2",
            original_filename="Safety_Guidelines.pdf",
            total_pages=5,
            purpose="Workplace hazard mitigation.",
            main_subjects=["Safety", "Compliance"],
            important_findings=["Wear PPE at all times."],
            key_sections=["1. Scope"],
            page_references=[1],
            summary_text="Safety guidelines overview.",
            summary_status="completed",
        ),
    ]

    with mock_llm_unconfigured():
        exec_data = summarization_service.generate_executive_overview(
            db=test_db,
            workspace_id=sample_workspace.id,
            doc_summaries=doc_summaries,
        )

    assert isinstance(exec_data, ExecutiveSummaryData)
    assert exec_data.document_count == 2
    assert exec_data.total_pages == 20
    assert "consolidates 2 document(s)" in exec_data.overview_text
    assert len(exec_data.key_themes) >= 2


# =============================================================================
# 3. PDF Composer Service & PyMuPDF Validation Tests
# =============================================================================

def test_pdf_composer_service_build_and_validate(tmp_path):
    """Verify ReportLab compiles full multi-page PDF with PyMuPDF validation."""
    pdf_out = tmp_path / "test_dossier.pdf"

    exec_summary = ExecutiveSummaryData(
        overview_text="This enterprise repository consolidates critical operational manuals and regulatory safety policies.",
        key_themes=["Gas Turbine Operations", "OSHA Safety Standards", "Preventive Maintenance"],
        documentation_gaps=["Auxiliary cooling loop manual is missing."],
        document_count=2,
        total_pages=12,
    )

    doc_summaries = [
        DocumentSummaryData(
            document_id="doc-aaa",
            original_filename="Turbine_Manual_AX450.pdf",
            total_pages=8,
            purpose="Technical specification for AX-450 gas turbine operations.",
            main_subjects=["Turbine Blade Inspection", "Thermal Load Limits"],
            important_findings=["Operating temperature must not exceed 1200 C."],
            key_sections=["1. Introduction", "2. Specifications", "3. Troubleshooting"],
            page_references=[1, 3, 5],
            summary_text="Detailed engineering reference guide covering operating cycles and lubrication schedules.",
            related_documents=["Safety_Policy_2024.pdf"],
            summary_status="completed",
        ),
        DocumentSummaryData(
            document_id="doc-bbb",
            original_filename="Safety_Policy_2024.pdf",
            total_pages=4,
            purpose="Mandatory safety standards for high-temperature equipment maintenance.",
            main_subjects=["OSHA Compliance", "Lockout/Tagout"],
            important_findings=["Isolation switches must be padlocked during inspection."],
            key_sections=["1. Purpose", "2. Isolation Protocols"],
            page_references=[1, 2],
            summary_text="Organizational policy on mechanical risk management and personal protective equipment.",
            related_documents=["Turbine_Manual_AX450.pdf"],
            summary_status="completed",
        ),
    ]

    relationships = [
        {
            "relationship_type": "shared_entity",
            "source_document_id": "doc-aaa",
            "target_document_id": "doc-bbb",
            "source_document_name": "Turbine_Manual_AX450.pdf",
            "target_document_name": "Safety_Policy_2024.pdf",
            "confidence_score": 0.95,
            "explanation": "Both documents establish required safety protocols for turbine operators.",
            "supporting_evidence": {"entity": "AX-450"},
        }
    ]

    # Render a real dummy Matplotlib chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "Knowledge Network Test", ha="center", va="center")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    graph_png_bytes = buf.getvalue()

    total_pages = pdf_composer_service.compose_dossier_pdf(
        output_path=pdf_out,
        workspace_name="Aerospace Engineering",
        workspace_id="ws-aerospace-101",
        executive_summary=exec_summary,
        document_summaries=doc_summaries,
        relationships=relationships,
        graph_png_bytes=graph_png_bytes,
        report_title="Aerospace Knowledge Dossier",
    )

    assert pdf_out.is_file()
    assert pdf_out.stat().st_size > 5000
    assert total_pages >= 5

    # PyMuPDF Structural & Hyperlink Verification
    with fitz.open(str(pdf_out)) as doc:
        assert len(doc) == total_pages

        # Check Cover Page (Page 0)
        cover_text = doc[0].get_text()
        assert "Enterprise Knowledge Navigator" in cover_text
        assert "Aerospace Knowledge Dossier" in cover_text
        assert "Aerospace Engineering" in cover_text

        # Verify running footers on pages 2+ ("Page X of Y")
        page_2_text = doc[1].get_text()
        assert f"Page 2 of {total_pages}" in page_2_text
        assert "Confidential — For Internal Enterprise Use Only" in page_2_text

        # Verify presence of clickable hyperlinks across the PDF
        all_links = []
        for page in doc:
            for link in page.get_links():
                if link.get("uri"):
                    all_links.append(link["uri"])

        assert len(all_links) > 0
        doc_links = [l for l in all_links if "document=" in l and "page=" in l]
        assert len(doc_links) >= 2
        assert any("doc-aaa" in l for l in doc_links)
        assert any("doc-bbb" in l for l in doc_links)


# =============================================================================
# 4. Report Generation Service Lifecycle Tests
# =============================================================================

def test_report_generation_lifecycle(test_db, sample_workspace, sample_documents_with_pages):
    """Verify complete report generation workflow from queued to completed."""
    req = ReportCreateRequest(
        report_name="Automated Test Dossier",
        report_type="dossier",
        include_graph=True,
    )

    report = report_generation_service.create_report_job(
        db=test_db,
        workspace_id=sample_workspace.id,
        request=req,
    )
    assert report.status == "queued"
    assert report.report_name == "Automated Test Dossier"

    # Execute generation
    with mock_llm_unconfigured():
        finalized_report = report_generation_service.execute_report_generation(
            db=test_db,
            workspace_id=sample_workspace.id,
            report_id=report.id,
        )

    assert finalized_report.status == "completed"
    assert finalized_report.document_count == 2
    assert finalized_report.total_pages_covered == 3
    assert finalized_report.file_size_bytes > 0
    assert finalized_report.storage_key is not None
    assert finalized_report.generation_completed_at is not None

    # Check physical file
    resolved_path = report_generation_service.get_report_file_path(finalized_report)
    assert resolved_path.is_file()

    # Test report deletion
    deleted = report_generation_service.delete_report(
        db=test_db,
        workspace_id=sample_workspace.id,
        report_id=report.id,
    )
    assert deleted is True
    assert not resolved_path.exists()
    assert test_db.query(Report).filter(Report.id == report.id).first() is None


def test_report_generation_empty_workspace(test_db, sample_workspace):
    """Verify report generation fails gracefully when workspace has no documents."""
    req = ReportCreateRequest(report_name="Empty Workspace Dossier")
    report = report_generation_service.create_report_job(test_db, sample_workspace.id, req)

    finalized = report_generation_service.execute_report_generation(
        db=test_db,
        workspace_id=sample_workspace.id,
        report_id=report.id,
    )

    assert finalized.status == "failed"
    assert "No documents found" in (finalized.error_message or "")


def test_report_generation_error_handling(test_db, sample_workspace, sample_documents_with_pages):
    """Verify unexpected failure in composer flags report as failed."""
    req = ReportCreateRequest(report_name="Failing Dossier")
    report = report_generation_service.create_report_job(test_db, sample_workspace.id, req)

    with patch.object(pdf_composer_service, "compose_dossier_pdf", side_effect=RuntimeError("Composer Disk Crash")):
        finalized = report_generation_service.execute_report_generation(
            db=test_db,
            workspace_id=sample_workspace.id,
            report_id=report.id,
        )

    assert finalized.status == "failed"
    assert "Composer Disk Crash" in (finalized.error_message or "")


# =============================================================================
# 5. FastAPI REST API Endpoint Tests
# =============================================================================

def test_api_create_and_list_reports(api_client, sample_workspace, sample_documents_with_pages):
    """Verify POST /reports creates a dossier and GET /reports lists it."""
    payload = {
        "report_name": "API Generated Dossier",
        "report_type": "dossier",
        "include_graph": False,
    }

    with mock_llm_unconfigured():
        resp = api_client.post(f"/api/v1/workspaces/{sample_workspace.id}/reports", json=payload)

    assert resp.status_code == 201
    data = resp.json()
    assert data["report_name"] == "API Generated Dossier"
    assert data["status"] in ("completed", "partially_completed")
    assert data["document_count"] == 2
    assert data["total_pages_covered"] == 3
    report_id = data["id"]

    # List reports
    list_resp = api_client.get(f"/api/v1/workspaces/{sample_workspace.id}/reports")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] >= 1
    assert any(r["id"] == report_id for r in list_data["items"])


def test_api_get_and_download_report(api_client, sample_workspace, sample_documents_with_pages):
    """Verify GET /reports/{id} details and GET /reports/{id}/download returns PDF."""
    payload = {"report_name": "Downloadable Dossier"}

    with mock_llm_unconfigured():
        create_resp = api_client.post(f"/api/v1/workspaces/{sample_workspace.id}/reports", json=payload)
    assert create_resp.status_code == 201
    report_id = create_resp.json()["id"]

    # GET details
    get_resp = api_client.get(f"/api/v1/workspaces/{sample_workspace.id}/reports/{report_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == report_id

    # GET download
    dl_resp = api_client.get(f"/api/v1/workspaces/{sample_workspace.id}/reports/{report_id}/download")
    assert dl_resp.status_code == 200
    assert dl_resp.headers["content-type"] == "application/pdf"
    assert "attachment;" in dl_resp.headers.get("content-disposition", "")
    assert len(dl_resp.content) > 1000

    # Test PDF can be opened by PyMuPDF
    doc = fitz.open(stream=dl_resp.content, filetype="pdf")
    assert len(doc) >= 4


def test_api_delete_report(api_client, sample_workspace, sample_documents_with_pages):
    """Verify DELETE /reports/{id} removes report and file."""
    with mock_llm_unconfigured():
        create_resp = api_client.post(
            f"/api/v1/workspaces/{sample_workspace.id}/reports",
            json={"report_name": "Temporary Report"},
        )
    report_id = create_resp.json()["id"]

    del_resp = api_client.delete(f"/api/v1/workspaces/{sample_workspace.id}/reports/{report_id}")
    assert del_resp.status_code in (200, 204)

    # Subsequent GET should return 404
    get_resp = api_client.get(f"/api/v1/workspaces/{sample_workspace.id}/reports/{report_id}")
    assert get_resp.status_code == 404


def test_api_workspace_isolation_reports(api_client, test_db, sample_workspace, sample_documents_with_pages):
    """Verify reports cannot be accessed or deleted across foreign workspaces."""
    # Create foreign workspace
    foreign_ws = Workspace(name="Foreign Workspace", slug="foreign-ws", is_active=True)
    test_db.add(foreign_ws)
    test_db.commit()

    with mock_llm_unconfigured():
        create_resp = api_client.post(
            f"/api/v1/workspaces/{sample_workspace.id}/reports",
            json={"report_name": "Isolated Report"},
        )
    report_id = create_resp.json()["id"]

    # Attempt to access through foreign workspace
    cross_get = api_client.get(f"/api/v1/workspaces/{foreign_ws.id}/reports/{report_id}")
    assert cross_get.status_code == 404

    cross_dl = api_client.get(f"/api/v1/workspaces/{foreign_ws.id}/reports/{report_id}/download")
    assert cross_dl.status_code == 404

    cross_del = api_client.delete(f"/api/v1/workspaces/{foreign_ws.id}/reports/{report_id}")
    assert cross_del.status_code == 404


# =============================================================================
# 6. BackendAPIClient Frontend Helper Tests
# =============================================================================

def test_api_client_reports_methods():
    """Verify BackendAPIClient methods for reports."""
    frontend_client = BackendAPIClient("http://test-server:8000")

    # 1. create_report
    mock_create = MagicMock()
    mock_create.status_code = 201
    mock_create.json.return_value = {"id": "rep-1", "report_name": "Test Dossier"}
    with patch("requests.post", return_value=mock_create):
        res = frontend_client.create_report("ws-1", "Test Dossier", include_graph=True)
        assert res["id"] == "rep-1"
        assert res["report_name"] == "Test Dossier"

    # 2. list_reports
    mock_list = MagicMock()
    mock_list.status_code = 200
    mock_list.json.return_value = {"total": 1, "items": [{"id": "rep-1", "report_name": "Test Dossier"}]}
    with patch("requests.get", return_value=mock_list):
        reports = frontend_client.list_reports("ws-1")
        assert reports["total"] == 1
        assert len(reports["items"]) == 1
        assert reports["items"][0]["id"] == "rep-1"

    # 3. get_report
    mock_get = MagicMock()
    mock_get.status_code = 200
    mock_get.json.return_value = {"id": "rep-1", "status": "completed"}
    with patch("requests.get", return_value=mock_get):
        rep = frontend_client.get_report("ws-1", "rep-1")
        assert rep["id"] == "rep-1"
        assert rep["status"] == "completed"

    # 4. download_report
    mock_dl = MagicMock()
    mock_dl.status_code = 200
    mock_dl.content = b"%PDF-1.4 mock binary pdf"
    mock_dl.headers = {"Content-Disposition": 'attachment; filename="Test_Dossier.pdf"'}
    with patch("requests.get", return_value=mock_dl):
        pdf_bytes, filename = frontend_client.download_report("ws-1", "rep-1")
        assert pdf_bytes.startswith(b"%PDF")
        assert filename == "Test_Dossier.pdf"

    # 5. delete_report
    mock_del = MagicMock()
    mock_del.status_code = 200
    mock_del.json.return_value = {"message": "Report deleted successfully."}
    with patch("requests.delete", return_value=mock_del):
        del_result = frontend_client.delete_report("ws-1", "rep-1")
        assert "message" in del_result
