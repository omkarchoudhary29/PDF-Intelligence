import json
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
import pymupdf
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.base import Base
from backend.database.connection import get_db, reset_db_state, set_test_engine
from backend.database.models import Document, IngestionJob, Workspace
from backend.main import app
from backend.services.bm25_service import bm25_service
from backend.services.chunking_service import chunking_service
from backend.services.ingestion_service import ingestion_service
from backend.services.pdf_extractor import pdf_extractor
from backend.services.storage_service import storage_service
from backend.worker import run_worker


def make_test_pdf(file_path: Path, text: str = "Technical Content") -> None:
    """Create a minimal real PDF on disk."""
    if file_path.exists():
        file_path.unlink()
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), text, fontsize=14)
    doc.save(str(file_path))
    doc.close()


@pytest.fixture
def worker_env():
    """Setup isolated SQLite database and storage roots for worker tests."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    storage_root = temp_path / "storage"
    upload_root = storage_root / "uploads"
    processed_root = storage_root / "processed"
    temp_work_root = storage_root / "temp"

    upload_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    temp_work_root.mkdir(parents=True, exist_ok=True)

    # Patch storage and extractor services
    storage_service.storage_root = storage_root
    storage_service.upload_root = upload_root
    storage_service.temp_root = temp_work_root

    pdf_extractor.storage_root = storage_root
    pdf_extractor.processed_root = processed_root
    pdf_extractor.temp_root = temp_work_root

    ingestion_service.storage_root = storage_root
    chunking_service.storage_root = storage_root
    chunking_service.processed_root = processed_root
    bm25_service.storage_root = storage_root
    bm25_service.processed_root = processed_root

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    set_test_engine(engine)

    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    # Create workspace and test documents
    with TestSessionLocal() as session:
        ws = Workspace(
            id=str(uuid.uuid4()),
            name="Worker Testing Workspace",
            slug="worker-test-ws",
        )
        session.add(ws)
        session.commit()
        ws_id = ws.id

    with TestClient(app) as client:
        yield {
            "client": client,
            "session_factory": TestSessionLocal,
            "workspace_id": ws_id,
            "storage_root": storage_root,
            "upload_root": upload_root,
        }

    app.dependency_overrides.clear()
    chunking_service.storage_root = None
    chunking_service.processed_root = None
    bm25_service.storage_root = None
    bm25_service.processed_root = None
    reset_db_state()
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_queued_job_selection_and_atomic_claim(worker_env):
    """Test 13 & 14: Worker safely claims oldest queued job and transitions status to 'processing'."""
    session_factory = worker_env["session_factory"]
    ws_id = worker_env["workspace_id"]
    upload_root = worker_env["upload_root"]

    # Create dummy PDF on disk
    doc_file = upload_root / "claim_test.pdf"
    make_test_pdf(doc_file, "Job claim test")

    with session_factory() as session:
        doc = Document(
            id=str(uuid.uuid4()),
            workspace_id=ws_id,
            original_filename="claim_test.pdf",
            display_name="claim_test",
            storage_key="uploads/claim_test.pdf",
            sha256_hash="hash_claim_123",
            file_size_bytes=doc_file.stat().st_size,
            processing_status="queued",
        )
        session.add(doc)
        job = IngestionJob(document_id=doc.id, status="queued")
        session.add(job)
        session.commit()
        job_id = job.id
        doc_id = doc.id

    # Claim job using ingestion service
    with session_factory() as session:
        claimed_job = ingestion_service.claim_next_job(session)
        assert claimed_job is not None
        assert claimed_job.id == job_id
        assert claimed_job.status == "processing"
        assert claimed_job.started_at is not None

        # Verify document status updated to processing
        d = session.query(Document).filter(Document.id == doc_id).first()
        assert d.processing_status == "processing"


def test_successful_ingestion_job_execution(worker_env):
    """Test 15: Executing a claimed job completes extraction, saves pages, and updates metadata."""
    session_factory = worker_env["session_factory"]
    ws_id = worker_env["workspace_id"]
    upload_root = worker_env["upload_root"]

    doc_file = upload_root / "exec_success.pdf"
    make_test_pdf(doc_file, "Successful processing execution")

    with session_factory() as session:
        doc = Document(
            id=str(uuid.uuid4()),
            workspace_id=ws_id,
            original_filename="exec_success.pdf",
            display_name="exec_success",
            storage_key="uploads/exec_success.pdf",
            sha256_hash="hash_success_456",
            file_size_bytes=doc_file.stat().st_size,
            processing_status="queued",
        )
        session.add(doc)
        job = IngestionJob(document_id=doc.id, status="queued")
        session.add(job)
        session.commit()
        doc_id = doc.id

    # Run worker in --once mode
    processed_count = run_worker(once=True)
    assert processed_count >= 1

    with session_factory() as session:
        doc_updated = session.query(Document).filter(Document.id == doc_id).first()
        assert doc_updated.processing_status == "completed"
        assert doc_updated.total_pages == 1
        assert doc_updated.extracted_pages_path is not None
        assert doc_updated.error_message is None

        job_updated = session.query(IngestionJob).filter(IngestionJob.document_id == doc_id).first()
        assert job_updated.status == "completed"
        assert job_updated.completed_at is not None


def test_failed_job_and_worker_continuation(worker_env):
    """Test 16 & 17: Worker handles a corrupted document failure gracefully and continues processing the next job."""
    session_factory = worker_env["session_factory"]
    ws_id = worker_env["workspace_id"]
    upload_root = worker_env["upload_root"]

    # Doc 1: Corrupted file
    bad_file = upload_root / "broken_job.pdf"
    bad_file.write_bytes(b"%PDF-1.4\nBROKEN_GARBAGE_PAYLOAD")

    # Doc 2: Valid file
    good_file = upload_root / "valid_job.pdf"
    make_test_pdf(good_file, "Valid Document Follow-Up")

    with session_factory() as session:
        doc_bad = Document(
            id=str(uuid.uuid4()),
            workspace_id=ws_id,
            original_filename="broken_job.pdf",
            display_name="broken_job",
            storage_key="uploads/broken_job.pdf",
            sha256_hash="hash_bad_001",
            file_size_bytes=bad_file.stat().st_size,
            processing_status="queued",
        )
        doc_good = Document(
            id=str(uuid.uuid4()),
            workspace_id=ws_id,
            original_filename="valid_job.pdf",
            display_name="valid_job",
            storage_key="uploads/valid_job.pdf",
            sha256_hash="hash_good_002",
            file_size_bytes=good_file.stat().st_size,
            processing_status="queued",
        )
        session.add_all([doc_bad, doc_good])
        session.flush()

        job_bad = IngestionJob(document_id=doc_bad.id, status="queued")
        job_good = IngestionJob(document_id=doc_good.id, status="queued")
        session.add_all([job_bad, job_good])
        session.commit()

        bad_id = doc_bad.id
        good_id = doc_good.id

    # Run worker in --once mode: must process BOTH jobs without crashing!
    processed = run_worker(once=True)
    assert processed == 2

    with session_factory() as session:
        # Verify bad document is marked failed with error message
        d_bad = session.query(Document).filter(Document.id == bad_id).first()
        assert d_bad.processing_status == "failed"
        assert d_bad.error_message is not None
        j_bad = session.query(IngestionJob).filter(IngestionJob.document_id == bad_id).first()
        assert j_bad.status == "failed"

        # Verify good document succeeded (worker continued!)
        d_good = session.query(Document).filter(Document.id == good_id).first()
        assert d_good.processing_status == "completed"
        assert d_good.total_pages == 1
        j_good = session.query(IngestionJob).filter(IngestionJob.document_id == good_id).first()
        assert j_good.status == "completed"


def test_retry_failed_document_and_page_inspection(worker_env):
    """Test 18: Retry endpoint resets failed status to queued, re-processing succeeds, and pages API returns records."""
    client = worker_env["client"]
    session_factory = worker_env["session_factory"]
    ws_id = worker_env["workspace_id"]
    upload_root = worker_env["upload_root"]

    doc_file = upload_root / "retry_doc.pdf"
    # Initially create invalid file so it fails
    doc_file.write_bytes(b"%PDF-1.4\nInitial Bad Data")

    with session_factory() as session:
        doc = Document(
            id=str(uuid.uuid4()),
            workspace_id=ws_id,
            original_filename="retry_doc.pdf",
            display_name="retry_doc",
            storage_key="uploads/retry_doc.pdf",
            sha256_hash="hash_retry_789",
            file_size_bytes=doc_file.stat().st_size,
            processing_status="queued",
        )
        session.add(doc)
        job = IngestionJob(document_id=doc.id, status="queued")
        session.add(job)
        session.commit()
        doc_id = doc.id

    # First worker pass -> fails
    run_worker(once=True)
    with session_factory() as session:
        d = session.query(Document).filter(Document.id == doc_id).first()
        assert d.processing_status == "failed"

    # Now fix the file on disk (simulate user re-supplying good data or fixing issue)
    make_test_pdf(doc_file, "Fixed PDF with valid content")

    # Call API retry endpoint
    retry_resp = client.post(f"/api/v1/workspaces/{ws_id}/documents/{doc_id}/retry")
    assert retry_resp.status_code == 200
    assert retry_resp.json()["processing_status"] == "queued"

    # Second worker pass -> succeeds
    run_worker(once=True)
    with session_factory() as session:
        d2 = session.query(Document).filter(Document.id == doc_id).first()
        assert d2.processing_status == "completed"
        assert d2.total_pages == 1

    # Call API pages endpoint to inspect extracted content
    pages_resp = client.get(f"/api/v1/workspaces/{ws_id}/documents/{doc_id}/pages")
    assert pages_resp.status_code == 200
    details = pages_resp.json()
    assert details["document_id"] == doc_id
    assert details["total_pages"] == 1
    assert len(details["pages"]) == 1
    page1 = details["pages"][0]
    assert page1["page_number"] == 1
    assert "Fixed PDF with valid content" in page1["text"]
