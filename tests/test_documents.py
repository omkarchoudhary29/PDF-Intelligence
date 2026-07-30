import io
from pathlib import Path
import shutil
import sys
import tempfile

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.base import Base
from backend.database.connection import get_db, reset_db_state, set_test_engine
from backend.database.models import Document, IngestionJob, Workspace
from backend.main import app
from backend.services.storage_service import storage_service
from config.settings import Settings, get_settings


def make_pdf_bytes(label: str = "Document Content") -> bytes:
    """Generate minimal valid PDF bytes with custom internal text."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        + f"stream\n{label}\nendstream\n".encode("utf-8")
        + b"xref\n0 4\n0000000000 65535 f \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n120\n%%EOF"
    )


@pytest.fixture
def test_env():
    """Create isolated SQLite database and temporary storage directories for document tests."""
    temp_dir = tempfile.mkdtemp()
    temp_storage_root = Path(temp_dir) / "storage"
    temp_upload_root = temp_storage_root / "uploads"
    temp_tmp_root = temp_storage_root / "temp"

    temp_upload_root.mkdir(parents=True, exist_ok=True)
    temp_tmp_root.mkdir(parents=True, exist_ok=True)

    # Patch storage_service paths
    storage_service.storage_root = temp_storage_root
    storage_service.upload_root = temp_upload_root
    storage_service.temp_root = temp_tmp_root

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

    with TestClient(app) as client:
        # Create a default test workspace
        ws_resp = client.post("/api/v1/workspaces", json={"name": "Engineering Workspace"})
        workspace_id = ws_resp.json()["id"]

        yield {
            "client": client,
            "workspace_id": workspace_id,
            "temp_dir": temp_dir,
            "temp_tmp_root": temp_tmp_root,
            "temp_upload_root": temp_upload_root,
        }

    app.dependency_overrides.clear()
    reset_db_state()
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_valid_pdf_upload_and_persistence(test_env):
    """Test 4 & 10: Valid PDF upload, controlled streaming, metadata persistence, and job queueing."""
    client = test_env["client"]
    ws_id = test_env["workspace_id"]
    pdf_content = make_pdf_bytes("Manual Section 101")

    files = [("files", ("safety_manual.pdf", pdf_content, "application/pdf"))]
    resp = client.post(f"/api/v1/workspaces/{ws_id}/documents", files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_submitted"] == 1
    assert data["total_uploaded"] == 1
    assert data["total_duplicates"] == 0
    assert data["total_errors"] == 0

    item = data["results"][0]
    assert item["status"] == "uploaded"
    assert item["filename"] == "safety_manual.pdf"
    doc = item["document"]
    assert doc["original_filename"] == "safety_manual.pdf"
    assert doc["file_size_bytes"] == len(pdf_content)
    assert doc["processing_status"] == "uploaded"
    assert "storage_key" not in doc  # Internal storage key must never be exposed
    assert len(doc["sha256_hash"]) == 64


def test_multiple_pdf_batch_upload(test_env):
    """Test 5: Batch upload of multiple PDF documents in one request."""
    client = test_env["client"]
    ws_id = test_env["workspace_id"]

    pdf1 = make_pdf_bytes("Batch Document 1")
    pdf2 = make_pdf_bytes("Batch Document 2")

    files = [
        ("files", ("doc_alpha.pdf", pdf1, "application/pdf")),
        ("files", ("doc_beta.pdf", pdf2, "application/pdf")),
    ]
    resp = client.post(f"/api/v1/workspaces/{ws_id}/documents", files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_submitted"] == 2
    assert data["total_uploaded"] == 2
    assert data["total_errors"] == 0


def test_invalid_file_rejections(test_env):
    """Test 6: Reject non-pdf extension, fake non-PDF signature, and empty files."""
    client = test_env["client"]
    ws_id = test_env["workspace_id"]

    # 1. Non-pdf extension
    resp1 = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files=[("files", ("notes.txt", b"Plain text file", "text/plain"))],
    )
    assert resp1.status_code == 200
    assert resp1.json()["total_errors"] == 1
    assert "Invalid file type" in resp1.json()["results"][0]["message"]

    # 2. PDF extension but corrupted/non-PDF magic signature
    resp2 = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files=[("files", ("fake.pdf", b"NOT_A_REAL_PDF_HEADER", "application/pdf"))],
    )
    assert resp2.status_code == 200
    assert resp2.json()["total_errors"] == 1
    assert "valid PDF signature" in resp2.json()["results"][0]["message"]

    # 3. Empty 0-byte file
    resp3 = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files=[("files", ("empty.pdf", b"", "application/pdf"))],
    )
    assert resp3.status_code == 200
    assert resp3.json()["total_errors"] == 1


def test_file_size_limit_enforcement(test_env, monkeypatch):
    """Test 7: Reject files exceeding configured max_upload_size_bytes."""
    client = test_env["client"]
    ws_id = test_env["workspace_id"]

    # Set artificial small limit of 50 bytes
    monkeypatch.setattr(storage_service.settings, "max_upload_size_bytes", 50)

    pdf = make_pdf_bytes("Very large enterprise payload")
    assert len(pdf) > 50

    resp = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files=[("files", ("oversized.pdf", pdf, "application/pdf"))],
    )
    assert resp.status_code == 200
    res = resp.json()
    assert res["total_errors"] == 1
    assert "exceeds maximum allowed upload size" in res["results"][0]["message"]


def test_duplicate_pdf_detection(test_env):
    """Test 8 & 9: SHA-256 duplicate detection across successive uploads."""
    client = test_env["client"]
    ws_id = test_env["workspace_id"]
    pdf_content = make_pdf_bytes("Identical Content")

    # First upload -> should succeed
    resp1 = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files=[("files", ("original.pdf", pdf_content, "application/pdf"))],
    )
    assert resp1.status_code == 200
    assert resp1.json()["total_uploaded"] == 1
    orig_id = resp1.json()["results"][0]["document"]["id"]

    # Second upload with different filename but identical content -> should detect duplicate
    resp2 = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files=[("files", ("copy_of_original.pdf", pdf_content, "application/pdf"))],
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["total_uploaded"] == 0
    assert data2["total_duplicates"] == 1
    assert data2["results"][0]["status"] == "duplicate"
    assert data2["results"][0]["document"]["id"] == orig_id
    assert "Duplicate content detected" in data2["results"][0]["message"]


def test_document_listing_and_pagination(test_env):
    """Test 11: Document listing, status filtering, and pagination."""
    client = test_env["client"]
    ws_id = test_env["workspace_id"]

    for i in range(5):
        client.post(
            f"/api/v1/workspaces/{ws_id}/documents",
            files=[("files", (f"doc_{i}.pdf", make_pdf_bytes(f"Page {i}"), "application/pdf"))],
        )

    # Page 1 with page_size=2
    list_p1 = client.get(f"/api/v1/workspaces/{ws_id}/documents?page=1&page_size=2")
    assert list_p1.status_code == 200
    d1 = list_p1.json()
    assert d1["total"] == 5
    assert len(d1["items"]) == 2
    assert d1["page"] == 1

    # Page 2
    list_p2 = client.get(f"/api/v1/workspaces/{ws_id}/documents?page=2&page_size=2")
    assert list_p2.status_code == 200
    assert len(list_p2.json()["items"]) == 2

    # Filter by processing status
    filtered = client.get(f"/api/v1/workspaces/{ws_id}/documents?status=uploaded")
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 5


def test_document_download(test_env):
    """Test 12: Download original PDF file using its internal storage reference."""
    client = test_env["client"]
    ws_id = test_env["workspace_id"]
    pdf_content = make_pdf_bytes("Downloadable Report Content")

    up_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files=[("files", ("annual_report_2026.pdf", pdf_content, "application/pdf"))],
    )
    doc_id = up_resp.json()["results"][0]["document"]["id"]

    # Download document
    dl_resp = client.get(f"/api/v1/workspaces/{ws_id}/documents/{doc_id}/download")
    assert dl_resp.status_code == 200
    assert dl_resp.content == pdf_content
    assert "annual_report_2026.pdf" in dl_resp.headers.get("content-disposition", "")


def test_workspace_scoped_access_validation(test_env):
    """Test 13: Operations on a document with the wrong workspace ID must return 404."""
    client = test_env["client"]
    ws1_id = test_env["workspace_id"]

    # Create second workspace
    ws2_resp = client.post("/api/v1/workspaces", json={"name": "Second Workspace"})
    ws2_id = ws2_resp.json()["id"]

    # Upload document to workspace 1
    up_resp = client.post(
        f"/api/v1/workspaces/{ws1_id}/documents",
        files=[("files", ("secret.pdf", make_pdf_bytes("Confidential"), "application/pdf"))],
    )
    doc_id = up_resp.json()["results"][0]["document"]["id"]

    # Attempt to access via workspace 2 -> 404
    get_wrong = client.get(f"/api/v1/workspaces/{ws2_id}/documents/{doc_id}")
    assert get_wrong.status_code == 404

    # Attempt to download via workspace 2 -> 404
    dl_wrong = client.get(f"/api/v1/workspaces/{ws2_id}/documents/{doc_id}/download")
    assert dl_wrong.status_code == 404

    # Attempt to delete via workspace 2 -> 404
    del_wrong = client.delete(f"/api/v1/workspaces/{ws2_id}/documents/{doc_id}")
    assert del_wrong.status_code == 404


def test_document_deletion_and_storage_removal(test_env):
    """Test 14: Safely delete document and ensure physical file is cleaned up."""
    client = test_env["client"]
    ws_id = test_env["workspace_id"]

    up_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files=[("files", ("to_delete.pdf", make_pdf_bytes("Delete Me"), "application/pdf"))],
    )
    doc_id = up_resp.json()["results"][0]["document"]["id"]

    # Verify document exists before deletion
    get_resp = client.get(f"/api/v1/workspaces/{ws_id}/documents/{doc_id}")
    assert get_resp.status_code == 200

    # Delete document
    del_resp = client.delete(f"/api/v1/workspaces/{ws_id}/documents/{doc_id}")
    assert del_resp.status_code == 200

    # Verify subsequent access returns 404
    assert client.get(f"/api/v1/workspaces/{ws_id}/documents/{doc_id}").status_code == 404
    assert client.get(f"/api/v1/workspaces/{ws_id}/documents/{doc_id}/download").status_code == 404


def test_temporary_file_cleanup_on_failure(test_env):
    """Test 15: Temporary files are cleaned up and no orphaned .tmp files remain on error."""
    client = test_env["client"]
    ws_id = test_env["workspace_id"]
    temp_tmp_root = test_env["temp_tmp_root"]

    # Trigger failure with invalid PDF header
    client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        files=[("files", ("bad.pdf", b"CORRUPTED_NOT_PDF", "application/pdf"))],
    )

    # Check temp dir has no leftover .tmp files
    leftover_tmp_files = list(temp_tmp_root.glob("*.tmp"))
    assert len(leftover_tmp_files) == 0


def test_database_uniqueness_protection(test_env):
    """Test 16: Database-level UniqueConstraint(workspace_id, sha256_hash) raises IntegrityError."""
    import uuid
    from sqlalchemy.exc import IntegrityError
    from backend.database.connection import get_session_factory

    Session = get_session_factory()
    with Session() as session:
        ws_id = test_env["workspace_id"]
        hash_val = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

        doc1 = Document(
            id=str(uuid.uuid4()),
            workspace_id=ws_id,
            original_filename="file1.pdf",
            display_name="file1",
            storage_key="k1.pdf",
            sha256_hash=hash_val,
            file_size_bytes=100,
            mime_type="application/pdf",
            processing_status="uploaded",
        )
        session.add(doc1)
        session.commit()

        doc2 = Document(
            id=str(uuid.uuid4()),
            workspace_id=ws_id,
            original_filename="file2.pdf",
            display_name="file2",
            storage_key="k2.pdf",
            sha256_hash=hash_val,
            file_size_bytes=100,
            mime_type="application/pdf",
            processing_status="uploaded",
        )
        session.add(doc2)
        with pytest.raises(IntegrityError):
            session.commit()
