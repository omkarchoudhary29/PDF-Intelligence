"""Unit and integration tests for Phase 7 Frontend, API Client, and Backend Extensions.

Tests:
- GET /api/v1/workspaces/{id}/stats
- GET /api/v1/workspaces/{id}/documents/{doc_id}/pages/{page}/preview
- BackendAPIClient methods
- Frontend style utilities and badge helpers
- Frontend view imports and structure
"""

import io
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import fitz
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database.base import Base
from backend.database.connection import get_db, set_test_engine
from backend.database.models import Document, DocumentIndex, IngestionJob, Workspace
from backend.main import app
from frontend.api_client import BackendAPIClient, APIClientError
from frontend.styles import format_bytes, format_date, get_status_badge_html
from frontend.views.overview import render_overview
from frontend.views.documents import render_documents
from frontend.views.chat import render_ask_knowledge_base
from frontend.views.citation_viewer import render_citations


@pytest.fixture
def test_db():
    """Isolated in-memory SQLite database for frontend tests."""
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
    """FastAPI TestClient wired to the isolated test database."""
    def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_workspace(test_db):
    """Create a sample workspace for tests."""
    ws = Workspace(
        name="Frontend Test Workspace",
        slug="frontend-test-ws",
        description="Testing Phase 7 frontend and API extensions",
        is_active=True,
    )
    test_db.add(ws)
    test_db.commit()
    test_db.refresh(ws)
    return ws


@pytest.fixture
def sample_pdf_document(test_db, sample_workspace, tmp_path, monkeypatch):
    """Create a real minimal PDF file and DB document for page preview testing."""
    # Create 2-page PDF using PyMuPDF
    pdf_bytes = io.BytesIO()
    doc = fitz.open()
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((50, 100), "Enterprise Knowledge Navigator — Page 1 Content")
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 100), "Enterprise Knowledge Navigator — Page 2 Content")
    doc.save(pdf_bytes)
    doc.close()
    content = pdf_bytes.getvalue()

    # Save physical file
    from backend.services.storage_service import storage_service
    storage_root = tmp_path / "storage"
    monkeypatch.setattr(storage_service, "storage_root", storage_root)
    workspace_dir = storage_root / "workspaces" / sample_workspace.id / "documents"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    file_path = workspace_dir / "sample.pdf"
    file_path.write_bytes(content)

    doc_record = Document(
        workspace_id=sample_workspace.id,
        original_filename="sample.pdf",
        display_name="sample.pdf",
        storage_key=f"workspaces/{sample_workspace.id}/documents/sample.pdf",
        file_size_bytes=len(content),
        sha256_hash="dummyhash12345",
        mime_type="application/pdf",
        processing_status="completed",
        total_pages=2,
    )
    test_db.add(doc_record)
    test_db.commit()
    test_db.refresh(doc_record)

    # Add index
    index_record = DocumentIndex(
        document_id=doc_record.id,
        status="completed",
        total_parents=2,
        total_children=4,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
    )
    test_db.add(index_record)
    test_db.commit()

    return doc_record


# =========================================================================
# 1. Backend Workspace Stats Tests
# =========================================================================

def test_workspace_stats_success(api_client, sample_workspace, sample_pdf_document):
    """Verify GET /api/v1/workspaces/{workspace_id}/stats returns accurate aggregated counts."""
    resp = api_client.get(f"/api/v1/workspaces/{sample_workspace.id}/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["workspace_id"] == sample_workspace.id
    assert data["total_documents"] == 1
    assert data["indexed_documents"] == 1
    assert data["total_pages"] == 2
    assert data["completed_extraction"] == 1
    assert data["processing_documents"] == 0
    assert data["failed_documents"] == 0


def test_workspace_stats_nonexistent_workspace(api_client):
    """Verify stats returns 404 for an invalid workspace."""
    resp = api_client.get("/api/v1/workspaces/nonexistent-id/stats")
    assert resp.status_code == 404


# =========================================================================
# 2. Backend Page Preview Tests (PyMuPDF)
# =========================================================================

def test_preview_document_page_success(api_client, sample_workspace, sample_pdf_document):
    """Verify GET /pages/{page_number}/preview renders a valid PNG image stream."""
    url = f"/api/v1/workspaces/{sample_workspace.id}/documents/{sample_pdf_document.id}/pages/1/preview"
    resp = api_client.get(url)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    # Check PNG magic bytes
    assert resp.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_preview_document_page_out_of_bounds(api_client, sample_workspace, sample_pdf_document):
    """Verify requesting page > total_pages returns 400 Bad Request."""
    url = f"/api/v1/workspaces/{sample_workspace.id}/documents/{sample_pdf_document.id}/pages/99/preview"
    resp = api_client.get(url)
    assert resp.status_code == 400
    assert "out of bounds" in resp.json()["detail"]


def test_preview_document_page_zero(api_client, sample_workspace, sample_pdf_document):
    """Verify requesting page 0 returns 400 Bad Request."""
    url = f"/api/v1/workspaces/{sample_workspace.id}/documents/{sample_pdf_document.id}/pages/0/preview"
    resp = api_client.get(url)
    assert resp.status_code == 400


def test_preview_document_page_nonexistent_doc(api_client, sample_workspace):
    """Verify preview returns 404 for nonexistent document."""
    url = f"/api/v1/workspaces/{sample_workspace.id}/documents/nonexistent-doc/pages/1/preview"
    resp = api_client.get(url)
    assert resp.status_code == 404


# =========================================================================
# 3. Frontend Formatting & Styling Helpers
# =========================================================================

def test_format_bytes():
    """Verify human-readable byte size representations."""
    assert format_bytes(500) == "500 B"
    assert format_bytes(1024 * 5) == "5.0 KB"
    assert format_bytes(1024 * 1024 * 2) == "2.0 MB"
    assert format_bytes(1024 * 1024 * 1024 * 3) == "3.00 GB"


def test_format_date():
    """Verify clean datetime string formatting."""
    assert format_date("") == "-"
    assert format_date("2026-11-14T20:00:00Z") == "2026-11-14 20:00"
    assert format_date("invalid") == "invalid"


def test_get_status_badge_html():
    """Verify status badge class mapping."""
    html_comp = get_status_badge_html("completed")
    assert "ekn-badge-completed" in html_comp
    assert "completed" in html_comp

    html_failed = get_status_badge_html("failed")
    assert "ekn-badge-failed" in html_failed

    html_proc = get_status_badge_html("processing")
    assert "ekn-badge-processing" in html_proc


# =========================================================================
# 4. BackendAPIClient Unit Tests
# =========================================================================

def test_api_client_workspace_stats():
    """Test BackendAPIClient.get_workspace_stats."""
    api = BackendAPIClient("http://test-server:8000")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "workspace_id": "ws-1",
        "total_documents": 5,
        "indexed_documents": 4,
        "total_pages": 42,
        "completed_extraction": 4,
        "processing_documents": 1,
        "failed_documents": 0,
    }

    with patch("requests.get", return_value=mock_resp) as mock_get:
        stats = api.get_workspace_stats("ws-1")
        assert stats["total_documents"] == 5
        assert stats["indexed_documents"] == 4
        mock_get.assert_called_once_with("http://test-server:8000/api/v1/workspaces/ws-1/stats", headers={}, timeout=30.0)


def test_api_client_preview_page():
    """Test BackendAPIClient.preview_page."""
    api = BackendAPIClient("http://test-server:8000")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"\x89PNG_mock_bytes"

    with patch("requests.get", return_value=mock_resp) as mock_get:
        data = api.preview_page("ws-1", "doc-1", 3, dpi=150)
        assert data == b"\x89PNG_mock_bytes"
        mock_get.assert_called_once()


def test_api_client_conversation_methods():
    """Test BackendAPIClient list_conversations, create_conversation, list_messages, send_message."""
    api = BackendAPIClient("http://test-server:8000")

    # list_conversations
    mock_conv_list = MagicMock()
    mock_conv_list.status_code = 200
    mock_conv_list.json.return_value = {"total": 1, "items": [{"id": "conv-1", "title": "Test"}]}
    with patch("requests.get", return_value=mock_conv_list):
        convs = api.list_conversations("ws-1")
        assert convs["total"] == 1

    # create_conversation
    mock_create = MagicMock()
    mock_create.status_code = 201
    mock_create.json.return_value = {"id": "conv-2", "title": "New"}
    with patch("requests.post", return_value=mock_create):
        created = api.create_conversation("ws-1", title="New")
        assert created["id"] == "conv-2"

    # list_messages
    mock_msg_list = MagicMock()
    mock_msg_list.status_code = 200
    mock_msg_list.json.return_value = {"total": 2, "items": [{"id": "m1", "content": "Hi"}]}
    with patch("requests.get", return_value=mock_msg_list):
        msgs = api.list_messages("ws-1", "conv-2")
        assert msgs["total"] == 2

    # send_message
    mock_send = MagicMock()
    mock_send.status_code = 200
    mock_send.json.return_value = {"answer": "Grounded answer", "citations": []}
    with patch("requests.post", return_value=mock_send):
        rag_resp = api.send_message("ws-1", "conv-2", "Question?")
        assert rag_resp["answer"] == "Grounded answer"


# =========================================================================
# 5. Frontend View Module Verification
# =========================================================================

def test_frontend_views_callable():
    """Verify view functions are callable and properly structured."""
    assert callable(render_overview)
    assert callable(render_documents)
    assert callable(render_ask_knowledge_base)
    assert callable(render_citations)
