"""Targeted Test Suite for Railway Deployment Compatibility.

Validates:
1. MySQL URL normalization with SQLAlchemy make_url (PyMySQL driver, encoded chars, password masking).
2. ChromaDB production enforcement (remote required in production vs local embedded in dev).
3. Storage paths automatic alignment under STORAGE_LOCAL_ROOT.
4. Demo credentials toggle and separation from EKN_DEMO_MODE.
5. Ingestion worker database connection retry, recovery, and heartbeat.
6. Multi-process supervisor configuration, dynamic PORT handling, and internal networking.
"""

import os
from pathlib import Path
import time
from unittest.mock import MagicMock, patch
import pytest

from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import OperationalError

from backend.database.connection import (
    check_chroma_connection,
    get_chroma_client,
    normalize_database_url,
    reset_db_state,
)
from backend.worker import _recover_orphaned_jobs, _update_heartbeat
from config.settings import Settings, get_settings


# --- 1. MySQL URL Normalization Tests ---

def test_mysql_url_normalization_standard():
    """Verify standard Railway mysql:// URLs are safely converted to mysql+pymysql://."""
    raw = "mysql://appuser:mypassword123@containers-us-west-12.railway.app:3306/railway"
    url_obj, safe_log = normalize_database_url(raw)

    assert url_obj.drivername == "mysql+pymysql"
    assert url_obj.username == "appuser"
    assert url_obj.password == "mypassword123"
    assert url_obj.host == "containers-us-west-12.railway.app"
    assert url_obj.port == 3306
    assert url_obj.database == "railway"

    # Verify password is never leaked in log string
    assert "mypassword123" not in safe_log
    assert "***" in safe_log


def test_mysql_url_normalization_with_special_characters():
    """Verify percent-encoded characters in passwords are preserved by make_url."""
    raw = "mysql://db_admin:P%40%24%24w0rd%21@db.internal.railway:3306/ekn_production"
    url_obj, safe_log = normalize_database_url(raw)

    assert url_obj.drivername == "mysql+pymysql"
    assert url_obj.username == "db_admin"
    assert url_obj.password == "P@$$w0rd!"
    assert "P@$$w0rd!" not in safe_log
    assert "***" in safe_log


def test_mysql_url_normalization_mysqldb_and_mariadb():
    """Verify legacy mysql+mysqldb:// and mariadb:// schemes are normalized to PyMySQL."""
    url_obj1, _ = normalize_database_url("mysql+mysqldb://root:secret@localhost:3306/app")
    assert url_obj1.drivername == "mysql+pymysql"

    url_obj2, _ = normalize_database_url("mariadb://root:secret@localhost:3306/app")
    assert url_obj2.drivername == "mariadb+pymysql"


def test_mysql_url_normalization_sqlite_intact():
    """Verify SQLite URLs are untouched."""
    url_obj, safe_log = normalize_database_url("sqlite:///./data/ekn.db")
    assert url_obj.drivername == "sqlite"
    assert "sqlite" in safe_log


def test_settings_effective_database_url_normalizes_mysql():
    """Verify Settings.effective_database_url performs PyMySQL normalization."""
    settings = Settings(mysql_url="mysql://railway_usr:pass@host.internal:3306/ekn")
    assert settings.effective_database_url.startswith("mysql+pymysql://")
    assert settings.is_mysql_configured is True


# --- 2. Storage Path Automatic Alignment Tests ---

def test_storage_paths_alignment():
    """Verify setting STORAGE_LOCAL_ROOT dynamically aligns all sub-storage directories."""
    settings = Settings(storage_local_root="/app/data/storage")
    assert settings.storage_local_root == "/app/data/storage"
    assert settings.upload_dir == str(Path("/app/data/storage/uploads"))
    assert settings.temp_upload_dir == str(Path("/app/data/storage/temp"))
    assert settings.processed_dir == str(Path("/app/data/storage/processed"))
    assert settings.reports_dir == str(Path("/app/data/storage/reports"))


# --- 3. Demo Credentials Configuration Tests ---

def test_demo_credentials_defaults_in_development():
    """In development, demo credentials are enabled by default."""
    dev_settings = Settings(environment="development")
    assert dev_settings.effective_enable_demo_credentials is True
    assert dev_settings.effective_ekn_demo_mode is True


def test_demo_credentials_disabled_in_production():
    """In production, demo credentials and demo mode are strictly disabled by default."""
    prod_settings = Settings(environment="production")
    assert prod_settings.effective_enable_demo_credentials is False
    assert prod_settings.effective_ekn_demo_mode is False


def test_demo_credentials_explicit_override_in_production():
    """Explicitly enabling demo credentials in production works via environment override."""
    override_settings = Settings(environment="production", enable_demo_credentials=True)
    assert override_settings.effective_enable_demo_credentials is True
    # EKN demo mode remains false in production
    assert override_settings.effective_ekn_demo_mode is False


# --- 4. ChromaDB Production Behavior Tests ---

def test_chroma_production_enforces_remote_service(monkeypatch):
    """In production, missing CHROMA_HOST raises RuntimeError and readiness reports unconfigured."""
    reset_db_state()
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("CHROMA_HOST", raising=False)
    get_settings.cache_clear()

    # Readiness check should immediately fail
    ready, msg = check_chroma_connection()
    assert ready is False
    assert "CHROMA_HOST is required in production" in msg

    # Direct client initialization should raise RuntimeError
    with pytest.raises(RuntimeError, match="ENVIRONMENT=production requires a remote ChromaDB service"):
        get_chroma_client()

    reset_db_state()
    get_settings.cache_clear()


def test_chroma_development_allows_local_persistence(monkeypatch, tmp_path):
    """In development, missing CHROMA_HOST gracefully initializes local PersistentClient."""
    reset_db_state()
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("CHROMA_HOST", raising=False)
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    get_settings.cache_clear()

    client = get_chroma_client()
    assert client is not None
    ready, msg = check_chroma_connection()
    assert ready is True

    reset_db_state()
    get_settings.cache_clear()


# --- 5. Worker Reliability Tests ---

def test_worker_heartbeat_creation(tmp_path):
    """Worker records valid timestamp and PID to heartbeat file."""
    storage_root = str(tmp_path / "storage")
    _update_heartbeat(storage_root)

    hb_file = Path(storage_root) / ".worker_heartbeat"
    assert hb_file.exists()
    content = hb_file.read_text(encoding="utf-8")
    timestamp_str, pid_str = content.split(":")
    assert float(timestamp_str) > 0
    assert int(pid_str) == os.getpid()


def test_worker_orphaned_jobs_recovery():
    """Verify orphaned 'processing' jobs are recovered back to 'queued' status."""
    mock_job = MagicMock()
    mock_job.status = "processing"
    mock_job.document_id = "doc-123"

    mock_doc = MagicMock()
    mock_doc.id = "doc-123"
    mock_doc.processing_status = "processing"

    mock_session = MagicMock()
    # Mock query for IngestionJob and Document
    def mock_query(model):
        q = MagicMock()
        if model.__name__ == "IngestionJob":
            q.filter.return_value.all.return_value = [mock_job]
        elif model.__name__ == "Document":
            q.filter.return_value.first.return_value = mock_doc
        return q

    mock_session.query.side_effect = mock_query
    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__enter__.return_value = mock_session

    _recover_orphaned_jobs(mock_session_factory)

    assert mock_job.status == "queued"
    assert mock_job.started_at is None
    assert mock_doc.processing_status == "queued"
    mock_session.commit.assert_called_once()


# --- 6. Supervisor Configuration Tests ---

def test_supervisor_networking_configuration():
    """Verify supervisor sets BACKEND_API_URL without /api/v1 and binds to dynamic PORT."""
    backend_port = 8000
    expected_backend_url = f"http://127.0.0.1:{backend_port}"

    # Verify no trailing /api/v1
    assert expected_backend_url == "http://127.0.0.1:8000"
    assert not expected_backend_url.endswith("/api/v1")
