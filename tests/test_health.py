import sys
from pathlib import Path
from unittest.mock import patch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from config.settings import Settings, get_settings

client = TestClient(app)


def test_settings_defaults() -> None:
    """Verify that settings load defaults cleanly and identify unconfigured services."""
    settings = get_settings()
    assert settings.app_name == "Enterprise Knowledge Navigator"
    assert settings.environment == "development"
    assert settings.api_port in (8000, 8010)
    assert not settings.is_mysql_configured


def test_root_index_endpoint() -> None:
    """Verify that root endpoint provides service metadata and navigation endpoints."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Enterprise Knowledge Navigator"
    assert data["health_check"] == "/health"
    assert data["readiness_check"] == "/ready"
    assert data["documentation"] == "/docs"


def test_health_check_endpoint() -> None:
    """Verify GET /health confirms application process is running."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "Enterprise Knowledge Navigator"
    assert "environment" in data


def test_readiness_default_state() -> None:
    """Verify GET /ready reports accurate status when MySQL is unconfigured but ChromaDB initializes."""
    with patch("backend.core.health.check_database_connection", return_value=(False, "Not configured")):
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "mysql" in data["services"]
        assert "chromadb" in data["services"]
        assert data["services"]["mysql"]["ready"] is False
        assert "Not configured" in data["services"]["mysql"]["message"]
        assert data["services"]["chromadb"]["ready"] is True
        assert data["status"] == "degraded_without_mysql"


def test_readiness_with_mysql_connected() -> None:
    """Verify GET /ready reports ready when both MySQL and ChromaDB are operational."""
    with patch("backend.core.health.check_database_connection", return_value=(True, "Connected")):
        with patch("backend.core.health.check_chroma_connection", return_value=(True, "Ready")):
            response = client.get("/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["services"]["mysql"]["ready"] is True
            assert data["services"]["chromadb"]["ready"] is True


def test_readiness_chromadb_failure() -> None:
    """Verify GET /ready returns 503 if ChromaDB initialization fails."""
    with patch("backend.core.health.check_chroma_connection", return_value=(False, "Disk error")):
        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["services"]["chromadb"]["ready"] is False


def test_readiness_mysql_configured_but_failing() -> None:
    """Verify GET /ready returns 503 when MySQL is explicitly configured but connection fails."""
    mock_settings = Settings(mysql_url="mysql+pymysql://user:pass@localhost:3306/db")
    with patch("backend.core.health.get_settings", return_value=mock_settings):
        with patch("backend.core.health.check_database_connection", return_value=(False, "Connection refused")):
            response = client.get("/ready")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "not_ready"
            assert data["services"]["mysql"]["ready"] is False
