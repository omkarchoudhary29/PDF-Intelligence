import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.base import Base
from backend.database.connection import get_db, init_db, reset_db_state, set_test_engine
from backend.main import app


@pytest.fixture
def test_client():
    """Create an isolated in-memory SQLite database and TestClient for workspace tests."""
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
        yield client

    app.dependency_overrides.clear()
    reset_db_state()


def test_create_workspace_success(test_client):
    """Test 1: Workspace creation with explicit name, slug, and description."""
    response = test_client.post(
        "/api/v1/workspaces",
        json={
            "name": "Operations & Safety",
            "slug": "operations-safety",
            "description": "Standard operating manuals and hazard policies",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Operations & Safety"
    assert data["slug"] == "operations-safety"
    assert data["description"] == "Standard operating manuals and hazard policies"
    assert data["is_active"] is True
    assert "id" in data
    assert data["document_count"] == 0


def test_create_workspace_auto_slug(test_client):
    """Test 2: Workspace creation auto-generates slug when omitted."""
    response = test_client.post(
        "/api/v1/workspaces",
        json={"name": "Research & Development Division"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["slug"] == "research-development-division"


def test_create_workspace_empty_name_validation(test_client):
    """Test 3: Workspace validation rejects empty or whitespace-only name."""
    response = test_client.post(
        "/api/v1/workspaces",
        json={"name": "   "},
    )
    assert response.status_code == 422


def test_create_workspace_duplicate_slug_rejection(test_client):
    """Test 4: Workspace validation rejects duplicate slug."""
    test_client.post(
        "/api/v1/workspaces",
        json={"name": "Legal", "slug": "legal-contracts"},
    )
    # Attempt duplicate slug
    dup_resp = test_client.post(
        "/api/v1/workspaces",
        json={"name": "Another Legal", "slug": "legal-contracts"},
    )
    assert dup_resp.status_code == 400
    assert "already exists" in dup_resp.json()["detail"]


def test_list_and_get_workspaces(test_client):
    """Test 5 & 6: Workspace listing and specific workspace retrieval."""
    r1 = test_client.post("/api/v1/workspaces", json={"name": "Workspace Alpha"})
    ws1_id = r1.json()["id"]

    test_client.post("/api/v1/workspaces", json={"name": "Workspace Beta"})

    list_resp = test_client.get("/api/v1/workspaces")
    assert list_resp.status_code == 200
    workspaces = list_resp.json()
    assert len(workspaces) == 2
    assert any(w["name"] == "Workspace Alpha" for w in workspaces)

    # Get specific workspace by ID
    get_resp = test_client.get(f"/api/v1/workspaces/{ws1_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Workspace Alpha"

    # Get non-existent workspace
    not_found = test_client.get("/api/v1/workspaces/00000000-0000-0000-0000-000000000000")
    assert not_found.status_code == 404
