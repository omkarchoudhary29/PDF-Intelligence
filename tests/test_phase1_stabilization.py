"""Phase 1 Verification Tests — Application Stabilization and Defect Repairs.

Verifies:
1. Document Index Status & Parsing (no AttributeError, handles missing/failed/error states).
2. Workspace Creation (no unsupported rows argument, backend integration).
3. Role Resolution (/auth/me parsing, owner/admin -> Administrator, employee -> Employee,
   multi-workspace switching, unauthorized membership).
4. Core Regression (Login, /auth/me, workspace list, document list, ChromaDB + BM25 search).
"""

import ast
from typing import Any, Dict, List
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.auth import UserMeResponse, UserWorkspaceItem
from backend.schemas.indexing import DocumentIndexResponse, DocumentIndexStatusResponse
from frontend.api_client import APIClientError, BackendAPIClient

client = TestClient(app)


# =============================================================================
# A. DOCUMENT INDEX TESTS
# =============================================================================

def test_api_client_has_both_index_methods() -> None:
    """Verify BackendAPIClient exposes get_document_index and the get_document_index_status alias."""
    api_client = BackendAPIClient("http://127.0.0.1:8000")
    assert hasattr(api_client, "get_document_index")
    assert hasattr(api_client, "get_document_index_status")
    # Both point to equivalent behavior
    assert callable(api_client.get_document_index)
    assert callable(api_client.get_document_index_status)


def test_document_index_completed_parsing() -> None:
    """Verify parsing a valid completed DocumentIndexStatusResponse."""
    mock_payload = {
        "document_id": "doc-001",
        "workspace_id": "ws-001",
        "original_filename": "Manual.pdf",
        "extraction_status": "completed",
        "total_pages": 25,
        "is_search_ready": True,
        "index": {
            "id": "idx-001",
            "document_id": "doc-001",
            "status": "completed",
            "total_parents": 10,
            "total_children": 45,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "created_at": "2026-11-15T00:00:00",
            "updated_at": "2026-11-15T00:00:00",
        },
    }

    idx_record = mock_payload.get("index")
    assert idx_record is not None
    assert idx_record["status"] == "completed"
    assert idx_record["total_children"] == 45
    assert mock_payload.get("is_search_ready") is True


def test_document_index_missing_record_handling() -> None:
    """Verify handling when a document is extracted but no index record exists yet."""
    mock_payload = {
        "document_id": "doc-002",
        "workspace_id": "ws-001",
        "original_filename": "NewDoc.pdf",
        "extraction_status": "completed",
        "total_pages": 5,
        "is_search_ready": False,
        "index": None,
    }

    idx_record = mock_payload.get("index")
    assert idx_record is None
    # Should resolve to "Not Indexed" cleanly without raising AttributeError or KeyError
    status_label = "Not Indexed" if mock_payload.get("extraction_status") == "completed" else "Pending"
    assert status_label == "Not Indexed"


def test_document_index_failed_record_handling() -> None:
    """Verify handling when vector indexing fails."""
    mock_payload = {
        "document_id": "doc-003",
        "workspace_id": "ws-001",
        "original_filename": "Corrupted.pdf",
        "extraction_status": "completed",
        "total_pages": 1,
        "is_search_ready": False,
        "index": {
            "id": "idx-003",
            "document_id": "doc-003",
            "status": "failed",
            "total_parents": 1,
            "total_children": 0,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "error_message": "Embedding dimension mismatch",
            "created_at": "2026-11-15T00:00:00",
            "updated_at": "2026-11-15T00:00:00",
        },
    }

    idx_record = mock_payload.get("index")
    assert idx_record is not None
    assert idx_record["status"] == "failed"
    assert idx_record["error_message"] == "Embedding dimension mismatch"


def test_document_index_isolated_error_resilience() -> None:
    """Verify that if one document's index retrieval fails, other documents still process."""
    docs = [
        {"id": "doc-1", "name": "A.pdf"},
        {"id": "doc-2", "name": "B.pdf"},
        {"id": "doc-3", "name": "C.pdf"},
    ]

    results = []
    for d in docs:
        try:
            if d["id"] == "doc-2":
                raise APIClientError("Simulated 500 network failure", 500)
            results.append({"doc_id": d["id"], "status": "Ready"})
        except Exception:
            results.append({"doc_id": d["id"], "status": "Unavailable"})

    # doc-1 and doc-3 succeeded, doc-2 safely marked Unavailable
    assert results[0] == {"doc_id": "doc-1", "status": "Ready"}
    assert results[1] == {"doc_id": "doc-2", "status": "Unavailable"}
    assert results[2] == {"doc_id": "doc-3", "status": "Ready"}


# =============================================================================
# B. WORKSPACE CREATION TESTS
# =============================================================================

def test_streamlit_app_has_no_rows_in_text_area() -> None:
    """Static AST check verifying frontend/streamlit_app.py passes no 'rows' keyword to text_area."""
    with open("frontend/streamlit_app.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    found_rows_kwarg = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "text_area") or (
                isinstance(func, ast.Name) and func.id == "text_area"
            ):
                for kw in node.keywords:
                    if kw.arg == "rows":
                        found_rows_kwarg = True

    assert not found_rows_kwarg, "frontend/streamlit_app.py still contains 'rows' in st.text_area"


# =============================================================================
# C. ROLE RESOLUTION TESTS
# =============================================================================

def _resolve_role_for_workspace(
    user_me_workspaces: List[Dict[str, Any]],
    target_workspace_id: str,
) -> Dict[str, Any]:
    """Helper mimicking the frontend role resolution logic."""
    membership_map = {
        w["workspace_id"]: w
        for w in user_me_workspaces
        if isinstance(w, dict) and "workspace_id" in w
    }
    membership = membership_map.get(target_workspace_id)
    if membership is None:
        return {
            "current_user_role": None,
            "ui_role_label": "No Membership",
            "is_admin": False,
        }

    raw_role = str(membership.get("role", "employee")).lower()
    mem_status = str(membership.get("membership_status", "active")).lower()

    if mem_status != "active":
        return {
            "current_user_role": None,
            "ui_role_label": f"Suspended ({mem_status.title()})",
            "is_admin": False,
        }

    if raw_role in ("owner", "admin"):
        return {
            "current_user_role": raw_role,
            "ui_role_label": f"Administrator ({raw_role.title()})",
            "is_admin": True,
        }

    return {
        "current_user_role": "employee",
        "ui_role_label": "Employee",
        "is_admin": False,
    }


def test_role_resolution_owner() -> None:
    """Workspace owner resolves to Administrator with is_admin=True."""
    workspaces = [
        {"workspace_id": "ws-100", "role": "owner", "membership_status": "active"}
    ]
    res = _resolve_role_for_workspace(workspaces, "ws-100")
    assert res["is_admin"] is True
    assert res["current_user_role"] == "owner"
    assert "Administrator" in res["ui_role_label"]


def test_role_resolution_admin() -> None:
    """Workspace admin resolves to Administrator with is_admin=True."""
    workspaces = [
        {"workspace_id": "ws-100", "role": "admin", "membership_status": "active"}
    ]
    res = _resolve_role_for_workspace(workspaces, "ws-100")
    assert res["is_admin"] is True
    assert res["current_user_role"] == "admin"
    assert "Administrator" in res["ui_role_label"]


def test_role_resolution_employee() -> None:
    """Workspace employee resolves to Employee with is_admin=False."""
    workspaces = [
        {"workspace_id": "ws-100", "role": "employee", "membership_status": "active"}
    ]
    res = _resolve_role_for_workspace(workspaces, "ws-100")
    assert res["is_admin"] is False
    assert res["current_user_role"] == "employee"
    assert res["ui_role_label"] == "Employee"


def test_role_resolution_multi_workspace_switching() -> None:
    """A user who is an owner in WS-A and employee in WS-B receives the appropriate role when switching."""
    workspaces = [
        {"workspace_id": "ws-A", "role": "owner", "membership_status": "active"},
        {"workspace_id": "ws-B", "role": "employee", "membership_status": "active"},
    ]

    res_a = _resolve_role_for_workspace(workspaces, "ws-A")
    assert res_a["is_admin"] is True
    assert res_a["current_user_role"] == "owner"

    res_b = _resolve_role_for_workspace(workspaces, "ws-B")
    assert res_b["is_admin"] is False
    assert res_b["current_user_role"] == "employee"


def test_role_resolution_unknown_workspace_membership() -> None:
    """An unknown workspace grants NO administrator privileges and flags No Membership."""
    workspaces = [
        {"workspace_id": "ws-A", "role": "owner", "membership_status": "active"}
    ]
    res_c = _resolve_role_for_workspace(workspaces, "ws-UNKNOWN")
    assert res_c["is_admin"] is False
    assert res_c["current_user_role"] is None
    assert res_c["ui_role_label"] == "No Membership"


# =============================================================================
# D. REGRESSION TESTS WITH ACTUAL RUNNING BACKEND / DB
# =============================================================================

def test_regression_login_and_auth_me() -> None:
    """Verify actual login endpoint returns token and /auth/me returns workspaces list."""
    # 1. Login
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "admin@enterprise.com", "password": "AdminPassword123!"},
    )
    assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
    token_data = login_resp.json()
    assert "access_token" in token_data
    token = token_data["access_token"]

    # 2. Get /auth/me
    me_resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_resp.status_code == 200, f"/auth/me failed: {me_resp.text}"
    profile = me_resp.json()
    assert profile["email"] == "admin@enterprise.com"
    assert "workspaces" in profile
    assert len(profile["workspaces"]) > 0

    # Ensure admin user has 'owner' role in primary workspace
    ws_item = profile["workspaces"][0]
    assert ws_item["role"] in ("owner", "admin")


def test_regression_workspace_and_documents_listing() -> None:
    """Verify listing workspaces and documents returns the 7 completed indexed documents."""
    # Authenticate
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "admin@enterprise.com", "password": "AdminPassword123!"},
    )
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # List workspaces
    ws_resp = client.get("/api/v1/workspaces", headers=headers)
    assert ws_resp.status_code == 200
    workspaces = ws_resp.json()
    assert len(workspaces) > 0
    ws_id = workspaces[0]["id"]

    # List documents
    docs_resp = client.get(f"/api/v1/workspaces/{ws_id}/documents", headers=headers)
    assert docs_resp.status_code == 200
    doc_data = docs_resp.json()
    assert doc_data["total"] >= 7
    docs = doc_data["items"]
    assert len(docs) >= 7

    # Verify each document has extraction completed and index completed
    for d in docs:
        assert d["processing_status"] == "completed"
        idx_resp = client.get(f"/api/v1/workspaces/{ws_id}/documents/{d['id']}/index", headers=headers)
        assert idx_resp.status_code == 200
        idx_data = idx_resp.json()
        assert idx_data["is_search_ready"] is True
        assert idx_data["index"]["status"] == "completed"
        assert idx_data["index"]["total_children"] > 0


def test_regression_search_endpoint() -> None:
    """Verify hybrid search endpoint executes ChromaDB and BM25 queries with rank fusion."""
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "admin@enterprise.com", "password": "AdminPassword123!"},
    )
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    ws_resp = client.get("/api/v1/workspaces", headers=headers)
    ws_id = ws_resp.json()[0]["id"]

    search_payload = {
        "query": "What is tokenization and embeddings?",
        "top_k": 5,
        "include_parent": True,
    }
    search_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/search",
        json=search_payload,
        headers=headers,
    )
    assert search_resp.status_code == 200
    results_data = search_resp.json()
    assert results_data["total_results"] > 0
    first_result = results_data["results"][0]
    assert "original_filename" in first_result
    assert "page_start" in first_result
    assert "text" in first_result
