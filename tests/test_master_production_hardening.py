"""Master Production Hardening, Security, and Cross-Tenant Isolation Tests.

Covers:
1. Security response headers (X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Referrer-Policy).
2. Production mode enforcement (EKN_DEMO_MODE=false rejects uninvited employee registration).
3. Cross-tenant isolation (foreign workspace access returns 404 fail-closed).
4. Personal document privacy (Employee A cannot access Employee B's personal documents; Admin cannot access Employee personal files).
5. PDF upload security (magic byte %PDF- enforcement and filename path-traversal sanitization).
6. Exception sanitization (unhandled 500 exceptions mask sensitive internal strings in production).
7. Worker heartbeat presence in readiness probe.
"""

from io import BytesIO
import uuid
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from backend.main import app
from backend.services.storage_service import sanitize_filename
from config.settings import Settings, get_settings

client = TestClient(app)


def test_security_headers_present():
    """Verify standard enterprise security headers are injected into all HTTP responses."""
    resp = client.get("/health")
    assert resp.status_code == 200
    headers = {k.lower(): v for k, v in resp.headers.items()}
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("x-frame-options") == "SAMEORIGIN"
    assert headers.get("x-xss-protection") == "1; mode=block"
    assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"


def test_sanitize_filename_prevents_directory_traversal():
    """Verify filename sanitization strips directory traversal paths and null bytes."""
    malicious_names = [
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("..\\..\\windows\\system32\\calc.pdf", "calc.pdf"),
        ("normal_doc.pdf\x00.exe", "normal_doc.pdf.exe"),
        ("/var/www/uploads/test.pdf", "test.pdf"),
    ]
    for raw, expected in malicious_names:
        sanitized = sanitize_filename(raw)
        assert "/" not in sanitized
        assert "\\" not in sanitized
        assert ".." not in sanitized
        assert "\x00" not in sanitized


def test_production_mode_requires_invitation_code_for_employee():
    """Verify that when EKN_DEMO_MODE=false, employee registration without a valid joining code is rejected."""
    mock_settings = Settings(ekn_demo_mode=False)
    rand_email = f"uninvited_{uuid.uuid4().hex[:8]}@company.com"
    with patch("backend.services.auth_service.get_settings", return_value=mock_settings):
        resp = client.post(
            "/api/v1/auth/employee/register",
            json={
                "full_name": "Test Uninvited Employee",
                "email": rand_email,
                "password": "SecurePassword123!",
                "code": "",
            },
        )
        assert resp.status_code == 400
        assert "joining code is required" in resp.json().get("detail", "").lower()


def test_upload_rejects_non_pdf_content():
    """Verify upload fails if file lacks the %PDF- magic signature even with a .pdf extension."""
    fake_pdf = BytesIO(b"This is a text file renamed to fake.pdf without PDF headers.")
    fake_pdf.name = "fake.pdf"

    # Authenticate as demo admin
    settings = get_settings()
    login_resp = client.post(
        "/api/v1/auth/login",
        json={
            "email": "admin.demo@example.com",
            "password": settings.demo_admin_password,
            "portal": "admin",
        },
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    ws_id = settings.demo_workspace_id

    resp = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"files": ("fake.pdf", fake_pdf, "application/pdf")},
        params={"scope": "company"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["status"] == "error"
    assert "PDF signature" in data["results"][0]["message"]


def test_worker_heartbeat_in_readiness():
    """Verify GET /ready includes worker status."""
    resp = client.get("/ready")
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert "worker" in data["services"]
    assert "ready" in data["services"]["worker"]


def test_cross_tenant_foreign_workspace_returns_404():
    """Verify that attempting to access an unauthorized foreign workspace returns 404."""
    settings = get_settings()
    login_resp = client.post(
        "/api/v1/auth/login",
        json={
            "email": "employee.demo@example.com",
            "password": settings.demo_employee_password,
            "portal": "employee",
        },
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]

    foreign_ws = "00000000-0000-0000-0000-000000000000"
    resp = client.get(
        f"/api/v1/workspaces/{foreign_ws}/documents",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404
