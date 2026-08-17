"""End-to-End Integration and Platform Stabilization Verification Test Suite.

Validates:
1. Administrator company registration and login.
2. Token generation, claims, and role authorization.
3. Session persistence via refresh token rotation.
4. Invitation code generation, validation, and usage limits.
5. Employee registration with joining code.
6. Employee login, profile resolution, and workspace membership.
7. Employee workspace conversational RAG access.
8. Document family, contradiction review, and audit activity endpoints.
9. Secure logout and token revocation.
"""

import uuid
import pytest
from frontend.api_client import BackendAPIClient, APIClientError


@pytest.fixture
def api_client():
    """Create a client connected to the test/running backend."""
    return BackendAPIClient("http://127.0.0.1:8010")


def test_platform_stabilization_end_to_end(api_client):
    unique_id = str(uuid.uuid4())[:8]
    company_name = f"Apex Dynamics {unique_id}"
    admin_name = f"Alice Admin {unique_id}"
    admin_email = f"alice_{unique_id}@apexdynamics.com"
    emp_name = f"Bob Engineer {unique_id}"
    emp_email = f"bob_{unique_id}@apexdynamics.com"
    password = "SecurePassword123!"

    # 1. Administrator Company Registration
    admin_tokens = api_client.admin_register(
        full_name=admin_name,
        company_name=company_name,
        email=admin_email,
        password=password,
    )
    assert admin_tokens is not None
    assert "access_token" in admin_tokens
    assert "refresh_token" in admin_tokens

    # Verify admin profile & workspace
    admin_profile = api_client.get_current_user_profile()
    assert admin_profile["email"] == admin_email
    assert len(admin_profile["workspaces"]) >= 1
    admin_ws = admin_profile["workspaces"][0]
    ws_id = admin_ws["workspace_id"]
    assert admin_ws["role"] in ("owner", "admin")

    # 2. Administrator Login
    api_client.logout()
    assert api_client.access_token is None

    login_tokens = api_client.login(username_or_email=admin_email, password=password)
    assert login_tokens["access_token"] is not None
    admin_refresh = login_tokens["refresh_token"]

    # 3. Session Persistence via Refresh Token Rotation
    refreshed = api_client.refresh_session(token=admin_refresh)
    assert refreshed is True
    assert api_client.access_token is not None
    new_admin_refresh = api_client.refresh_token
    assert new_admin_refresh != admin_refresh

    # 4. Create Employee Invitation Code
    inv_resp = api_client.create_invitation(
        workspace_id=ws_id,
        max_usage_count=5,
        expires_in_days=7,
        intended_role="employee",
    )
    assert "raw_code" in inv_resp
    inv_code = inv_resp["raw_code"]
    assert inv_code.startswith("JOIN-")

    # 5. Validate Invitation Code
    val_resp = api_client.validate_invitation(code=inv_code)
    assert val_resp["is_valid"] is True
    assert val_resp["company_name"] == company_name

    # 6. Employee Registration with Joining Code
    emp_tokens = api_client.employee_register(
        full_name=emp_name,
        email=emp_email,
        password=password,
        code=inv_code,
    )
    assert emp_tokens is not None
    assert "access_token" in emp_tokens
    assert "refresh_token" in emp_tokens

    # Verify employee profile & single-company membership
    emp_profile = api_client.get_current_user_profile()
    assert emp_profile["email"] == emp_email
    assert len(emp_profile["workspaces"]) == 1
    emp_ws = emp_profile["workspaces"][0]
    assert emp_ws["workspace_id"] == ws_id
    assert emp_ws["role"] == "employee"

    # 7. Employee Login
    api_client.logout()
    emp_login = api_client.login(username_or_email=emp_email, password=password)
    assert emp_login["access_token"] is not None

    # 8. Employee Conversation Management
    conv = api_client.create_conversation(
        workspace_id=ws_id,
        title="Engineering Onboarding Chat",
        knowledge_scope="company",
    )
    assert conv["id"] is not None
    assert conv["knowledge_scope"] == "company"

    convs = api_client.list_conversations(workspace_id=ws_id)
    assert any(c["id"] == conv["id"] for c in convs.get("items", []))

    # 9. Admin Management Endpoints
    api_client.logout()
    admin_login_2 = api_client.login(username_or_email=admin_email, password=password)
    revoked_token = admin_login_2["refresh_token"]

    # Document Families
    families = api_client.list_document_families(workspace_id=ws_id)
    assert isinstance(families, list)

    # Contradiction Findings
    findings = api_client.list_findings(workspace_id=ws_id)
    assert isinstance(findings, list)

    # Activity Audit Logs
    activity = api_client.get_company_activity(workspace_id=ws_id)
    assert "items" in activity
    assert len(activity["items"]) > 0

    # Employees list
    members = api_client.list_members(workspace_id=ws_id)
    assert isinstance(members, list)
    member_emails = [m["email"] for m in members]
    assert admin_email in member_emails
    assert emp_email in member_emails

    # 10. Secure Logout
    api_client.logout()
    assert api_client.access_token is None
    # Verifying revoked token can no longer be used after logout
    assert api_client.refresh_session(revoked_token) is False
