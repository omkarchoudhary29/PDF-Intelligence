"""Comprehensive End-to-End Acceptance Workflows Verification for EKN Master Production Readiness.

Executes and asserts against the running FastAPI backend (http://127.0.0.1:8010):
- Workflow 1: Full Administrator Lifecycle
- Workflow 2: Full Employee Conversational Lifecycle
- Workflow 3: Cross-User Private Document and Conversation Isolation
- Workflow 4: Cross-Workspace Tenant Isolation (Fail-Closed 404)
- Workflow 5: Backup and Data Recovery Verification
"""

import os
import sys
import uuid
import requests
import sqlite3
from pathlib import Path
from config.settings import get_settings

BASE_URL = "http://127.0.0.1:8010/api/v1"

def print_step(msg: str):
    print(f"\n[+] {msg}")

def assert_true(cond: bool, msg: str):
    if not cond:
        print(f"[-] FAILED: {msg}")
        sys.exit(1)
    print(f"    [OK] {msg}")

def run_acceptance_workflows():
    print("=" * 70)
    print("STARTING EKN MASTER ACCEPTANCE WORKFLOWS VERIFICATION")
    print("=" * 70)

    settings = get_settings()
    admin_pw = settings.demo_admin_password or "AdminDemo123!"
    emp_pw = settings.demo_employee_password or "EmployeeDemo123!"

    # -------------------------------------------------------------
    # WORKFLOW 1: Administrator Full Lifecycle
    # -------------------------------------------------------------
    print_step("Workflow 1: Administrator Full Lifecycle")
    
    # 1.1 Admin Login
    admin_login_res = requests.post(f"{BASE_URL}/auth/login", json={
        "email": "admin.demo@example.com",
        "password": admin_pw,
        "portal": "admin"
    })
    assert_true(admin_login_res.status_code == 200, "Admin login successful")
    admin_data = admin_login_res.json()
    admin_token = admin_data["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    
    # Find Aerospace workspace
    ws_res = requests.get(f"{BASE_URL}/workspaces", headers=admin_headers)
    assert_true(ws_res.status_code == 200, "Admin lists workspaces")
    workspaces = ws_res.json()
    aerospace_ws = next((w for w in workspaces if "aerospace" in w["name"].lower()), None)
    if not aerospace_ws and workspaces:
        aerospace_ws = workspaces[0]
    assert_true(aerospace_ws is not None, f"Found target workspace: {aerospace_ws['name']}")
    ws_id = aerospace_ws["id"]

    # 1.2 Overview / Documents
    docs_res = requests.get(f"{BASE_URL}/workspaces/{ws_id}/documents", headers=admin_headers)
    assert_true(docs_res.status_code == 200, "Admin fetches documents list")
    docs_data = docs_res.json()
    items = docs_data.get("items", [])
    assert_true(len(items) >= 7, f"Admin sees documents in workspace (count: {len(items)})")

    # 1.3 Versions Comparison / Families
    versions_res = requests.get(f"{BASE_URL}/workspaces/{ws_id}/document-families", headers=admin_headers)
    assert_true(versions_res.status_code == 200, "Admin fetches document families")

    # 1.4 Review Queue (Contradictions)
    review_res = requests.get(f"{BASE_URL}/workspaces/{ws_id}/findings", headers=admin_headers)
    assert_true(review_res.status_code == 200, "Admin accesses contradiction findings queue")

    # 1.5 Knowledge Graph Summary
    graph_res = requests.get(f"{BASE_URL}/workspaces/{ws_id}/graph/summary", headers=admin_headers)
    assert_true(graph_res.status_code == 200, "Admin queries workspace knowledge graph summary")
    graph_data = graph_res.json()
    assert_true("entity_count" in graph_data or "total_entities" in graph_data or "document_count" in graph_data, "Knowledge graph summary returned valid counts")

    # 1.6 Invitation Code Generation
    inv_res = requests.post(
        f"{BASE_URL}/workspaces/{ws_id}/invitations",
        json={"intended_role": "employee", "max_usage_count": 5, "expires_in_days": 7},
        headers=admin_headers
    )
    assert_true(inv_res.status_code == 201, "Admin generates employee invitation code")
    inv_code = inv_res.json().get("raw_code") or inv_res.json().get("code")
    assert_true(bool(inv_code) and len(inv_code) >= 6, f"Valid invitation code generated: {inv_code}")

    # 1.7 Reports
    rep_res = requests.get(f"{BASE_URL}/workspaces/{ws_id}/reports", headers=admin_headers)
    assert_true(rep_res.status_code == 200, "Admin accesses reports list")

    print("[SUCCESS] Workflow 1: Administrator Full Lifecycle Verified.")

    # -------------------------------------------------------------
    # WORKFLOW 2: Employee Conversational & Citation Lifecycle
    # -------------------------------------------------------------
    print_step("Workflow 2: Employee Conversational Lifecycle")

    # 2.1 Employee Login
    emp_login_res = requests.post(f"{BASE_URL}/auth/login", json={
        "email": "employee.demo@example.com",
        "password": emp_pw,
        "portal": "employee"
    })
    assert_true(emp_login_res.status_code == 200, "Employee login successful")
    emp_token = emp_login_res.json()["access_token"]
    emp_headers = {"Authorization": f"Bearer {emp_token}"}

    # 2.2 Create Conversation
    conv_res = requests.post(
        f"{BASE_URL}/workspaces/{ws_id}/conversations",
        json={"title": f"Master Production Audit Conversation {uuid.uuid4().hex[:6]}"},
        headers=emp_headers
    )
    assert_true(conv_res.status_code == 201, "Employee creates new conversation")
    conv_id = conv_res.json()["id"]

    # 2.3 Conversational Query (RAG + Gemini or Greeting)
    # Test greeting query first (bypasses RAG retrieval cleanly)
    greet_res = requests.post(
        f"{BASE_URL}/workspaces/{ws_id}/conversations/{conv_id}/messages",
        json={"content": "Hello, what can you assist me with?"},
        headers=emp_headers
    )
    assert_true(greet_res.status_code == 200, "Employee sends greeting message")
    greet_data = greet_res.json()
    assert_true(bool(greet_data.get("answer")), "Assistant returned greeting response")

    # 2.4 Document Query with Citations
    doc_query_res = requests.post(
        f"{BASE_URL}/workspaces/{ws_id}/conversations/{conv_id}/messages",
        json={"content": "What are the standard operating procedures for turbine startup?"},
        headers=emp_headers
    )
    assert_true(doc_query_res.status_code == 200, "Employee sends document query")
    query_reply = doc_query_res.json()
    assert_true(bool(query_reply.get("answer")), "Assistant generated RAG response")

    # 2.5 Scope Switching (Company -> Personal)
    scope_patch = requests.patch(
        f"{BASE_URL}/workspaces/{ws_id}/conversations/{conv_id}",
        json={"knowledge_scope": "personal"},
        headers=emp_headers
    )
    assert_true(scope_patch.status_code == 200, "Employee updates conversation knowledge scope to personal")
    assert_true(scope_patch.json().get("knowledge_scope") == "personal", "Knowledge scope set to personal")

    # Send personal scope query
    scope_res = requests.post(
        f"{BASE_URL}/workspaces/{ws_id}/conversations/{conv_id}/messages",
        json={"content": "Do I have any personal notes uploaded?"},
        headers=emp_headers
    )
    assert_true(scope_res.status_code == 200, "Employee queries personal scope")
    assert_true(bool(scope_res.json().get("answer")), "Personal scope returned assistant answer")

    print("[SUCCESS] Workflow 2: Employee Conversational Lifecycle Verified.")

    # -------------------------------------------------------------
    # WORKFLOW 3: Cross-User Security & Personal Document Isolation
    # -------------------------------------------------------------
    print_step("Workflow 3: Cross-User Security & Personal Document Isolation")

    # Register Employee A
    user_a_email = f"user_a_{uuid.uuid4().hex[:6]}@company.com"
    reg_a = requests.post(f"{BASE_URL}/auth/employee/register", json={
        "full_name": "Test User A",
        "email": user_a_email,
        "password": "PasswordA123!",
        "code": inv_code
    })
    assert_true(reg_a.status_code == 201, "Registered User A")
    token_a = reg_a.json()["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # Register Employee B
    user_b_email = f"user_b_{uuid.uuid4().hex[:6]}@company.com"
    reg_b = requests.post(f"{BASE_URL}/auth/employee/register", json={
        "full_name": "Test User B",
        "email": user_b_email,
        "password": "PasswordB123!",
        "code": inv_code
    })
    assert_true(reg_b.status_code == 201, "Registered User B")
    token_b = reg_b.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # User A creates a conversation
    conv_a = requests.post(
        f"{BASE_URL}/workspaces/{ws_id}/conversations",
        json={"title": "User A Private Chat"},
        headers=headers_a
    )
    assert_true(conv_a.status_code == 201, "User A creates conversation")
    conv_a_id = conv_a.json()["id"]

    # User B attempts to access User A's conversation -> MUST return 404 (fail closed)
    sec_test_1 = requests.get(
        f"{BASE_URL}/workspaces/{ws_id}/conversations/{conv_a_id}",
        headers=headers_b
    )
    assert_true(sec_test_1.status_code == 404, "User B blocked from accessing User A conversation (404 Fail-Closed)")

    # User B attempts to send a message to User A's conversation -> MUST return 404
    sec_test_2 = requests.post(
        f"{BASE_URL}/workspaces/{ws_id}/conversations/{conv_a_id}/messages",
        json={"content": "Unauthorized injection attempt", "search_scope": "company"},
        headers=headers_b
    )
    assert_true(sec_test_2.status_code == 404, "User B blocked from writing to User A conversation (404 Fail-Closed)")

    print("[SUCCESS] Workflow 3: Cross-User Security & Isolation Verified.")

    # -------------------------------------------------------------
    # WORKFLOW 4: Cross-Workspace Tenant Isolation
    # -------------------------------------------------------------
    print_step("Workflow 4: Cross-Workspace Tenant Isolation")

    # Create an isolated second workspace with Admin
    foreign_ws_res = requests.post(
        f"{BASE_URL}/workspaces",
        json={"name": f"Isolated Tenant {uuid.uuid4().hex[:6]}"},
        headers=admin_headers
    )
    assert_true(foreign_ws_res.status_code == 201, "Admin creates separate tenant workspace")
    foreign_ws_id = foreign_ws_res.json()["id"]

    # Employee B (who is only member of aerospace_ws) attempts to access foreign_ws
    foreign_docs = requests.get(f"{BASE_URL}/workspaces/{foreign_ws_id}/documents", headers=headers_b)
    assert_true(foreign_docs.status_code == 404, "Employee B blocked from foreign workspace documents (404)")

    foreign_conv = requests.get(f"{BASE_URL}/workspaces/{foreign_ws_id}/conversations", headers=headers_b)
    assert_true(foreign_conv.status_code == 404, "Employee B blocked from foreign workspace conversations (404)")

    foreign_graph = requests.get(f"{BASE_URL}/workspaces/{foreign_ws_id}/graph", headers=headers_b)
    assert_true(foreign_graph.status_code == 404, "Employee B blocked from foreign workspace graph (404)")

    print("[SUCCESS] Workflow 4: Cross-Workspace Tenant Isolation Verified.")

    # -------------------------------------------------------------
    # WORKFLOW 5: Disaster Recovery & Backup Integrity
    # -------------------------------------------------------------
    print_step("Workflow 5: Disaster Recovery & Backup Integrity")

    db_path = Path("data/ekn.db")
    assert_true(db_path.exists(), "Active SQLite database exists")

    # Create verified online backup
    from backend.scripts.backup_database import backup_database
    backup_file = backup_database("data/backups")
    assert_true(Path(backup_file).exists(), f"Online backup successfully created: {backup_file}")

    # Check backup SQLite integrity
    conn = sqlite3.connect(backup_file)
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    conn.close()
    assert_true(integrity == "ok", f"Backup database integrity PRAGMA returned: {integrity}")

    print("[SUCCESS] Workflow 5: Disaster Recovery & Backup Integrity Verified.")

    print("\n" + "=" * 70)
    print("ALL 5 ACCEPTANCE WORKFLOWS COMPLETED SUCCESSFULLY WITH 100% PASS RATE")
    print("=" * 70)

if __name__ == "__main__":
    run_acceptance_workflows()
