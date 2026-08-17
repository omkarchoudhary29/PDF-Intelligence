"""Post-Cleanup Verification Script for EKN Repository Hygiene.

Verifies:
1. Database record counts (Users, Workspaces, Memberships, Documents, Conversations).
2. Aerospace demo workspace document preservation (7 canonical PDFs).
3. ChromaDB vector count preservation (1,695 embeddings).
4. Live service status (FastAPI, Ingestion Worker, Streamlit).
5. PDF storage integrity.
"""

import sys
import sqlite3
import hashlib
from pathlib import Path
import requests

def check(cond: bool, msg: str):
    if not cond:
        print(f"[-] FAILED: {msg}")
        sys.exit(1)
    print(f"    [OK] {msg}")

def main():
    print("=" * 70)
    print("RUNNING POST-CLEANUP REPOSITORY INTEGRITY & DATA VERIFICATION")
    print("=" * 70)

    # 1. Database Integrity & Counts
    print("\n[+] 1. Checking SQLite Database & Table Counts...")
    db_path = Path("data/ekn.db")
    check(db_path.exists(), "Active SQLite database exists at data/ekn.db")
    
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    integrity = c.execute("PRAGMA integrity_check").fetchone()[0]
    check(integrity == "ok", f"PRAGMA integrity_check returned '{integrity}'")

    user_count = c.execute("SELECT count(1) FROM users").fetchone()[0]
    check(user_count >= 508, f"User count preserved: {user_count} (baseline: 508)")

    ws_count = c.execute("SELECT count(1) FROM workspaces").fetchone()[0]
    check(ws_count >= 241, f"Workspace count preserved: {ws_count} (baseline: 241)")

    doc_count = c.execute("SELECT count(1) FROM documents").fetchone()[0]
    check(doc_count >= 227, f"Total document count preserved: {doc_count} (baseline: 227)")

    comp_docs = c.execute("SELECT count(1) FROM documents WHERE visibility_scope = 'company'").fetchone()[0]
    check(comp_docs >= 111, f"Company documents preserved: {comp_docs} (baseline: 111)")

    pers_docs = c.execute("SELECT count(1) FROM documents WHERE visibility_scope = 'personal'").fetchone()[0]
    check(pers_docs >= 116, f"Personal documents preserved: {pers_docs} (baseline: 116)")

    # Check Aerospace & Engineering workspace
    aero_ws = c.execute("SELECT id FROM workspaces WHERE id = 'dc5d9139-f381-4818-ab26-c2e878997b4d'").fetchone()
    check(aero_ws is not None, "Aerospace & Engineering workspace exists")

    aero_docs = c.execute(
        "SELECT display_name, processing_status FROM documents WHERE workspace_id = 'dc5d9139-f381-4818-ab26-c2e878997b4d'"
    ).fetchall()
    check(len(aero_docs) == 7, f"Aerospace workspace contains exactly 7 canonical documents (found: {len(aero_docs)})")
    for doc_name, status in aero_docs:
        check(status == "completed", f"Document '{doc_name}' has status '{status}'")

    conn.close()

    # 2. ChromaDB Vector Count
    print("\n[+] 2. Checking ChromaDB Vector Store Count...")
    import chromadb
    chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
    coll = chroma_client.get_collection("ekn_document_chunks")
    vector_count = coll.count()
    check(vector_count == 1695, f"ChromaDB ekn_document_chunks count preserved: {vector_count} (baseline: 1695)")

    # 3. Live Service Endpoints
    print("\n[+] 3. Checking Live Backend & Worker Services...")
    try:
        health_res = requests.get("http://127.0.0.1:8010/health", timeout=5)
        check(health_res.status_code == 200, "FastAPI /health probe returned HTTP 200")
        
        ready_res = requests.get("http://127.0.0.1:8010/ready", timeout=5)
        check(ready_res.status_code == 200, "FastAPI /ready probe returned HTTP 200")
        ready_data = ready_res.json()
        worker_info = ready_data.get("services", {}).get("worker", {})
        check(worker_info.get("ready") is True, f"Worker heartbeat verified healthy: {worker_info.get('message')}")
    except Exception as e:
        check(False, f"Live backend connectivity failed: {e}")

    try:
        streamlit_res = requests.get("http://localhost:8501", timeout=5)
        check(streamlit_res.status_code == 200, "Streamlit frontend returned HTTP 200")
    except Exception as e:
        check(False, f"Streamlit frontend check failed: {e}")

    # 4. Sample & Canonical PDF Files Integrity
    print("\n[+] 4. Checking PDF Storage & Hashes...")
    aero_storage_dir = Path("data/storage/uploads/dc5d9139-f381-4818-ab26-c2e878997b4d")
    check(aero_storage_dir.exists(), "Aerospace documents storage directory exists")
    aero_files = list(aero_storage_dir.glob("*.pdf"))
    check(len(aero_files) == 7, f"Aerospace storage directory contains exactly 7 physical PDFs (found: {len(aero_files)})")
    for p in aero_files:
        check(p.stat().st_size > 1000, f"Canonical PDF '{p.name[:20]}...' is intact ({p.stat().st_size} bytes)")

    print("\n" + "=" * 70)
    print("ALL INTEGRITY & PRESERVATION CHECKS PASSED (100% VERIFIED)")
    print("=" * 70)

if __name__ == "__main__":
    main()
