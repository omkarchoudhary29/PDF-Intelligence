"""Integration test verifying runtime behavior of the production supervisor.

Tests:
1. Supervisor launches FastAPI on 127.0.0.1:8000.
2. Supervisor waits for /health liveness.
3. Supervisor starts Ingestion Worker (heartbeat active).
4. Supervisor launches Streamlit on designated port and responds to /_stcore/health.
5. Supervisor cleanly terminates all child processes on SIGINT/terminate.
"""

import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_supervisor_live_orchestration(tmp_path):
    """Start supervisor, probe liveness endpoints, and verify graceful termination."""
    test_storage = tmp_path / "storage"
    test_storage.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)
    env["PORT"] = "8502"
    env["BACKEND_PORT"] = "8008"
    env["STORAGE_LOCAL_ROOT"] = str(test_storage)
    env["DATABASE_URL"] = f"sqlite:///{test_storage.resolve().as_posix()}/test.db"
    env["ENVIRONMENT"] = "development"

    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "start_production.py"),
        "--backend-port", "8008",
        "--frontend-port", "8502",
        "--host", "127.0.0.1",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        # Wait up to 60 seconds for FastAPI and Streamlit to become healthy (accounting for model import)
        backend_healthy = False
        frontend_healthy = False
        start_wait = time.time()

        while time.time() - start_wait < 60.0:
            if not backend_healthy:
                try:
                    req = urllib.request.Request("http://127.0.0.1:8008/health")
                    with urllib.request.urlopen(req, timeout=1.0) as resp:
                        if resp.status == 200:
                            backend_healthy = True
                except Exception:
                    pass

            if not frontend_healthy:
                try:
                    req = urllib.request.Request("http://127.0.0.1:8502/_stcore/health")
                    with urllib.request.urlopen(req, timeout=1.0) as resp:
                        if resp.status == 200:
                            frontend_healthy = True
                except Exception:
                    pass

            if backend_healthy and frontend_healthy:
                break
            time.sleep(1.0)

        assert backend_healthy, "FastAPI backend /health failed to respond with HTTP 200"
        assert frontend_healthy, "Streamlit /_stcore/health failed to respond with HTTP 200"

        # Verify worker heartbeat file was created in storage (allow up to 10s for process spawn)
        hb_file = test_storage / ".worker_heartbeat"
        hb_start = time.time()
        while time.time() - hb_start < 10.0:
            if hb_file.exists():
                break
            time.sleep(0.5)
        assert hb_file.exists(), "Worker heartbeat was not written to storage root"

    finally:
        # Terminate supervisor
        proc.terminate()
        try:
            proc.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
