"""Enterprise Knowledge Navigator — Unified Multi-Process Production Runner.

Orchestrates the three core processes:
1. FastAPI Backend Server (Uvicorn)
2. Document Ingestion Worker Daemon (backend.worker)
3. Streamlit Enterprise Frontend

Monitors process health, handles SIGINT/SIGTERM gracefully, and shuts down all child processes cleanly.
"""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent.parent


def start_processes(backend_port: int = 8010, frontend_port: int = 8501, host: str = "0.0.0.0"):
    python_exe = sys.executable

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)

    # 1. Start FastAPI Backend
    backend_cmd = [
        python_exe, "-m", "uvicorn", "backend.main:app",
        "--host", host,
        "--port", str(backend_port),
    ]
    print(f"[EKN Supervisor] Launching FastAPI backend on {host}:{backend_port}...")
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(ROOT_DIR), env=env)

    # 2. Start Ingestion Worker
    worker_cmd = [python_exe, "-m", "backend.worker"]
    print("[EKN Supervisor] Launching Ingestion Worker Daemon...")
    worker_proc = subprocess.Popen(worker_cmd, cwd=str(ROOT_DIR), env=env)

    # 3. Start Streamlit Frontend
    frontend_cmd = [
        python_exe, "-m", "streamlit", "run", "frontend/streamlit_app.py",
        "--server.headless", "true",
        "--server.port", str(frontend_port),
        "--server.address", host,
    ]
    print(f"[EKN Supervisor] Launching Streamlit frontend on {host}:{frontend_port}...")
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(ROOT_DIR), env=env)

    procs = [backend_proc, worker_proc, frontend_proc]

    def terminate_all(signum=None, frame=None):
        print("\n[EKN Supervisor] Termination signal received. Stopping all processes...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        time.sleep(2)
        for p in procs:
            if p.poll() is None:
                p.kill()
        print("[EKN Supervisor] All processes terminated cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, terminate_all)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, terminate_all)

    print("[EKN Supervisor] All services active. Press Ctrl+C to terminate.")
    try:
        while True:
            for name, p in [("FastAPI Backend", backend_proc), ("Ingestion Worker", worker_proc), ("Streamlit", frontend_proc)]:
                ret = p.poll()
                if ret is not None:
                    print(f"[EKN Supervisor] CRITICAL: {name} exited unexpectedly with code {ret}!")
                    terminate_all()
            time.sleep(2)
    except KeyboardInterrupt:
        terminate_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EKN Multi-Process Production Supervisor")
    parser.add_argument("--backend-port", type=int, default=8010)
    parser.add_argument("--frontend-port", type=int, default=8501)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    start_processes(backend_port=args.backend_port, frontend_port=args.frontend_port, host=args.host)
