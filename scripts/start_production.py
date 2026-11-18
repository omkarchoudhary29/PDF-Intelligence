"""Enterprise Knowledge Navigator — Unified Multi-Process Production Supervisor.

Orchestrates the three core application processes for Railway deployment:
1. FastAPI Backend Server (Uvicorn, internal loopback only on 127.0.0.1:8000)
2. Document Ingestion Worker Daemon (backend.worker, background child process)
3. Streamlit Enterprise UI Frontend (public ingress on 0.0.0.0:$PORT)

Monitors process health, restarts a crashed worker with recurring capped exponential backoff
WITHOUT interrupting the healthy frontend or backend, treats FastAPI or Streamlit termination
as an application-level failure, handles POSIX SIGINT/SIGTERM signals gracefully, and cleans
up all child processes without leaving orphans.
"""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT_DIR = Path(__file__).resolve().parent.parent


def wait_for_backend_health(url: str, timeout: float = 30.0) -> bool:
    """Poll backend /health endpoint until it responds with HTTP 200 or timeout expires."""
    start_time = time.time()
    print(f"[EKN Supervisor] Waiting for FastAPI liveness at {url} (timeout: {timeout}s)...")
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "EKN-Supervisor/1.0"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    print(f"[EKN Supervisor] FastAPI backend is live and healthy ({resp.status}).")
                    return True
        except (urllib.error.URLError, ConnectionResetError, OSError):
            time.sleep(0.5)
    return False


def start_processes(backend_port: int = 8000, frontend_port: int = 8501, host: str = "0.0.0.0"):
    python_exe = sys.executable

    # 1. Resolve ports and networking
    backend_port_env = os.environ.get("BACKEND_PORT")
    if backend_port_env and backend_port_env.isdigit():
        backend_port = int(backend_port_env)

    # Railway injects $PORT for public web ingress
    railway_port = os.environ.get("PORT")
    if railway_port and railway_port.isdigit():
        frontend_port = int(railway_port)
        print(f"[EKN Supervisor] Detected Railway PORT environment variable: {frontend_port}")
    else:
        print(f"[EKN Supervisor] Using local frontend port: {frontend_port}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)
    # Streamlit connects to internal FastAPI.
    # IMPORTANT: Do not append /api/v1 — BackendAPIClient appends the API path automatically.
    env["BACKEND_API_URL"] = f"http://127.0.0.1:{backend_port}"

    procs = {}
    is_shutting_down = False
    exit_status = 0

    def terminate_all(exit_code: int = 0, signum=None, frame=None):
        nonlocal is_shutting_down, exit_status
        if is_shutting_down:
            return
        is_shutting_down = True
        exit_status = exit_code

        sig_name = f"signal {signum}" if signum else f"exit code {exit_code}"
        print(f"\n[EKN Supervisor] Shutdown triggered ({sig_name}). Stopping all child processes...")

        # Step 1: Send SIGTERM for graceful shutdown
        for name, p in procs.items():
            if p and p.poll() is None:
                print(f"[EKN Supervisor] Sending SIGTERM to {name} (PID {p.pid})...")
                try:
                    p.terminate()
                except OSError:
                    pass

        # Step 2: Wait up to 6 seconds for children to terminate
        grace_period = 6.0
        start_wait = time.time()
        while time.time() - start_wait < grace_period:
            if all(p.poll() is not None for p in procs.values() if p):
                break
            time.sleep(0.4)

        # Step 3: Escalate to SIGKILL for any lingering processes
        for name, p in procs.items():
            if p and p.poll() is None:
                print(f"[EKN Supervisor] Escalate to SIGKILL for {name} (PID {p.pid})...")
                try:
                    p.kill()
                    p.wait(timeout=2.0)
                except OSError:
                    pass

        print(f"[EKN Supervisor] All child processes stopped. Exiting with code {exit_status}.")
        sys.exit(exit_status)

    signal.signal(signal.SIGINT, lambda s, f: terminate_all(exit_code=0, signum=s, frame=f))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda s, f: terminate_all(exit_code=0, signum=s, frame=f))

    # --- Process A: Launch FastAPI Backend (internal loopback only) ---
    backend_cmd = [
        python_exe, "-m", "uvicorn", "backend.main:app",
        "--host", "127.0.0.1",
        "--port", str(backend_port),
    ]
    print(f"[EKN Supervisor] Launching FastAPI backend on 127.0.0.1:{backend_port}...")
    procs["backend"] = subprocess.Popen(backend_cmd, cwd=str(ROOT_DIR), env=env)

    # Wait for backend liveness endpoint before launching dependent services
    health_url = f"http://127.0.0.1:{backend_port}/health"
    if not wait_for_backend_health(health_url, timeout=45.0):
        backend_ret = procs["backend"].poll()
        print(f"[EKN Supervisor] FATAL: FastAPI backend failed to become healthy (exit code: {backend_ret}).")
        terminate_all(exit_code=backend_ret if backend_ret is not None else 1)

    # --- Process B: Launch Document Ingestion Worker ---
    def spawn_worker():
        worker_cmd = [python_exe, "-m", "backend.worker"]
        print("[EKN Supervisor] Launching Ingestion Worker Daemon...")
        return subprocess.Popen(worker_cmd, cwd=str(ROOT_DIR), env=env)

    procs["worker"] = spawn_worker()

    # --- Process C: Launch Streamlit Frontend (public ingress on 0.0.0.0:$PORT) ---
    # NOTE: XSRF protection is kept enabled for enterprise session security.
    frontend_cmd = [
        python_exe, "-m", "streamlit", "run", "frontend/streamlit_app.py",
        "--server.headless", "true",
        "--server.port", str(frontend_port),
        "--server.address", host,
        "--server.enableCORS", "false",
    ]
    print(f"[EKN Supervisor] Launching Streamlit UI on {host}:{frontend_port} (XSRF protection enabled)...")
    procs["frontend"] = subprocess.Popen(frontend_cmd, cwd=str(ROOT_DIR), env=env)

    print(f"[EKN Supervisor] All processes running. Monitoring system...")

    # Worker restart tracking with recurring, capped exponential backoff
    worker_consecutive_failures = 0
    base_backoff_sec = 2.0
    max_backoff_sec = 30.0
    last_spawn_time = time.time()
    stable_window_sec = 120.0  # 2 minutes of uptime resets consecutive failure counter

    try:
        while not is_shutting_down:
            time.sleep(1.5)

            # Check FastAPI Backend (fatal if terminated)
            b_ret = procs["backend"].poll()
            if b_ret is not None:
                print(f"[EKN Supervisor] CRITICAL: FastAPI backend exited unexpectedly with code {b_ret}!")
                terminate_all(exit_code=b_ret or 1)

            # Check Streamlit Frontend (fatal if terminated)
            f_ret = procs["frontend"].poll()
            if f_ret is not None:
                print(f"[EKN Supervisor] CRITICAL: Streamlit frontend exited unexpectedly with code {f_ret}!")
                terminate_all(exit_code=f_ret or 1)

            # Check Ingestion Worker (with recurring capped exponential backoff without killing UI)
            w_ret = procs["worker"].poll()
            if w_ret is not None and not is_shutting_down:
                now = time.time()
                # If worker ran stably before crashing, reset failure count
                if now - last_spawn_time >= stable_window_sec:
                    worker_consecutive_failures = 0

                worker_consecutive_failures += 1
                backoff = min(base_backoff_sec * (2 ** (worker_consecutive_failures - 1)), max_backoff_sec)
                print(
                    f"[EKN Supervisor] WARNING: Ingestion worker exited with code {w_ret}. "
                    f"Retrying in {backoff:.1f}s (consecutive failures: {worker_consecutive_failures}). "
                    f"Web UI and API remain active."
                )

                # Wait with backoff before re-spawning
                backoff_start = time.time()
                while time.time() - backoff_start < backoff:
                    if is_shutting_down:
                        break
                    time.sleep(0.5)

                if not is_shutting_down:
                    last_spawn_time = time.time()
                    procs["worker"] = spawn_worker()

    except KeyboardInterrupt:
        terminate_all(exit_code=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EKN Multi-Process Production Supervisor")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=8501)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    start_processes(backend_port=args.backend_port, frontend_port=args.frontend_port, host=args.host)
