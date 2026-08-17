"""Capture the 12 required product screenshots for EKN via Chrome DevTools Protocol (CDP).

Implements strict page-specific DOM assertions:
- Verifies absence of skeleton loaders.
- Asserts presence of required textual headings, cards, and buttons for each view.
- Strictly marks capture FAILED if assertions do not pass within the timeout.
- Dynamically verifies the demo conversation ID from the SQLite database.
"""

import asyncio
import base64
import json
from pathlib import Path
import sqlite3
import subprocess
import time
import urllib.parse
from typing import Dict, List, Tuple
import requests
import websockets

CHROME_PATH = os.getenv("EKN_CHROME_PATH", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
TARGET_DIR = Path(os.getenv("EKN_SCREENSHOT_DIR", r"C:\Users\mukesh\.gemini\antigravity\brain\d0ad46fd-6e29-4b89-88dd-4084c525e6ca\screenshots"))
CDP_PORT = int(os.getenv("EKN_CDP_PORT", "9229"))


def get_verified_demo_conversation_id() -> str:
    """Dynamically verify or resolve the active demo conversation ID from the SQLite database."""
    db_path = Path("data/ekn.db")
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            c = conn.cursor()
            # Check if default turbine inquiry exists
            row = c.execute(
                "SELECT id FROM conversations WHERE id = '8388cac0-13e4-4b9d-a4b3-a4a7228145ee'"
            ).fetchone()
            if row:
                conn.close()
                return row[0]

            # Fallback to demo employee's most active conversation
            row2 = c.execute(
                """
                SELECT c.id FROM conversations c
                JOIN users u ON c.user_id = u.id
                JOIN messages m ON c.id = m.conversation_id
                WHERE u.email LIKE '%employee%'
                GROUP BY c.id
                ORDER BY count(m.id) DESC
                LIMIT 1
                """
            ).fetchone()
            conn.close()
            if row2:
                return row2[0]
        except Exception:
            pass
    return "8388cac0-13e4-4b9d-a4b3-a4a7228145ee"


DEMO_CONV_ID = get_verified_demo_conversation_id()

# The 14 required targets with page-specific text assertions
SCREENSHOT_TARGETS: List[Tuple[str, str, List[str]]] = [
    (
        "01_welcome_page.png",
        "http://localhost:8501/?logout=true&portal=role_selection",
        [
            "Knowledge Intelligence Platform",
            "Enterprise Knowledge Navigator",
            "Administrator Portal",
            "Employee Portal",
            "Continue as Administrator",
            "Continue as Employee",
            "Quick Demo Sign In",
        ],
    ),
    (
        "02_admin_signin.png",
        "http://localhost:8501/?logout=true&portal=admin&admin_tab=signin",
        [
            "Administrator Portal",
            "Welcome to your company workspace",
            "Work Email",
            "Sign In as Administrator",
            "Demo Credentials Pre-filled",
        ],
    ),
    (
        "03_employee_signin.png",
        "http://localhost:8501/?logout=true&portal=employee&emp_tab=signin",
        [
            "Employee Portal",
            "Sign in to your team workspace",
            "Sign In as Employee",
            "Demo Credentials Pre-filled",
        ],
    ),
    (
        "04_admin_overview.png",
        "http://localhost:8501/?demo_login=admin&view=Overview",
        [
            "Good",
            "Company Documents",
            "Search Ready",
            "Pending Reviews",
            "Active Employees",
            "Recent Documents",
            "Knowledge Review",
            "Quick Actions",
        ],
    ),
    (
        "05_admin_documents.png",
        "http://localhost:8501/?demo_login=admin&view=Documents&doc_tab=library",
        [
            "Documents & Knowledge Library",
            "Document Library",
            "Filter by Status",
        ],
    ),
    (
        "06_admin_knowledge_review.png",
        "http://localhost:8501/?demo_login=admin&view=Knowledge+Review",
        [
            "Knowledge Review",
            "All Pending",
            "Scan for Issues",
        ],
    ),
    (
        "07_admin_knowledge_graph.png",
        "http://localhost:8501/?demo_login=admin&view=Knowledge+Graph",
        [
            "Knowledge Graph",
            "DOCUMENT NODES",
            "CONFIRMED ENTITIES",
            "DISCOVERED EDGES",
        ],
    ),
    (
        "08_admin_team.png",
        "http://localhost:8501/?demo_login=admin&view=Team",
        [
            "Team & Access Management",
            "Employee Directory",
            "Active Workspace Members",
        ],
    ),
    (
        "09_admin_reports_settings.png",
        "http://localhost:8501/?demo_login=admin&view=Reports+%26+Settings",
        [
            "Reports & Company Settings",
            "Knowledge Dossiers & Reports",
            "Company Activity History",
            "Company Settings",
        ],
    ),
    (
        "10_employee_new_chat.png",
        "http://localhost:8501/?demo_login=employee&conv_id=new",
        [
            "Document Assistant",
            "New Conversation",
            "How can I help you today?",
            "Workspace Panel",
            "Source Scope",
        ],
    ),
    (
        "11_employee_active_chat.png",
        f"http://localhost:8501/?demo_login=employee&conv_id={DEMO_CONV_ID}",
        [
            "Document Assistant",
            "Turbine",
            "Workspace Panel",
            "Verified Citations",
        ],
    ),
    (
        "12_employee_my_files.png",
        f"http://localhost:8501/?demo_login=employee&conv_id={DEMO_CONV_ID}&emp_right_view=My+Files",
        [
            "Document Assistant",
            "Workspace Panel",
            "Upload Personal PDF",
            "Your Documents",
        ],
    ),
    (
        "13_employee_company_files.png",
        f"http://localhost:8501/?demo_login=employee&conv_id={DEMO_CONV_ID}&emp_right_view=Company+Files",
        [
            "Document Assistant",
            "Workspace Panel",
            "Shared Company Files",
        ],
    ),
    (
        "14_employee_citation_preview.png",
        f"http://localhost:8501/?demo_login=employee&conv_id={DEMO_CONV_ID}&emp_right_view=Context+%26+Sources",
        [
            "Document Assistant",
            "Workspace Panel",
            "Source Scope",
            "Sources Used",
        ],
    ),
]


async def capture_target(filename: str, target_url: str, required_strings: List[str]):
    new_tab_url = f"http://127.0.0.1:{CDP_PORT}/json/new"
    res = requests.put(new_tab_url)
    target_info = res.json()
    ws_url = target_info["webSocketDebuggerUrl"]
    target_id = target_info["id"]

    try:
        async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
            msg_id = 0

            async def send(method, params=None):
                nonlocal msg_id
                msg_id += 1
                cur_id = msg_id
                msg = {"id": cur_id, "method": method}
                if params:
                    msg["params"] = params
                await ws.send(json.dumps(msg))
                while True:
                    resp = json.loads(await ws.recv())
                    if resp.get("id") == cur_id:
                        return resp

            await send("Page.enable")
            await send("DOM.enable")
            await send("Runtime.enable")

            await send(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": 1440,
                    "height": 900,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
            )

            # Navigate to target URL explicitly
            await send("Page.navigate", {"url": target_url})

            # Poll for page-specific readiness assertions
            max_wait = 20.0
            start_t = time.time()
            assertion_passed = False
            missing_assert = None

            req_json = json.dumps(required_strings)
            eval_script = f"""
            (() => {{
                const skeletons = document.querySelectorAll('.stSkeleton, [data-testid="stSkeleton"]');
                if (skeletons.length > 0) return {{ ready: false, reason: 'skeletons_active' }};
                const bodyText = document.body ? document.body.innerText : '';
                if (!bodyText || bodyText.length < 50) return {{ ready: false, reason: 'body_too_short' }};
                
                const required = {req_json};
                const lowerText = bodyText.toLowerCase();
                for (const item of required) {{
                    if (!lowerText.includes(item.toLowerCase())) {{
                        return {{ ready: false, reason: 'missing_' + item }};
                    }}
                }}
                return {{ ready: true }};
            }})()
            """

            while time.time() - start_t < max_wait:
                eval_ready = await send(
                    "Runtime.evaluate",
                    {
                        "expression": eval_script,
                        "returnByValue": True,
                    },
                )
                res_val = eval_ready.get("result", {}).get("result", {}).get("value")
                if res_val is None:
                    res_val = eval_ready.get("result", {}).get("value")

                if isinstance(res_val, dict) and res_val.get("ready") is True:
                    assertion_passed = True
                    break
                else:
                    if isinstance(res_val, dict):
                        missing_assert = res_val.get("reason")
                await asyncio.sleep(0.5)

            if not assertion_passed:
                print(f"FAILED assertion for {filename}: {missing_assert} after {max_wait}s", flush=True)
                # Do not capture unrelated or unhydrated page
                return False

            # Cushion for rendering stabilization
            await asyncio.sleep(1.0)

            # Capture viewport screenshot
            screenshot_resp = await send(
                "Page.captureScreenshot",
                {
                    "format": "png",
                    "clip": {
                        "x": 0,
                        "y": 0,
                        "width": 1440,
                        "height": 900,
                        "scale": 1,
                    },
                },
            )

            img_base64 = screenshot_resp.get("result", {}).get("data")
            if img_base64:
                out_path = TARGET_DIR / filename
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(img_base64))
                print(f"VERIFIED & CAPTURED: {filename} ({len(img_base64)} bytes) [ASSERTIONS_PASSED]", flush=True)
                return True
            else:
                print(f"FAILED to capture {filename}: empty screenshot response", flush=True)
                return False
    finally:
        try:
            requests.put(f"http://127.0.0.1:{CDP_PORT}/json/close/{target_id}")
        except Exception:
            pass


async def run_all_captures():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures = 0
    for filename, url, required_strings in SCREENSHOT_TARGETS:
        print(f"Capturing {filename} with assertions...", flush=True)
        ok = await capture_target(filename, url, required_strings)
        if ok:
            successes += 1
        else:
            failures += 1
        await asyncio.sleep(0.5)
    print(f"\nCapture run completed: {successes} passed, {failures} failed.", flush=True)


def main():
    print(f"Launching isolated Chrome instance on port {CDP_PORT}...", flush=True)
    proc = subprocess.Popen(
        [
            CHROME_PATH,
            "--headless=new",
            "--disable-gpu",
            f"--remote-debugging-port={CDP_PORT}",
            "--window-size=1440,900",
            "--no-first-run",
            "--no-default-browser-check",
            "about:blank",
        ]
    )
    time.sleep(2.0)
    try:
        asyncio.run(run_all_captures())
    finally:
        proc.terminate()
        print("Chrome session terminated.", flush=True)


if __name__ == "__main__":
    main()
