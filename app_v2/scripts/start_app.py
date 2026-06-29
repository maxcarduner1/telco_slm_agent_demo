#!/usr/bin/env python3
"""Start backend (uvicorn) and frontend (e2e-chatbot-app-next) concurrently.

Backend runs on port 8000 (public-facing).
Frontend runs on CHAT_APP_PORT (default 3000), proxied via enable_chat_proxy=True middleware.
"""
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "8000"))
CHAT_APP_PORT = int(os.environ.get("CHAT_APP_PORT", "3000"))
FRONTEND_DIR = Path(__file__).parent.parent / "e2e-chatbot-app-next"

BACKEND_READY = [r"Application startup complete", r"Uvicorn running on"]
FRONTEND_READY = [r"Server is running on http://localhost", r"Ready on http://localhost"]


def stream_output(proc, name, ready_patterns, ready_event):
    is_ready = False
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            line = line.rstrip()
            print(f"[{name}] {line}", flush=True)
            if not is_ready and any(re.search(p, line, re.IGNORECASE) for p in ready_patterns):
                is_ready = True
                ready_event.set()
        proc.wait()
    except Exception as e:
        print(f"[{name}] monitor error: {e}", flush=True)


def main():
    # Set API_PROXY so the frontend knows where to find the backend
    os.environ.setdefault("API_PROXY", f"http://localhost:{BACKEND_PORT}/invocations")

    # Build and install frontend dependencies
    if FRONTEND_DIR.exists():
        print("[setup] Installing frontend dependencies...", flush=True)
        result = subprocess.run(
            ["npm", "install"],
            cwd=FRONTEND_DIR,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[setup] npm install failed:\n{result.stderr}", flush=True)
            sys.exit(1)
        print("[setup] Building frontend...", flush=True)
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=FRONTEND_DIR,
            capture_output=True,
            text=True,
            env={**os.environ, "CHAT_APP_PORT": str(CHAT_APP_PORT)},
        )
        if result.returncode != 0:
            print(f"[setup] npm build failed:\n{result.stderr}", flush=True)
            # Print stdout too in case there are useful messages
            print(f"[setup] stdout:\n{result.stdout}", flush=True)
            sys.exit(1)
        print("[setup] Frontend built successfully.", flush=True)
    else:
        print("[setup] e2e-chatbot-app-next not found, skipping frontend.", flush=True)

    # Start backend
    backend_ready = threading.Event()
    backend_proc = subprocess.Popen(
        ["uvicorn", "agent_app.server:app", "--host", "0.0.0.0", "--port", str(BACKEND_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=Path(__file__).parent.parent,
    )
    threading.Thread(
        target=stream_output,
        args=(backend_proc, "backend", BACKEND_READY, backend_ready),
        daemon=True,
    ).start()

    # Start frontend if available
    frontend_proc = None
    if FRONTEND_DIR.exists():
        frontend_ready = threading.Event()
        frontend_proc = subprocess.Popen(
            ["npm", "run", "start"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=FRONTEND_DIR,
            env={**os.environ, "CHAT_APP_PORT": str(CHAT_APP_PORT)},
        )
        threading.Thread(
            target=stream_output,
            args=(frontend_proc, "frontend", FRONTEND_READY, frontend_ready),
            daemon=True,
        ).start()

    # Monitor for failures
    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print(f"[monitor] backend exited with code {backend_proc.returncode}", flush=True)
                if frontend_proc:
                    frontend_proc.terminate()
                sys.exit(backend_proc.returncode or 1)
            if frontend_proc and frontend_proc.poll() is not None:
                print(f"[monitor] frontend exited with code {frontend_proc.returncode}", flush=True)
                backend_proc.terminate()
                sys.exit(frontend_proc.returncode or 1)
    except KeyboardInterrupt:
        print("\n[monitor] Shutting down...", flush=True)
        backend_proc.terminate()
        if frontend_proc:
            frontend_proc.terminate()
        sys.exit(0)


if __name__ == "__main__":
    main()
