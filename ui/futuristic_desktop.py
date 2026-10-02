"""
Futuristic Desktop Application Runner
AGB1303 - AI Problem Solving Techniques
Batch 6: Human Emotion Recognition Using Facial Expressions and Speech Modulation
Launches the FastAPI backend and opens the futuristic glassmorphism dashboard
in a native Windows desktop window using pywebview with browser fallback.
"""

import sys
import os
import time
import threading
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from ui.futuristic_server import app


def start_fastapi_server(host="127.0.0.1", port=8000):
    """Runs Uvicorn in background daemon thread."""
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


def wait_for_server(url, timeout=10):
    """Polls server URL until responsive."""
    start_t = time.time()
    while time.time() - start_t < timeout:
        try:
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def launch_futuristic_app():
    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    print("=" * 70)
    print("🚀 LAUNCHING FUTURISTIC MULTIMODAL EMOTION RECOGNITION DESKTOP APP")
    print("   Course: AGB1303 – AI Problem Solving Techniques (Batch 6)")
    print(f"   Local Server Address: {url}")
    print("=" * 70)

    # Start FastAPI Server in daemon thread
    server_thread = threading.Thread(target=start_fastapi_server, args=(host, port), daemon=True)
    server_thread.start()

    # Wait for server to become responsive
    if not wait_for_server(url):
        print("[Warning] Server startup delay. Attempting to proceed...")

    # Launch Native Windows Desktop Window using pywebview
    try:
        import webview
        print("[Desktop] Creating native futuristic Windows desktop window...")
        window = webview.create_window(
            title="Human Emotion Recognition — Multimodal AI Dashboard",
            url=url,
            width=1320,
            height=860,
            min_size=(1050, 700),
            background_color="#070B14",
            resizable=True
        )
        webview.start()
    except Exception as e:
        print(f"[Desktop Notice] pywebview native window info: {e}")
        print(f"[Fallback] Opening dashboard in your default browser: {url}")
        webbrowser.open(url)
        # Keep server thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Desktop] Shutting down application...")


if __name__ == "__main__":
    launch_futuristic_app()
