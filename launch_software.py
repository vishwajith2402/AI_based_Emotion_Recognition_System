"""
Human Emotion Recognition AI - Desktop Software Launcher
Launches the full desktop software in a dedicated native application window.
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def launch_electron():
    # Try electron first
    npx_path = shutil.which("npx") or shutil.which("npx.cmd")
    npm_path = shutil.which("npm") or shutil.which("npm.cmd")
    
    if npx_path:
        print("[AIPS Software] Launching via Electron desktop runtime...")
        try:
            return subprocess.call([npx_path, "electron", "."], cwd=str(ROOT_DIR))
        except Exception as e:
            print(f"[AIPS Software] Electron launch failed: {e}")

    # Fallback to Chromium/Edge app-mode
    print("[AIPS Software] Launching in standalone application window...")
    import http.server
    import socketserver
    import threading

    os.chdir(str(ROOT_DIR / "ui" / "static"))
    handler = http.server.SimpleHTTPRequestHandler
    
    # Find free port
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        
        target_url = f"http://127.0.0.1:{port}/index.html"
        print(f"[AIPS Software] Running local server at {target_url}")
        
        # Try MS Edge or Google Chrome in standalone app mode
        edge_paths = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
        ]
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe")
        ]
        
        browser_exe = None
        for p in edge_paths + chrome_paths:
            if os.path.exists(p):
                browser_exe = p
                break
                
        if browser_exe:
            subprocess.call([browser_exe, f"--app={target_url}", "--window-size=1366,860"])
        else:
            import webbrowser
            webbrowser.open(target_url)

if __name__ == "__main__":
    sys.exit(launch_electron())
