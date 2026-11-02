"""
EmotiSync - Desktop Software Launcher
Launches EmotiSync in a dedicated native application window.
Prioritizes:
1. Compiled standalone executable (dist/EmotiSync.exe)
2. Electron desktop runtime (main.js)
3. Native Edge/Chrome application mode (--app)
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def launch():
    # 1. Check if compiled standalone executable exists
    exe_path = ROOT_DIR / "dist" / "EmotiSync.exe"
    if exe_path.is_file():
        print(f"[EmotiSync] Launching standalone executable from {exe_path}...")
        try:
            return subprocess.call([str(exe_path)])
        except Exception as e:
            print(f"[EmotiSync] Direct .exe execution failed: {e}")

    # 2. Try Electron runtime via npx
    npx_path = shutil.which("npx") or shutil.which("npx.cmd")
    if npx_path:
        print("[EmotiSync] Launching via Electron desktop runtime...")
        try:
            return subprocess.call([npx_path, "electron", "."], cwd=str(ROOT_DIR))
        except Exception as e:
            print(f"[EmotiSync] Electron launch failed: {e}")

    # 3. Fallback to native Chromium/Edge standalone app-mode
    print("[EmotiSync] Launching in standalone native application window...")
    import http.server
    import socketserver
    import threading

    static_dir = ROOT_DIR / "ui" / "static"
    os.chdir(str(static_dir))
    handler = http.server.SimpleHTTPRequestHandler

    # Find free port on localhost
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

        target_url = f"http://127.0.0.1:{port}/index.html"
        print(f"[EmotiSync] Running local server at {target_url}")

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
            return subprocess.call([browser_exe, f"--app={target_url}", "--window-size=1366,860"])
        else:
            import webbrowser
            webbrowser.open(target_url)
            return 0

if __name__ == "__main__":
    sys.exit(launch())
