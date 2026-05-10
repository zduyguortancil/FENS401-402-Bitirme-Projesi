"""
BiletBul desktop launcher (Customer Portal).
"""

from __future__ import annotations

import os
import sys
import socket
import threading
import time
import urllib.error
import urllib.request

from werkzeug.serving import make_server

# ── Import Flask app ──────────────────────────────────────
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_CUSTOMER_DIR = os.path.join(_DASHBOARD_DIR, "customer")
sys.path.insert(0, _CUSTOMER_DIR)

from customer_app import app as biletbul_app  # noqa: E402

try:
    import webview
except ImportError:
    print("pywebview is not installed.")
    print("Install it with: pip install pywebview")
    sys.exit(1)


# ── Config ────────────────────────────────────────────────
BB_TITLE   = "BiletBul — Flight Search"
BB_HOST, BB_PORT = "127.0.0.1", 5006
BB_W, BB_H = 1280, 860


# ── Helpers ───────────────────────────────────────────────
def _find_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No open port found near {preferred}")


def _wait_ready(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.3)
    raise RuntimeError(f"Server did not become ready: {url}")


class _FlaskThread(threading.Thread):
    def __init__(self, flask_app, host: str, port: int):
        super().__init__(daemon=True)
        self._server = make_server(host, port, flask_app, threaded=True)
        ctx = flask_app.app_context()
        ctx.push()

    def run(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()


# ── Main ──────────────────────────────────────────────────
def main() -> None:
    bb_port = _find_port(BB_HOST, BB_PORT)
    bb_url = f"http://{BB_HOST}:{bb_port}"

    # Start BiletBul
    bb_server = _FlaskThread(biletbul_app, BB_HOST, bb_port)
    bb_server.start()

    print(f"[BiletBul ] Waiting for server at {bb_url} ...")
    _wait_ready(bb_url)

    print(f"\n✔  BiletBul  → {bb_url}/login\n")

    # Create pywebview window
    bb_window = webview.create_window(
        BB_TITLE,
        f"{bb_url}/login",
        width=BB_W,
        height=BB_H,
        min_size=(1024, 700),
    )

    def _on_closed():
        bb_server.shutdown()

    bb_window.events.closed += _on_closed

    webview.start(debug=False)


if __name__ == "__main__":
    main()
