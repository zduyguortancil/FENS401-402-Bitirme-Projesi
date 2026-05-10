"""
SeatWise desktop launcher (Company Portal).
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
sys.path.insert(0, _DASHBOARD_DIR)

from app import app as seatwise_app  # noqa: E402

try:
    import webview
except ImportError:
    print("pywebview is not installed.")
    print("Install it with: pip install pywebview")
    sys.exit(1)


# ── Config ────────────────────────────────────────────────
SW_TITLE   = "SeatWise — Revenue Management"
SW_HOST, SW_PORT = "127.0.0.1", 5005
WINDOW_W, WINDOW_H   = 1600, 980
MIN_W,    MIN_H      = 1200, 760


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
    sw_port = _find_port(SW_HOST, SW_PORT)
    sw_url = f"http://{SW_HOST}:{sw_port}"

    # Start SeatWise
    sw_server = _FlaskThread(seatwise_app, SW_HOST, sw_port)
    sw_server.start()

    print(f"[SeatWise ] Waiting for server at {sw_url} ...")
    _wait_ready(sw_url)

    print(f"\n✔  SeatWise  → {sw_url}/login\n")

    # Create pywebview window
    sw_window = webview.create_window(
        SW_TITLE,
        f"{sw_url}/login",
        width=WINDOW_W,
        height=WINDOW_H,
        min_size=(MIN_W, MIN_H),
    )

    def _on_closed():
        sw_server.shutdown()

    sw_window.events.closed += _on_closed

    webview.start(debug=False)


if __name__ == "__main__":
    main()
