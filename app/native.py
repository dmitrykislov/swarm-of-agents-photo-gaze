"""Native (no-Docker) launcher for the packaged desktop app.

Wires the server-less stack together and starts it as ONE process:
  - SQLite at  <data dir>/app.db   (DATABASE_URL)
  - a bundled Qdrant binary as a sidecar on a free loopback port (so HNSW /
    300k-scale search is preserved), or an external QDRANT_URL if no binary
  - the FastAPI app on a fixed loopback port, serving the bundled React build
    same-origin and the ONNX embedding model
  - opens the default browser once /health is up; stops the sidecar on exit

The heavy logic lives in small, unit-tested helpers; main() just orchestrates.
"""
import atexit
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

# Fixed loopback port so the UI bundle (built with this URL) and the backend
# agree. Override with PHOTO_GAZE_PORT.
DEFAULT_API_PORT = 8765


def app_root() -> str:
    """Directory bundled resources live under. Under PyInstaller that's the
    unpacked bundle (sys._MEIPASS); otherwise the repo root."""
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bundled(*parts: str) -> str:
    return os.path.join(app_root(), *parts)


def data_dir() -> str:
    """Per-user writable data directory (created if missing)."""
    base = os.getenv("PHOTO_GAZE_DATA_DIR")
    if not base:
        if sys.platform == "darwin":
            base = os.path.expanduser("~/Library/Application Support/PhotoGaze")
        elif os.name == "nt":
            base = os.path.join(os.getenv("APPDATA", os.path.expanduser("~")), "PhotoGaze")
        else:
            base = os.path.expanduser("~/.local/share/PhotoGaze")
    os.makedirs(base, exist_ok=True)
    return base


def find_free_port() -> int:
    """An OS-assigned free TCP port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_http(url: str, timeout: float = 30.0, interval: float = 0.3) -> bool:
    """Poll `url` until it responds (any < 500 status) or `timeout` elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def start_qdrant_sidecar(binary: str, storage_dir: str, http_port: int):
    """Launch the bundled Qdrant binary as a child process. Returns the Popen,
    or None if no binary is configured/found (the caller then relies on an
    external QDRANT_URL). Qdrant is configured entirely via QDRANT__* env."""
    if not binary or not os.path.isfile(binary):
        return None
    os.makedirs(storage_dir, exist_ok=True)
    env = dict(os.environ)
    env["QDRANT__STORAGE__STORAGE_PATH"] = storage_dir
    env["QDRANT__SERVICE__HTTP_PORT"] = str(http_port)
    env["QDRANT__SERVICE__GRPC_PORT"] = str(http_port + 1)
    env["QDRANT__TELEMETRY_DISABLED"] = "true"
    return subprocess.Popen(
        [binary], env=env, cwd=os.path.dirname(binary) or ".",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def configure_environment(base: str) -> None:
    """Point the FastAPI app at SQLite, the bundled ONNX model and React build,
    all before app.main is imported. setdefault so explicit env (dev/tests)
    always wins."""
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{os.path.join(base, 'app.db')}")
    os.environ.setdefault("DINOV2_ONNX_PATH", bundled("models", "dinov2_vits14.onnx"))
    os.environ.setdefault("EMBEDDING_BACKEND", "onnx")
    os.environ.setdefault("TRASH_DIR", os.path.join(base, "trash"))
    # Empty → thumbnail links are RELATIVE ("/thumbnails/..."), so the
    # same-origin UI reaches them on whatever port we end up on.
    os.environ.setdefault("BACKEND_PUBLIC_URL", "")
    fe = bundled("build")
    if os.path.isdir(fe):
        os.environ.setdefault("FRONTEND_DIR", fe)


def main() -> None:
    base = data_dir()
    api_port = int(os.getenv("PHOTO_GAZE_PORT") or DEFAULT_API_PORT)

    # Qdrant sidecar on its own free port (backend reads QDRANT_URL at startup).
    qbin = os.getenv("QDRANT_BINARY", bundled("qdrant"))
    qport = find_free_port()
    proc = start_qdrant_sidecar(qbin, os.path.join(base, "qdrant"), qport)
    if proc is not None:
        os.environ["QDRANT_URL"] = f"http://127.0.0.1:{qport}"
        atexit.register(proc.terminate)
        if not wait_for_http(f"http://127.0.0.1:{qport}/readyz", timeout=30):
            print("WARNING: Qdrant sidecar did not become ready in time", file=sys.stderr)
    os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:6333")

    configure_environment(base)

    url = f"http://127.0.0.1:{api_port}"
    threading.Thread(
        target=lambda: wait_for_http(f"{url}/health", 90) and webbrowser.open(url),
        daemon=True,
    ).start()

    import uvicorn
    print(f"Photo Gaze running at {url}  (data: {base})")
    uvicorn.run("app.main:app", host="127.0.0.1", port=api_port, log_level="info")


if __name__ == "__main__":
    main()
