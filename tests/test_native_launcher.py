"""Tests for the native launcher helpers (app/native.py).

Cover the pure/IO-light pieces — data dir, free port, HTTP readiness wait,
sidecar guard, and environment wiring. The full main() (spawns Qdrant +
uvicorn) is integration-only and not exercised here."""
import http.server
import os
import socket
import threading

from app import native


def test_find_free_port_is_actually_free():
    port = native.find_free_port()
    assert isinstance(port, int) and 1024 < port < 65536
    # We can bind it again → it was genuinely free and released.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))


def test_data_dir_created(tmp_path, monkeypatch):
    target = tmp_path / "PhotoGaze"
    monkeypatch.setenv("PHOTO_GAZE_DATA_DIR", str(target))
    d = native.data_dir()
    assert d == str(target) and os.path.isdir(d)


def test_wait_for_http_true_when_server_up():
    port = native.find_free_port()

    class _H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        def log_message(self, *a):  # silence
            pass

    srv = http.server.HTTPServer(("127.0.0.1", port), _H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        assert native.wait_for_http(f"http://127.0.0.1:{port}/", timeout=5) is True
    finally:
        srv.shutdown()


def test_wait_for_http_false_on_timeout():
    port = native.find_free_port()  # nothing listening here
    assert native.wait_for_http(f"http://127.0.0.1:{port}/", timeout=0.6, interval=0.2) is False


def test_qdrant_sidecar_none_without_binary(tmp_path):
    # No binary configured → returns None (caller uses an external QDRANT_URL).
    assert native.start_qdrant_sidecar("", str(tmp_path / "q"), 7000) is None
    assert native.start_qdrant_sidecar(str(tmp_path / "missing"), str(tmp_path / "q"), 7000) is None


def test_configure_environment_sets_sqlite_and_backend(tmp_path, monkeypatch):
    # Clear anything that would pre-empt setdefault.
    for k in ("DATABASE_URL", "DINOV2_ONNX_PATH", "EMBEDDING_BACKEND",
              "TRASH_DIR", "BACKEND_PUBLIC_URL", "FRONTEND_DIR",
              "THUMBNAILS_DIR", "BACKUP_DIR"):
        monkeypatch.delenv(k, raising=False)
    base = str(tmp_path / "data")
    os.makedirs(base, exist_ok=True)
    native.configure_environment(base)
    assert os.environ["DATABASE_URL"] == f"sqlite:///{os.path.join(base, 'app.db')}"
    assert os.environ["EMBEDDING_BACKEND"] == "onnx"
    # Empty → relative thumbnail URLs (same-origin UI, port-independent).
    assert os.environ["BACKEND_PUBLIC_URL"] == ""
    assert os.environ["DINOV2_ONNX_PATH"].endswith(os.path.join("models", "dinov2_vits14.onnx"))
    # All writable dirs must be under the data dir — never the read-only bundle
    # or a cwd-relative path (Finder launches with cwd="/").
    assert os.environ["THUMBNAILS_DIR"] == os.path.join(base, "thumbnails")
    assert os.environ["BACKUP_DIR"] == os.path.join(base, "backups")
    assert os.environ["TRASH_DIR"] == os.path.join(base, "trash")
