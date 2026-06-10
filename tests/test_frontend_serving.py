"""Tests for serving the built React UI from FastAPI (_mount_frontend),
used by the native single-process app."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import _mount_frontend


def _make_build(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>Photo Gaze</title>")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("console.log('ui')")
    return str(tmp_path)


def test_mount_serves_index_and_assets(tmp_path):
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    assert _mount_frontend(app, _make_build(tmp_path)) is True
    client = TestClient(app)

    # API route still wins (registered before the catch-all mount).
    assert client.get("/health").json() == {"status": "healthy"}
    # UI index served at root, assets served under /assets.
    root = client.get("/")
    assert root.status_code == 200 and "Photo Gaze" in root.text
    assert client.get("/assets/app.js").status_code == 200


def test_mount_is_noop_without_directory(tmp_path):
    app = FastAPI()
    # Missing/empty dir → not mounted (the Docker case where FRONTEND_DIR unset).
    assert _mount_frontend(app, "") is False
    assert _mount_frontend(app, str(tmp_path / "does-not-exist")) is False
    assert client_routes_have_no_catchall(app)


def client_routes_have_no_catchall(app) -> bool:
    return not any(getattr(r, "name", "") == "frontend" for r in app.routes)
