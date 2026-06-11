"""Demonstrating tests for the audit findings (written RED-first).

Each test encodes the CORRECT behavior for a confirmed bug, so it fails against
the current code (proving the defect) and passes once the fix lands. Grouped by
the finding number in the audit.
"""
import asyncio
import os
import sqlite3
import types
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Photo, Embedding, ProcessingState, JobQueue


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _RecordingQdrant:
    """Minimal fake Qdrant client that records deletions."""

    def __init__(self, *a, **k):
        self.deleted_points = []

    def delete(self, collection_name=None, points_selector=None, **k):
        if points_selector:
            self.deleted_points.extend(points_selector)


def _make_image(path, size=(32, 32), color="red"):
    Image.new("RGB", size, color=color).save(str(path), "JPEG")


def _file_db(tmp_path):
    """A file-backed SQLite engine (":memory:" gives a *separate* db per
    connection, which JobQueueManager's sessionmaker would not share)."""
    url = f"sqlite:///{tmp_path / 'audit.db'}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return url, engine


# --------------------------------------------------------------------------- #
# #1 — Qdrant backup must persist the actual vectors, not null.
# --------------------------------------------------------------------------- #
async def test_qdrant_backup_includes_vectors(tmp_path, monkeypatch):
    import json
    from app import backup_manager as bm_mod

    class _FakeQdrant:
        """Faithful to qdrant-client: scroll returns vectors ONLY when
        with_vectors=True is passed (default is False)."""

        def __init__(self, *a, **k):
            pass

        def get_collection(self, name):
            return types.SimpleNamespace(
                config=types.SimpleNamespace(
                    params=types.SimpleNamespace(
                        vectors=types.SimpleNamespace(size=384),
                        distance="Cosine",
                    )
                )
            )

        def scroll(self, collection_name=None, limit=100, offset=None,
                   with_vectors=False, with_payload=True):
            if offset:  # one page only
                return ([], None)
            vec = [0.1] * 384 if with_vectors else None
            pt = types.SimpleNamespace(id="pt-1", vector=vec, payload={"photo_id": 1})
            return ([pt], None)

    monkeypatch.setattr(bm_mod, "QdrantClient", _FakeQdrant)
    bm = bm_mod.BackupManager(backup_dir=str(tmp_path / "backups"))
    await bm._backup_qdrant(Path(tmp_path))

    saved = json.loads((Path(tmp_path) / "qdrant_points.json").read_text())
    assert saved, "no points written"
    assert saved[0]["vector"] is not None, "backup stored a NULL vector — unrestorable"
    assert len(saved[0]["vector"]) == 384


# --------------------------------------------------------------------------- #
# #2 — Deleting files on disk + rescanning must purge orphaned Qdrant vectors.
# --------------------------------------------------------------------------- #
def test_rescan_purges_orphaned_qdrant_points(tmp_path):
    from app.folder_scanner import FolderScanner

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    folder = tmp_path / "imgs"
    folder.mkdir()
    img = folder / "a.jpg"
    _make_image(img)

    fake = _RecordingQdrant()
    scanner = FolderScanner(qdrant_client=fake, qdrant_collection="embeddings")
    ids, _ = scanner.scan_folder(str(folder), session)
    pid = ids[0]
    session.add(Embedding(
        photo_id=pid, embedding_model="dinov2_vits14",
        vector_dimension=384, qdrant_point_id="pt-1",
    ))
    session.commit()

    os.remove(str(img))                 # user deletes the file on disk
    scanner.scan_folder(str(folder), session)

    assert "pt-1" in fake.deleted_points, "orphaned Qdrant vector was not purged"


# --------------------------------------------------------------------------- #
# #4 — recover_from_checkpoint must actually re-queue the pending photos
#      (not leave a permanent "processing" ghost job).
# --------------------------------------------------------------------------- #
async def test_recover_from_checkpoint_requeues_pending(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock
    from app.job_queue import JobQueueManager

    url, engine = _file_db(tmp_path)
    Session = sessionmaker(bind=engine)
    s = Session()
    s.add(JobQueue(job_id="old", status="processing", total_photos=3, processed_photos=1))
    for i in range(2):  # two photos still pending
        p = Photo(filename=f"p{i}.jpg", file_path=f"/x/p{i}.jpg",
                  file_size=1, mime_type="image/jpeg")
        s.add(p)
        s.flush()
        s.add(ProcessingState(photo_id=p.id, status="pending"))
    s.commit()
    s.close()

    mgr = JobQueueManager(database_url=url)
    spy = AsyncMock(return_value=True)
    monkeypatch.setattr(mgr, "process_photo", spy)

    await mgr.recover_from_checkpoint()
    await asyncio.sleep(0)  # let any scheduled tasks start

    assert spy.await_count == 2, (
        f"recovery re-queued {spy.await_count} photos; the 2 pending ones must resume"
    )


# --------------------------------------------------------------------------- #
# #5 — process_photo must be idempotent: a photo that already has an embedding
#      must not get a second one (double /process-pending → self-duplicates).
# --------------------------------------------------------------------------- #
async def test_process_photo_does_not_duplicate_embedding(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    import app.main as main_mod
    from app.job_queue import JobQueueManager

    url, engine = _file_db(tmp_path)
    Session = sessionmaker(bind=engine)
    s = Session()
    p = Photo(filename="p.jpg", file_path="/x/p.jpg", file_size=1, mime_type="image/jpeg")
    s.add(p)
    s.flush()
    pid = p.id
    s.add(ProcessingState(photo_id=pid, status="pending"))
    s.add(Embedding(photo_id=pid, embedding_model="dinov2_vits14",
                    vector_dimension=384, qdrant_point_id="already-here"))
    s.commit()
    s.close()

    mgr = JobQueueManager(database_url=url)
    monkeypatch.setattr(mgr.embedding_generator, "generate", AsyncMock(return_value=[0.1] * 384))
    monkeypatch.setattr(mgr.metadata_extractor, "extract", AsyncMock(return_value={}))
    mgr.qdrant_client = MagicMock()
    monkeypatch.setattr(main_mod, "notify_embeddings_changed", lambda *a, **k: None)

    mgr.create_job("j1", 1)
    await mgr.process_photo("j1", pid)

    check = Session()
    count = check.query(Embedding).filter(Embedding.photo_id == pid).count()
    check.close()
    assert count == 1, f"photo got {count} embeddings — duplicate vector created"


# --------------------------------------------------------------------------- #
# #6 — SQLite backup must capture WAL-committed rows (plain file-copy misses
#      data still living in the -wal file).
# --------------------------------------------------------------------------- #
async def test_sqlite_backup_captures_wal_committed_rows(tmp_path, monkeypatch):
    from app import backup_manager as bm_mod

    db_path = tmp_path / "live.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.execute("INSERT INTO t VALUES (42)")
    conn.commit()  # committed to -wal, NOT yet checkpointed into live.db

    try:
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
        bm = bm_mod.BackupManager(backup_dir=str(tmp_path / "backups"))
        out = tmp_path / "out"
        out.mkdir()
        await bm._backup_postgresql(out)

        # Open ONLY the copied database.db (no -wal alongside it).
        bconn = sqlite3.connect(str(out / "database.db"))
        rows = bconn.execute("SELECT x FROM t").fetchall()
        bconn.close()
        assert rows == [(42,)], "backup missed a WAL-committed row"
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# #3 — Native launcher lifecycle robustness helpers.
# --------------------------------------------------------------------------- #
def test_app_already_running_detects_live_server():
    import http.server
    import threading
    from app import native

    port = native.find_free_port()

    class _H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", port), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert native.app_already_running(port) is True
    finally:
        srv.shutdown()

    assert native.app_already_running(native.find_free_port()) is False


def test_stop_sidecar_force_kills_when_terminate_hangs():
    from app import native

    class _StuckProc:
        def __init__(self):
            self.terminated = self.killed = False

        def poll(self):
            return None  # never exits on its own

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            raise __import__("subprocess").TimeoutExpired(cmd="qdrant", timeout=timeout)

        def kill(self):
            self.killed = True

    proc = _StuckProc()
    native.stop_sidecar(proc)
    assert proc.terminated and proc.killed, "must escalate terminate -> kill"


def test_kill_stray_sidecars_invokes_pkill(monkeypatch):
    from app import native

    calls = []
    monkeypatch.setattr(native.subprocess, "run",
                        lambda *a, **k: calls.append(a[0]) or types.SimpleNamespace(returncode=0))
    native.kill_stray_sidecars("/some/bundle/qdrant")
    assert any("pkill" in c and "/some/bundle/qdrant" in c for c in calls), \
        "stale sidecars from our bundle must be reaped before launch"


# --------------------------------------------------------------------------- #
# #7 — Cache removal on delete must be serialized by the recompute lock, so it
#      can't race an in-flight incremental add running on the executor thread.
# --------------------------------------------------------------------------- #
async def test_remove_photos_index_is_lock_serialized(monkeypatch):
    import app.main as m

    # The recompute lock is a lazily-built module global; reset it so it binds
    # to THIS test's event loop (a prior test may have created it on another).
    m._sim_recompute_lock = None
    called = []
    monkeypatch.setattr(m, "_remove_photos_from_cache", lambda pids: called.append(pids))

    lock = m._get_recompute_lock()
    await lock.acquire()
    try:
        task = asyncio.create_task(m.remove_photos_from_index({7}))
        await asyncio.sleep(0.05)
        assert called == [], "removal ran while the recompute lock was held (race)"
    finally:
        lock.release()
    await task
    assert called == [{7}], "removal must run once the lock frees"
