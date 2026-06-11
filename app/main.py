import os

# Pin BLAS thread pools to 1 BEFORE numpy/torch import — NumPy's bundled
# OpenBLAS (pthreads) otherwise warns and risks a nested-parallel deadlock
# when called from inside PyTorch's OpenMP region. The Dockerfile sets the
# same vars at the image level; this block protects non-Docker runs too.
# setdefault so an operator override still wins.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import asyncio
import uuid
import logging
import traceback
from datetime import datetime
from typing import Optional, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from alembic.config import Config
from alembic.command import upgrade
from app.job_queue import JobQueueManager
from app.folder_scanner import FolderScanner
from app.thumbnail import ThumbnailService
from app.similarity_search import SimilarityGroupService
from app.models import Photo
from app.backup_manager import BackupManager
from app.validators import (
    validate_pagination,
    validate_similarity_filters,
    validate_photo_id,
    validate_thumbnail_size,
)
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time

logger = logging.getLogger(__name__)

app = FastAPI(title="App API")

# CORS. The UI's host port is configurable (REACT_PORT in start.sh), so a
# fixed allowlist breaks the moment someone runs the UI on a non-default port
# (e.g. 3001 because 3000 was taken) — every browser fetch then fails with an
# opaque "Failed to fetch". For this self-hosted, single-user tool we instead
# allow ANY localhost/127.0.0.1 origin on any port via a regex. An explicit
# extra allowlist can still be supplied via CORS_ORIGINS (comma-separated) for
# non-localhost deployments.
_cors_extra = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_extra,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics for monitoring
request_count = Counter(
    'fastapi_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)
request_duration = Histogram(
    'fastapi_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint']
)
active_requests = Gauge(
    'fastapi_active_requests',
    'Number of active HTTP requests'
)
errors_total = Counter(
    'fastapi_errors_total',
    'Total errors',
    ['error_type']
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to track request metrics for Prometheus."""
    active_requests.inc()
    start_time = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        request_duration.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        return response
    except Exception as e:
        errors_total.labels(error_type=type(e).__name__).inc()
        raise
    finally:
        active_requests.dec()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler so unhandled errors return structured JSON instead of 500 HTML."""
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    errors_total.labels(error_type=type(exc).__name__).inc()
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return HTTPException errors in a consistent JSON envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": str(request.url.path),
        },
    )


job_queue_manager = None
backup_manager = None
thumbnail_service = ThumbnailService()
# Alias used by similarity endpoints and tests
thumbnail_generator = thumbnail_service
similarity_group_service = SimilarityGroupService()


def run_migrations():
    """Run Alembic migrations on startup to ensure schema is up-to-date.

    Skipped for SQLite (the native app): the Alembic revision chain targets
    Postgres, so on SQLite we rely on init_db()/create_all to build the schema
    (the models are SQLite-safe). This avoids a noisy, always-failing upgrade."""
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/app_db")
    if not database_url or database_url.startswith("sqlite"):
        return
    try:
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)
        upgrade(alembic_cfg, "head")
    except Exception as e:
        print(f"Warning: Migration failed: {e}")


@app.on_event("startup")
async def startup_event():
    """Run migrations and recover job queue state on application startup."""
    global job_queue_manager, backup_manager
    run_migrations()
    # Create any tables that aren't covered by Alembic migrations yet
    # (e.g. job_queue, which is defined in models.py but has no migration).
    from app.database import init_db, init_qdrant_collection
    init_db()
    init_qdrant_collection()
    # Initialize job queue manager and recover from last checkpoint
    job_queue_manager = JobQueueManager()
    await job_queue_manager.recover_from_checkpoint()
    # Initialize backup manager for disaster recovery
    backup_manager = BackupManager()
    await backup_manager.schedule_automated_backups()
    # Eagerly compute similarity matrix so first request is fast
    await _recompute_sim_cache()


@app.post("/backup/manual")
async def trigger_manual_backup():
    """Trigger an immediate backup of PostgreSQL and Qdrant data."""
    if backup_manager is None:
        return JSONResponse(status_code=503, content={"error": "Backup manager not initialized"})
    try:
        backup_id = await backup_manager.create_backup()
        return JSONResponse(status_code=202, content={
            "backup_id": backup_id,
            "message": "Backup initiated",
            "status": "in_progress"
        })
    except Exception as e:
        logger.error("Error creating backup: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={
            "error": "Failed to create backup",
            "detail": str(e)
        })


@app.get("/backup/status")
async def get_backup_status():
    """Get status of recent backups and recovery options."""
    if backup_manager is None:
        return JSONResponse(status_code=503, content={"error": "Backup manager not initialized"})
    try:
        status = await backup_manager.get_backup_status()
        return JSONResponse(status_code=200, content=status)
    except Exception as e:
        logger.error("Error getting backup status: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={
            "error": "Failed to get backup status",
            "detail": str(e)
        })


@app.post("/backup/recover/{backup_id}")
async def recover_from_backup(backup_id: str):
    """Recover PostgreSQL and Qdrant data from a specific backup."""
    if backup_manager is None:
        return JSONResponse(status_code=503, content={"error": "Backup manager not initialized"})
    try:
        success = await backup_manager.restore_backup(backup_id)
        if success:
            return JSONResponse(status_code=200, content={
                "backup_id": backup_id,
                "message": "Recovery completed successfully",
                "status": "recovered"
            })
        else:
            return JSONResponse(status_code=400, content={
                "error": "Backup not found or recovery failed",
                "backup_id": backup_id
            })
    except Exception as e:
        logger.error("Error recovering from backup %s: %s", backup_id, e, exc_info=True)
        return JSONResponse(status_code=500, content={
            "error": "Failed to recover from backup",
            "detail": str(e)
        })


@app.post("/process-pending")
async def process_pending_photos():
    """Queue all photos with pending processing state for embedding generation."""
    if job_queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Job queue not initialized"})
    # Already processing? Refuse rather than queue the same pending photos under
    # a second job — two batches racing the same photo create duplicate
    # embeddings (the idempotency check can't catch a concurrent first-embed).
    if job_queue_manager.active_jobs:
        return JSONResponse(status_code=409, content={
            "error": "Processing already in progress",
            "active_jobs": list(job_queue_manager.active_jobs.keys()),
        })
    from app.models import Photo as _Photo, ProcessingState as _PS
    session = job_queue_manager.SessionLocal()
    try:
        pending = (
            session.query(_Photo.id)
            .join(_PS, _PS.photo_id == _Photo.id)
            .filter(_PS.status == "pending")
            .all()
        )
        photo_ids = [row[0] for row in pending]
    finally:
        session.close()

    if not photo_ids:
        return JSONResponse(status_code=200, content={"message": "No pending photos", "queued": 0})

    job_id = str(uuid.uuid4())
    job_queue_manager.create_job(job_id, len(photo_ids))
    for pid in photo_ids:
        asyncio.create_task(job_queue_manager.process_photo(job_id, pid))
    return JSONResponse(status_code=202, content={
        "job_id": job_id,
        "message": "Processing started",
        "queued": len(photo_ids),
    })


@app.post("/stop-processing")
async def stop_processing():
    """Cancel all active processing jobs. Pending photos remain in pending state for later resume."""
    if job_queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Job queue not initialized"})
    cancelled = await job_queue_manager.cancel_all_jobs()
    return JSONResponse(status_code=200, content={
        "message": "Processing stopped",
        "cancelled_jobs": cancelled,
    })


@app.get("/stats")
async def get_stats():
    """Return high-level processing stats for the UI progress panel.
    Includes similarity_index sub-object so the UI can render
    "groups updated 12s ago" instead of guessing."""
    from app.database import SessionLocal as _SL
    from app.models import Photo as _Photo, Embedding as _Emb, ProcessingState as _PS
    session = _SL()
    try:
        total_photos = session.query(_Photo).count()
        total_embeddings = session.query(_Emb).count()
        completed = session.query(_PS).filter(_PS.status == "completed").count()
        pending = session.query(_PS).filter(_PS.status == "pending").count()
        failed = session.query(_PS).filter(_PS.status == "failed").count()
        return {
            "photos": total_photos,
            "embeddings": total_embeddings,
            "completed": completed,
            "pending": pending,
            "failed": failed,
            "similarity_index": dict(_sim_index_info),
        }
    finally:
        session.close()


@app.get("/health")
async def health_check():
    """Health check endpoint for service verification."""
    return JSONResponse(status_code=200, content={"status": "healthy"})


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint for monitoring."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/job-queue/status")
async def get_job_queue_status():
    """Get current job queue status and checkpoint information."""
    if job_queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Job queue not initialized"})
    status = await job_queue_manager.get_status()
    return JSONResponse(status_code=200, content=status)


TRASH_DIR = os.getenv("TRASH_DIR", os.path.expanduser("~/.photo-gaze-trash"))

# Public base URL the BROWSER uses to reach this backend. Thumbnail links in
# the /similarity-groups payload are absolute (the React app and the backend
# are served on different host ports), so they must point at the host-published
# backend port. Configurable so changing FASTAPI_PORT doesn't 404 every
# thumbnail — docker-compose sets it to http://localhost:${FASTAPI_PORT}.
BACKEND_PUBLIC_URL = os.getenv("BACKEND_PUBLIC_URL", "http://localhost:8000").rstrip("/")


# Mime types we prefer to keep when ranking duplicates (universally
# decodable across viewers and OSes). Lifted to module scope so the
# auto-deduplicate planner can reuse the same ranking the per-group
# clustering uses.
_PREFERRED_MIME_TYPES = {"image/jpeg", "image/png"}


def _taken_timestamp(meta: dict) -> float:
    """Best estimate of when a photo was TAKEN — the provenance tiebreaker
    among equal-quality duplicates. Resolution order:
        EXIF DateTimeOriginal → file mtime (both via _read_image_info)
        → Photo.uploaded_at → +inf.
    Reads the file only when it still exists on disk; _read_image_info LRU-
    caches the parse, so repeated ranking of the same photo is cheap."""
    fpath = meta.get("file_path") or ""
    if fpath and os.path.isfile(fpath):
        try:
            _, _, created_iso = _read_image_info(fpath)
        except Exception:
            created_iso = None
        if created_iso:
            try:
                return datetime.fromisoformat(created_iso).timestamp()
            except Exception:
                pass
    uploaded = meta.get("uploaded_at")
    if uploaded:
        try:
            return datetime.fromisoformat(uploaded).timestamp()
        except Exception:
            pass
    return float("inf")


def _keeper_key(meta: dict):
    """Unified ranking for the photo to KEEP in a duplicate set. Used by BOTH
    the group view ("★ Best") and auto-dedupe, so they always agree on which
    copy survives. The keeper sorts FIRST under an ascending sort. Preference:

      1. Highest quality — largest effective file size (+20% bonus for the
         universal JPEG/PNG formats). Less compression = more detail; this
         is the dominant signal so we never keep a worse copy over a better.
      2. Earliest-taken — among equal-quality (typically byte-identical)
         copies, the one taken first is most likely the original. See
         _taken_timestamp.
      3. Shortest filename — "photo.jpg" beats "photo (1).jpg".
      4. Lexically-smallest path — final determinism.
    """
    return (
        -_effective_size(meta),
        _taken_timestamp(meta),
        len(meta.get("filename") or ""),
        meta.get("file_path") or "",
    )


def _read_manifest(path: str):
    """Read a trash manifest. Returns [] on any failure (corrupt file,
    missing, partial write). Caller decides whether to log."""
    try:
        with open(path) as f:
            data = __import__("json").load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_manifest(path: str, entries: list) -> None:
    import json as _json
    with open(path, "w") as f:
        _json.dump(entries, f, indent=2)


def _is_inside_trash(candidate: str) -> bool:
    """Reject any path that resolves outside TRASH_DIR. Defends the
    recover endpoint against caller-supplied paths that try to move
    arbitrary files via path traversal."""
    trash_abs = os.path.realpath(os.path.abspath(TRASH_DIR))
    abs_candidate = os.path.realpath(os.path.abspath(candidate))
    return abs_candidate == trash_abs or abs_candidate.startswith(trash_abs + os.sep)


@app.get("/trash")
async def list_trash():
    """List every photo currently in the dedupe trash with the original
    path it would be restored to. The UI uses this to render the recovery
    page; each item carries `trash_path` as a stable identifier the
    /trash/recover endpoint expects.

    Skips manifest entries whose trash file no longer exists on disk
    (e.g. user emptied Finder's Trash manually) — they're invisible from
    the UI's perspective and silently pruned on the next recover call.
    """
    items = []
    if not os.path.isdir(TRASH_DIR):
        return {"items": items, "trash_dir": TRASH_DIR}

    for name in sorted(os.listdir(TRASH_DIR)):
        if not name.endswith("_manifest.json"):
            continue
        manifest_path = os.path.join(TRASH_DIR, name)
        ts = name[: -len("_manifest.json")]   # "20260501_120000"
        for entry in _read_manifest(manifest_path):
            trash_path = entry.get("trash")
            original = entry.get("original")
            if not trash_path or not os.path.isfile(trash_path):
                continue  # file gone: skip — recover can't help anyway
            items.append({
                "trash_path": trash_path,
                "original_path": original,
                "filename": os.path.basename(original or trash_path),
                "trashed_at": ts,
                "file_size": os.path.getsize(trash_path),
            })
    return {"items": items, "trash_dir": TRASH_DIR}


def _restore_db_and_qdrant_from_snapshot(session, qdrant_client, entry: dict) -> dict:
    """Recreate Photo + ProcessingState + Embedding rows + Qdrant point
    from a v2 manifest entry. Returns a status dict the recover endpoint
    surfaces back to the caller.

    Behavior:
      - v1 (legacy) entries with no "photo" snapshot are a no-op here;
        the caller will rely on the next folder rescan to re-ingest.
      - If a Photo with the same file_path already exists in DB
        (re-imported from elsewhere) we skip DB writes — the file move
        already did the user-visible work.
      - Vector dimension mismatch: skip the Qdrant upsert but still
        rewrite the Photo + ProcessingState rows. Embedding row is
        skipped since it'd be referentially broken.
      - Each step is independently catch-and-warn so a partial failure
        (e.g. Qdrant down) doesn't roll back the file move.
    """
    from app.models import Photo as _Photo, Embedding as _Emb, ProcessingState as _PS

    photo_snap = entry.get("photo")
    if not photo_snap:
        return {"db_restored": False, "reason": "legacy_v1_entry"}

    fp = photo_snap.get("file_path")
    if not fp:
        return {"db_restored": False, "reason": "snapshot_missing_file_path"}

    # Idempotency: if a row for this file_path already exists, don't touch.
    existing = session.query(_Photo).filter(_Photo.file_path == fp).first()
    if existing is not None:
        return {"db_restored": False, "reason": "photo_row_already_exists",
                "existing_photo_id": existing.id}

    def _parse(ts):
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return None

    new_photo = _Photo(
        filename=photo_snap.get("filename") or os.path.basename(fp),
        file_path=fp,
        file_size=photo_snap.get("file_size") or 0,
        mime_type=photo_snap.get("mime_type") or "image/unknown",
        file_hash=photo_snap.get("file_hash"),
        uploaded_at=_parse(photo_snap.get("uploaded_at")) or datetime.utcnow(),
        user_id=photo_snap.get("user_id"),
    )
    session.add(new_photo)
    session.flush()  # populate new_photo.id without committing yet

    ps_snap = entry.get("processing_state") or {}
    session.add(_PS(
        photo_id=new_photo.id,
        status=ps_snap.get("status") or "completed",
        extraction_status=ps_snap.get("extraction_status") or "completed",
        embedding_status=ps_snap.get("embedding_status") or "completed",
        error_message=ps_snap.get("error_message"),
        started_at=_parse(ps_snap.get("started_at")),
        completed_at=_parse(ps_snap.get("completed_at")),
    ))

    emb_snap = entry.get("embedding") or {}
    vector = emb_snap.get("vector")
    new_point_id = None
    qdrant_upserted = False
    if vector and qdrant_client is not None:
        from qdrant_client.http.models import PointStruct
        import uuid as _uuid
        new_point_id = str(_uuid.uuid4())
        try:
            qdrant_client.upsert(
                collection_name="embeddings",
                points=[PointStruct(
                    id=new_point_id,
                    vector=list(vector),
                    payload={"photo_id": new_photo.id},
                )],
            )
            qdrant_upserted = True
        except Exception as e:
            logger.warning(
                "Qdrant upsert failed during recovery (photo %s): %s. "
                "Photo + ProcessingState rows still restored; the next "
                "scan will rebuild the embedding.",
                new_photo.id, e,
            )
            new_point_id = None

    session.add(_Emb(
        photo_id=new_photo.id,
        embedding_model=emb_snap.get("embedding_model") or "dinov2_vits14",
        vector_dimension=emb_snap.get("vector_dimension") or (len(vector) if vector else 384),
        qdrant_point_id=new_point_id,
    ))

    return {
        "db_restored": True,
        "photo_id": new_photo.id,
        "qdrant_upserted": qdrant_upserted,
    }


@app.get("/trash/thumbnail")
async def get_trash_thumbnail(path: str, size: int = 240):
    """Return a JPEG thumbnail for a file currently in the trash.

    Used by the Trash page to preview the photo without restoring it.
    The path-traversal guard (_is_inside_trash) rejects any path that
    resolves outside TRASH_DIR — the endpoint is read-only but it
    would still be a leak to let arbitrary host paths be rendered.

    Cache key is the md5 of the trash path. Trash filenames are
    timestamped + photo-id-prefixed, so they're stable per trashed
    file; recovery + re-trashing produces a new path → new key.
    """
    if not path:
        return JSONResponse(status_code=400, content={"error": "path is required"})
    if not _is_inside_trash(path):
        return JSONResponse(status_code=400, content={
            "error": "path is not inside the trash directory",
        })
    if not os.path.isfile(path):
        return JSONResponse(status_code=404, content={"error": "file not found"})

    import hashlib as _hl
    cache_key = _hl.md5(path.encode("utf-8")).hexdigest()
    try:
        thumb_path = thumbnail_service.get_thumbnail(
            path, cache_key, size=(size, size),
        )
        return FileResponse(thumb_path, media_type="image/jpeg")
    except Exception as e:
        logger.error("Trash thumbnail generation failed for %s: %s", path, e)
        return JSONResponse(status_code=500, content={
            "error": "Failed to generate thumbnail",
            "detail": str(e),
        })


@app.post("/trash/recover")
async def recover_from_trash(request: Request):
    """Move selected photos back from trash to their original paths and
    rebuild Postgres + Qdrant from the manifest snapshot, so the photo
    is back in the index immediately — no rescan, no re-embedding.

    Backward compatibility: v1 manifest entries (file-only) still
    recover the file; their DB/Qdrant state is rebuilt by the next
    folder rescan exactly as before.

    Path-traversal-defended: every trash_path is rejected if it doesn't
    resolve under TRASH_DIR before any file or DB write.
    """
    if job_queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Service not initialized"})

    body = await request.json()
    trash_paths = body.get("trash_paths", [])
    if not trash_paths:
        return JSONResponse(status_code=400, content={"error": "trash_paths is required"})

    import shutil

    requested_abs: set = set()
    errors: list = []
    for tp in trash_paths:
        if not _is_inside_trash(tp):
            errors.append({"trash_path": tp, "error": "not inside trash directory"})
            continue
        requested_abs.add(os.path.realpath(os.path.abspath(tp)))

    recovered: list = []
    if not os.path.isdir(TRASH_DIR):
        return {"recovered": 0, "items": [], "errors": errors or None}

    qc = job_queue_manager.qdrant_client
    session = job_queue_manager.SessionLocal()
    db_dirty = False
    try:
        for name in sorted(os.listdir(TRASH_DIR)):
            if not name.endswith("_manifest.json"):
                continue
            manifest_path = os.path.join(TRASH_DIR, name)
            entries = _read_manifest(manifest_path)
            if not entries:
                continue
            kept: list = []
            changed = False
            for entry in entries:
                trash_path = entry.get("trash")
                trash_abs = (os.path.realpath(os.path.abspath(trash_path))
                             if trash_path else None)
                if trash_abs not in requested_abs:
                    kept.append(entry)
                    continue

                original = entry.get("original")
                if not trash_path or not os.path.isfile(trash_path):
                    errors.append({"trash_path": trash_path, "error": "file missing"})
                    changed = True
                    continue
                if not original:
                    errors.append({"trash_path": trash_path,
                                   "error": "no original path recorded"})
                    kept.append(entry)
                    continue
                if os.path.exists(original):
                    errors.append({
                        "trash_path": trash_path,
                        "error": f"a file already exists at {original}",
                    })
                    kept.append(entry)
                    continue

                # 1) Move file back.
                try:
                    os.makedirs(os.path.dirname(original), exist_ok=True)
                    shutil.move(trash_path, original)
                except Exception as e:
                    errors.append({"trash_path": trash_path, "error": str(e)})
                    kept.append(entry)
                    continue

                # 2) Rebuild DB + Qdrant from the snapshot. v1 entries are
                # a no-op; their state is rebuilt by the next folder rescan.
                try:
                    db_status = _restore_db_and_qdrant_from_snapshot(
                        session, qc, entry
                    )
                    if db_status.get("db_restored"):
                        db_dirty = True
                except Exception as e:
                    logger.error(
                        "Snapshot-based DB restore failed for %s: %s. "
                        "File is back on disk; rescan will re-ingest.",
                        original, e,
                    )
                    db_status = {"db_restored": False, "reason": "exception"}

                recovered.append({
                    "trash_path": trash_path,
                    "restored_to": original,
                    **db_status,
                })
                changed = True

            if not changed:
                continue
            if kept:
                _write_manifest(manifest_path, kept)
            else:
                try:
                    os.remove(manifest_path)
                except OSError as e:
                    logger.warning("Could not remove empty manifest %s: %s",
                                   manifest_path, e)

        if db_dirty:
            session.commit()
    finally:
        session.close()

    if db_dirty:
        # Refresh similarity index so the recovered photos show up in
        # /similarity-groups without waiting for the debounce.
        await _recompute_sim_cache()

    return {
        "recovered": len(recovered),
        "items": recovered,
        "errors": errors or None,
    }


# Trash manifest schema versions:
#   1 — legacy: {photo_id, original, trash}. File-only recovery.
#   2 — full snapshot: above + {photo, processing_state, embedding{vector}}.
#       Recovery rebuilds the Postgres rows AND the Qdrant point WITHOUT
#       re-running DINOv2 or re-extracting metadata. Saves ~1–2s per
#       photo on recovery (a v2 manifest entry is ~6 KB; for thousands
#       of photos this adds tens of MB to the trash dir, which is fine).
TRASH_MANIFEST_SCHEMA = 2


def _capture_photo_snapshot(session, qdrant_client, photo_id: int) -> dict:
    """Build the v2 snapshot for a photo BEFORE its rows are deleted.

    Pulls Photo, ProcessingState, and Embedding rows and the actual vector
    from Qdrant. Each missing piece falls through to None so a partially-
    ingested photo can still be trashed and recovered to whatever state it
    had. Pure read; no writes.
    """
    from app.models import Photo as _Photo, Embedding as _Emb, ProcessingState as _PS

    snap: dict = {}

    p = session.query(_Photo).filter(_Photo.id == photo_id).first()
    if p:
        snap["photo"] = {
            "filename": p.filename,
            "file_path": p.file_path,
            "file_size": p.file_size,
            "mime_type": p.mime_type,
            "file_hash": p.file_hash,
            "uploaded_at": p.uploaded_at.isoformat() if p.uploaded_at else None,
            "user_id": p.user_id,
        }

    ps = session.query(_PS).filter(_PS.photo_id == photo_id).first()
    if ps:
        snap["processing_state"] = {
            "status": ps.status,
            "extraction_status": ps.extraction_status,
            "embedding_status": ps.embedding_status,
            "error_message": ps.error_message,
            "started_at": ps.started_at.isoformat() if ps.started_at else None,
            "completed_at": ps.completed_at.isoformat() if ps.completed_at else None,
        }

    emb = session.query(_Emb).filter(_Emb.photo_id == photo_id).first()
    if emb:
        emb_snap = {
            "embedding_model": emb.embedding_model,
            "vector_dimension": emb.vector_dimension,
            "vector": None,
        }
        # Pull the actual vector — without it, recovery has to re-embed.
        if emb.qdrant_point_id and qdrant_client is not None:
            try:
                records = qdrant_client.retrieve(
                    collection_name="embeddings",
                    ids=[emb.qdrant_point_id],
                    with_vectors=True,
                )
                if records and records[0].vector is not None:
                    # Coerce to plain Python floats — JSON can't serialize
                    # numpy / float32 directly.
                    emb_snap["vector"] = [float(x) for x in records[0].vector]
            except Exception as e:
                logger.warning(
                    "Qdrant retrieve failed for point %s (photo %d); "
                    "snapshot will lack vector: %s",
                    emb.qdrant_point_id, photo_id, e,
                )
        snap["embedding"] = emb_snap

    return snap


async def _execute_dedupe(session, photo_ids: list) -> dict:
    """Snapshot + move the listed photos to trash, write a v2 manifest,
    delete from Qdrant + Postgres, and refresh the similarity index.
    Returns a dict matching the legacy /deduplicate response shape.

    The snapshot phase happens BEFORE deletion so recovery doesn't have
    to re-run DINOv2 or re-extract metadata — see _capture_photo_snapshot.

    Shared between /deduplicate (manual) and /auto-deduplicate (sweep).
    """
    from app.models import Photo as _Photo, Embedding as _Emb, ProcessingState as _PS
    import shutil

    qc = job_queue_manager.qdrant_client if job_queue_manager else None

    photos = session.query(_Photo).filter(_Photo.id.in_(photo_ids)).all()
    file_paths = {p.id: p.file_path for p in photos}

    # Capture full snapshots BEFORE any deletion, so even if the file move
    # below fails, the source-of-truth rows are still on disk + DB.
    snapshots: dict = {pid: _capture_photo_snapshot(session, qc, pid)
                       for pid in photo_ids}

    os.makedirs(TRASH_DIR, exist_ok=True)
    moved_entries = []
    move_errors = []
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    # Track which photo_ids it's safe to purge from the DB / Qdrant.
    # SUCCESS: file moved into trash. SAFE_PURGE.
    # MISSING: file already gone from disk. SAFE_PURGE (cleanup orphan).
    # MOVE_FAILED: file still at original path. DO NOT PURGE — otherwise
    # the user is left with a file on disk and no DB/index record AND no
    # manifest entry to recover from. We surface the per-photo error
    # and let the caller retry.
    ids_to_purge: list = []
    for pid, src in file_paths.items():
        if not src or not os.path.isfile(src):
            ids_to_purge.append(pid)  # nothing left on disk → clean DB rows
            continue
        basename = os.path.basename(src)
        dest = os.path.join(TRASH_DIR, f"{ts}_{pid}_{basename}")
        try:
            shutil.move(src, dest)
            moved_entries.append({
                "schema_version": TRASH_MANIFEST_SCHEMA,
                "photo_id": pid,
                "original": src,
                "trash": dest,
                "trashed_at": datetime.utcnow().isoformat(),
                **snapshots.get(pid, {}),
            })
            ids_to_purge.append(pid)
        except Exception as e:
            move_errors.append({"photo_id": pid, "error": str(e)})

    if moved_entries:
        import json as _json
        manifest_path = os.path.join(TRASH_DIR, f"{ts}_manifest.json")
        existing = []
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                existing = _json.load(f)
        existing.extend(moved_entries)
        with open(manifest_path, "w") as f:
            _json.dump(existing, f, indent=2)

    if not ids_to_purge:
        # Nothing was successfully moved AND nothing was already missing —
        # leave the DB / Qdrant / index alone.
        return {
            "deleted": 0,
            "moved_to_trash": 0,
            "trash_dir": TRASH_DIR,
            "errors": move_errors if move_errors else None,
        }

    qdrant_point_ids = [
        qid for (qid,) in session.query(_Emb.qdrant_point_id)
        .filter(_Emb.photo_id.in_(ids_to_purge))
        .filter(_Emb.qdrant_point_id.isnot(None))
        .all()
    ]
    if qdrant_point_ids:
        try:
            job_queue_manager.qdrant_client.delete(
                collection_name="embeddings",
                points_selector=qdrant_point_ids,
            )
        except Exception as e:
            logger.warning("Qdrant delete failed: %s", e)

    session.query(_Emb).filter(_Emb.photo_id.in_(ids_to_purge)).delete(synchronize_session=False)
    session.query(_PS).filter(_PS.photo_id.in_(ids_to_purge)).delete(synchronize_session=False)
    deleted = session.query(_Photo).filter(_Photo.id.in_(ids_to_purge)).delete(synchronize_session=False)
    session.commit()

    # Incremental cache update — deletion only removes nodes/edges, so we
    # filter the in-memory index instead of a full Qdrant re-scroll + re-search
    # (which would block this request for minutes on a 300k-photo collection).
    await remove_photos_from_index(set(ids_to_purge))

    return {
        "deleted": deleted,
        "moved_to_trash": len(moved_entries),
        "trash_dir": TRASH_DIR,
        "errors": move_errors if move_errors else None,
    }


@app.post("/deduplicate")
async def deduplicate_photos(request: Request):
    """Move selected photos to ${TRASH_DIR} and remove from DB + Qdrant."""
    if job_queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Service not initialized"})

    body = await request.json()
    photo_ids = body.get("photo_ids", [])
    if not photo_ids:
        return JSONResponse(status_code=400, content={"error": "photo_ids is required"})

    session = job_queue_manager.SessionLocal()
    try:
        return await _execute_dedupe(session, photo_ids)
    finally:
        session.close()


def _is_under(child: str, parent_abs: str) -> bool:
    """True iff `child` (taken AS-IS, only abspath-normalized) equals
    `parent_abs` or is strictly inside it. parent_abs must already be a
    realpath. Uses the trailing-separator trick so /a/bc is NOT
    considered inside /a/b.

    We deliberately do NOT realpath `child`. A symlink in the keep
    folder pointing to a file elsewhere is at the path the user placed
    it — that's what they see and curate. realpath would resolve to
    the target's location and miscategorize the alias as an outsider,
    causing auto-dedupe to delete the user's curated entry. Trash-side
    path-traversal defense uses _is_inside_trash, which DOES realpath
    on purpose; that's a security check, not a user-intent check."""
    if not child:
        return False
    try:
        child_abs = os.path.abspath(child)
    except OSError:
        return False
    if child_abs == parent_abs:
        return True
    return child_abs.startswith(parent_abs.rstrip(os.sep) + os.sep)


def _plan_auto_dedupe(threshold: float, keep_folder: str) -> dict:
    """Build the action plan for an auto-dedupe sweep.

    Spec: the chosen `keep_folder` defines a "source of truth" zone.
    For each cluster of pure duplicates (connected component above
    `threshold`):

      - If at least one member's file lives under keep_folder, pick the
        single SURVIVOR among the in-keep members by _keeper_key —
        highest quality (largest effective size), with earliest-taken
        (EXIF DateTimeOriginal → mtime → uploaded_at) as the tiebreaker.
        Every other member of the cluster is deleted, INCLUDING any other
        in-folder duplicates — duplicates within the keep folder are still
        duplicates, and only one canonical copy needs to survive.
        See _keeper_key for the full ranking and tiebreakers.
      - If no member is under keep_folder → skip the whole cluster
        (we never make a destructive choice without an explicit
        anchor in the user's chosen folder).
      - Singletons (only one member in the connected component) →
        nothing to do.

    Why connected components and not greedy single-link clustering:
    Qdrant's HNSW + top_k limit can produce asymmetric adjacency, and
    greedy iteration is order-dependent. With greedy, an outsider
    duplicate of an in-keep photo can be missed if its only neighbours
    were "visited" by a prior cluster. BFS through the adjacency
    catches every transitive duplicate of an in-keep anchor. See the
    regression tests test_outsider_pure_duplicate_missed_via_asymmetric_adjacency
    and test_chain_of_duplicates_outside_keep_all_deleted.

    Returns:
        {
          "groups_processed": int,    # components with deletions
          "groups_skipped": int,      # components with no in-keep anchor
          "to_delete": [photo_id...],
          "kept": [photo_id...],
          "groups": [
            {"kept_ids": [...], "kept_paths": [...],
             "delete_ids": [...], "delete_paths": [...]}
          ],
        }
    """
    EMPTY_PLAN = {
        "groups_processed": 0, "groups_skipped": 0,
        "to_delete": [], "kept": [], "groups": [],
    }

    if threshold > 1.0:
        return EMPTY_PLAN

    cache_data, photo_meta = _get_cached_data()
    if cache_data is None:
        return EMPTY_PLAN

    photo_ids = cache_data["photo_ids"]
    cache_floor = cache_data.get("cache_threshold", _SIM_CACHE_THRESHOLD)
    if threshold >= 1.0:
        # See _PURE_DUPE_EPSILON: float32 normalize-then-dot returns
        # ~0.9999998 for byte-identical photos. Without this slack a
        # strict s >= 1.0 filter would drop the very pairs we're after.
        threshold = 1.0 - _PURE_DUPE_EPSILON
    effective_threshold = max(threshold, cache_floor)
    keep_abs = os.path.realpath(os.path.abspath(keep_folder))
    n = len(photo_ids)

    # Pre-compute which indices are inside the keep folder.
    in_keep_idx: set = set()
    for idx, pid in enumerate(photo_ids):
        meta = photo_meta.get(pid, {}) if photo_meta else {}
        if _is_under(meta.get("file_path") or "", keep_abs):
            in_keep_idx.add(idx)

    # Connected components of the duplicate graph at this threshold.
    # _threshold_components symmetrizes (Qdrant's top-k can drop a back-edge,
    # so an outsider whose only edge points INTO an in-keep anchor is still
    # captured) and is O(active edges), so this scales to 300k-photo
    # collections instead of allocating a set per photo. Components are
    # index-sorted, keeping the plan reproducible. See the regression tests
    # test_outsider_pure_duplicate_missed_via_asymmetric_adjacency and
    # test_chain_of_duplicates_outside_keep_all_deleted.
    plan_groups: list = []
    to_delete: list = []
    kept: list = []
    skipped = 0

    for component in _threshold_components(cache_data, effective_threshold):
        in_comp = [k for k in component if k in in_keep_idx]
        if not in_comp:
            # No anchor in the keep folder → never make a destructive choice.
            # Counted so the UI can report "X groups skipped".
            skipped += 1
            continue
        # Pick the survivor among the in-keep members: highest quality, with
        # earliest-taken as the tiebreaker (see _keeper_key). Same ranking the
        # group view uses, so manual and auto agree on which copy is kept.
        in_comp_with_keys = sorted(
            [
                (
                    _keeper_key(photo_meta.get(photo_ids[k]) or {}),
                    k,
                    photo_ids[k],
                    photo_meta.get(photo_ids[k]) or {},
                )
                for k in in_comp
            ],
            key=lambda t: t[0],
        )
        survivor_idx, survivor_pid, survivor_meta = (
            in_comp_with_keys[0][1],
            in_comp_with_keys[0][2],
            in_comp_with_keys[0][3],
        )
        # Everyone else in the component (in-folder runner-ups + outsiders)
        # is deleted. Skip if only the survivor is left (singleton).
        delete_idx_list = [k for k in component if k != survivor_idx]
        if not delete_idx_list:
            continue
        delete_meta = [
            {"photo_id": photo_ids[k], **(photo_meta.get(photo_ids[k]) or {})}
            for k in delete_idx_list
        ]
        kept_meta = [{"photo_id": survivor_pid, **survivor_meta}]
        plan_groups.append({
            "kept_ids":     [m["photo_id"] for m in kept_meta],
            "kept_paths":   [m.get("file_path") for m in kept_meta],
            "delete_ids":   [m["photo_id"] for m in delete_meta],
            "delete_paths": [m.get("file_path") for m in delete_meta],
        })
        kept.extend(m["photo_id"] for m in kept_meta)
        to_delete.extend(m["photo_id"] for m in delete_meta)

    return {
        "groups_processed": len(plan_groups),
        "groups_skipped": skipped,
        "to_delete": to_delete,
        "kept": kept,
        "groups": plan_groups,
    }


@app.post("/auto-deduplicate")
async def auto_deduplicate(request: Request):
    """Sweep all near-perfect duplicate groups and keep one copy in the
    user-selected folder, deleting the rest.

    Body: {
        "folder_path": str (required) — the folder where the kept copy
                       must live. Photos in OTHER folders that match
                       a cluster anchored here are deleted; duplicates
                       within this folder are reduced to one.
        "threshold":   float (default 1.0) — cluster inclusion floor.
                       1.0 = only pure-duplicate clusters; lower values
                       widen to near-duplicates.
        "dry_run":     bool (default false) — when true, returns the
                       plan without touching files / DB / Qdrant. The
                       UI uses this for the confirmation dialog.
    }
    """
    if job_queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Service not initialized"})

    body = await request.json()
    folder_path = (body.get("folder_path") or "").strip()
    # Defensive parsing: `dict.get(k, default)` returns the default ONLY
    # when the key is absent. `{"threshold": null}` makes .get() return
    # None — float(None) raises and the handler 500s. Likewise non-numeric
    # strings (`"abc"`) raise ValueError. Treat null as "use the default"
    # and turn type errors into a clean 400.
    threshold_raw = body.get("threshold")
    if threshold_raw is None:
        threshold = 1.0
    else:
        try:
            threshold = float(threshold_raw)
        except (TypeError, ValueError):
            return JSONResponse(status_code=400, content={
                "error": f"threshold must be a number, got {threshold_raw!r}",
            })
    dry_run = bool(body.get("dry_run", False))

    if not folder_path:
        return JSONResponse(status_code=400, content={"error": "folder_path is required"})
    if not os.path.isdir(folder_path):
        return JSONResponse(status_code=400, content={
            "error": f"folder_path is not a directory: {folder_path}",
        })
    if threshold > 1.0 or threshold <= 0.0:
        return JSONResponse(status_code=400, content={
            "error": f"threshold must be in (0, 1], got {threshold}",
        })
    # Reject the trash directory itself — keeping duplicates "in the trash"
    # is incoherent (next scan won't see them anyway).
    trash_abs = os.path.realpath(os.path.abspath(TRASH_DIR))
    candidate_abs = os.path.realpath(os.path.abspath(folder_path))
    if candidate_abs == trash_abs or candidate_abs.startswith(trash_abs + os.sep):
        return JSONResponse(status_code=400, content={
            "error": "folder_path cannot be inside the trash directory",
        })

    plan = _plan_auto_dedupe(threshold, folder_path)

    # Execute or short-circuit. Either way the response shape is the
    # same — same keys whether dry_run, empty plan, or real execute.
    if dry_run or not plan["to_delete"]:
        result = {"deleted": 0, "moved_to_trash": 0, "errors": None}
    else:
        session = job_queue_manager.SessionLocal()
        try:
            result = await _execute_dedupe(session, plan["to_delete"])
        finally:
            session.close()

    return {
        "dry_run": dry_run,
        "threshold": threshold,
        "folder_path": folder_path,
        "groups_processed": plan["groups_processed"],
        "groups_skipped": plan["groups_skipped"],
        "kept": plan["kept"],
        "to_delete": plan["to_delete"],
        "groups": plan["groups"],
        "deleted": result.get("deleted", 0),
        "moved_to_trash": result.get("moved_to_trash", 0),
        "errors": result.get("errors"),
    }


@app.get("/browse")
async def browse_directory(path: str = "/"):
    """List subdirectories and image-file counts at a path for the folder picker."""
    path = path or "/"
    if not os.path.isdir(path):
        return JSONResponse(status_code=400, content={"error": f"Not a directory: {path}"})

    image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".heif"}
    entries = []
    image_count = 0
    try:
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if name.startswith("."):
                continue  # skip hidden
            if os.path.isdir(full):
                entries.append({"name": name, "type": "dir"})
            else:
                ext = os.path.splitext(name)[1].lower()
                if ext in image_exts:
                    image_count += 1
    except PermissionError:
        return JSONResponse(status_code=403, content={"error": f"Permission denied: {path}"})

    parent = os.path.dirname(path.rstrip("/")) or "/"
    return {
        "path": path,
        "parent": parent if parent != path else None,
        "dirs": entries,
        "image_count": image_count,
    }


def _count_supported_files(folder_path: str) -> list:
    """Return the set of supported image extensions found in a folder (non-recursive head probe)."""
    supported = {
        ".jpg", ".jpeg", ".jfif", ".png", ".gif", ".bmp", ".webp",
        ".heic", ".heif", ".tiff", ".tif", ".avif", ".ico",
        ".dng", ".cr2", ".nef", ".arw", ".orf", ".rw2", ".pef",
    }
    found = set()
    try:
        for name in os.listdir(folder_path):
            ext = os.path.splitext(name)[1].lower()
            if ext in supported:
                found.add(ext)
    except Exception:
        pass
    return sorted(found)


@app.get("/folders")
async def list_folders():
    """Return registered photo folders."""
    if job_queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Service not initialized"})
    from app.models import FolderPath
    session = job_queue_manager.SessionLocal()
    try:
        rows = session.query(FolderPath).order_by(FolderPath.id.asc()).all()
        return [
            {
                "id": f.id,
                "path": f.path,
                "is_accessible": f.is_accessible,
                "supported_formats_found": f.supported_formats_found or [],
                "created_at": f.created_at.isoformat() if f.created_at else None,
            }
            for f in rows
        ]
    finally:
        session.close()


@app.post("/folders")
async def add_folder(request: Request):
    """Register a new folder to scan. Validates accessibility server-side
    AND refuses to register the trash directory (or any path inside it) —
    indexing the trash would re-ingest just-deleted duplicates."""
    if job_queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Service not initialized"})
    from app.models import FolderPath
    body = await request.json()
    path = (body.get("path") or "").strip()
    if not path:
        return JSONResponse(status_code=400, content={"error": "path is required"})

    # Reject the trash directory and any subpath of it.
    trash_abs = os.path.realpath(os.path.abspath(TRASH_DIR))
    candidate_abs = os.path.realpath(os.path.abspath(path))
    if candidate_abs == trash_abs or candidate_abs.startswith(trash_abs + os.sep):
        return JSONResponse(status_code=400, content={
            "error": "Cannot register a path inside the trash directory",
            "trash_dir": TRASH_DIR,
        })

    is_accessible = os.path.isdir(path) and os.access(path, os.R_OK)
    formats = _count_supported_files(path) if is_accessible else []

    session = job_queue_manager.SessionLocal()
    try:
        existing = session.query(FolderPath).filter(FolderPath.path == path).first()
        if existing:
            existing.is_accessible = is_accessible
            existing.supported_formats_found = formats
            existing.updated_at = datetime.utcnow()
            session.commit()
            folder = existing
        else:
            folder = FolderPath(path=path, is_accessible=is_accessible, supported_formats_found=formats)
            session.add(folder)
            session.commit()
            session.refresh(folder)
        return {
            "id": folder.id,
            "path": folder.path,
            "is_accessible": folder.is_accessible,
            "supported_formats_found": folder.supported_formats_found or [],
        }
    finally:
        session.close()


@app.delete("/folders/{folder_id}")
async def delete_folder(folder_id: int):
    """Remove a folder from the registry AND purge every photo / embedding
    whose file_path lives under that folder. Qdrant points are deleted too.
    """
    if job_queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Service not initialized"})
    from app.models import FolderPath, Photo, Embedding, ProcessingState

    session = job_queue_manager.SessionLocal()
    try:
        folder = session.query(FolderPath).filter(FolderPath.id == folder_id).first()
        if not folder:
            return JSONResponse(status_code=404, content={"error": "Folder not found"})

        # Match any photo whose path is inside the folder (prefix match with
        # trailing separator so /photos/a doesn't match /photos/abc).
        prefix = folder.path.rstrip("/") + "/"
        photo_ids = [
            pid for (pid,) in session.query(Photo.id)
            .filter(Photo.file_path.like(prefix + "%"))
            .all()
        ]

        qdrant_point_ids = []
        if photo_ids:
            qdrant_point_ids = [
                pid for (pid,) in session.query(Embedding.qdrant_point_id)
                .filter(Embedding.photo_id.in_(photo_ids))
                .filter(Embedding.qdrant_point_id.isnot(None))
                .all()
            ]

        # Remove from Qdrant first — if this fails we don't want the DB rows
        # gone already (otherwise orphaned vectors would linger).
        if qdrant_point_ids:
            try:
                job_queue_manager.qdrant_client.delete(
                    collection_name="embeddings",
                    points_selector=qdrant_point_ids,
                )
            except Exception as e:
                logger.warning("Failed to delete %d Qdrant points: %s", len(qdrant_point_ids), e)

        # Cascade deletes in child-first order to satisfy FKs.
        if photo_ids:
            session.query(Embedding).filter(Embedding.photo_id.in_(photo_ids)).delete(synchronize_session=False)
            session.query(ProcessingState).filter(ProcessingState.photo_id.in_(photo_ids)).delete(synchronize_session=False)
            session.query(Photo).filter(Photo.id.in_(photo_ids)).delete(synchronize_session=False)

        session.delete(folder)
        session.commit()

        # Incrementally drop the removed photos from the similarity index
        # (O(edges), no full Qdrant re-scroll) — see _remove_photos_from_cache.
        if photo_ids:
            await remove_photos_from_index(set(photo_ids))

        return {
            "deleted": folder_id,
            "photos_removed": len(photo_ids),
            "embeddings_removed": len(qdrant_point_ids),
        }
    finally:
        session.close()


@app.post("/folders/{folder_id}/scan")
async def scan_folder_by_id(folder_id: int):
    """Trigger a scan of a registered folder, reusing the /rescan logic."""
    if job_queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Service not initialized"})
    from app.models import FolderPath
    session = job_queue_manager.SessionLocal()
    try:
        folder = session.query(FolderPath).filter(FolderPath.id == folder_id).first()
        folder_path = folder.path if folder else None
    finally:
        session.close()
    if not folder_path:
        return JSONResponse(status_code=404, content={"error": "Folder not found"})
    return await rescan_folder(folder_path=folder_path)


@app.post("/rescan")
async def rescan_folder(folder_path: str = None):
    """Trigger manual folder re-scan with change detection and incremental processing.
    
    Scans folder for new/modified/deleted photos, queues changes for processing.
    Returns job_id for tracking progress via WebSocket.
    """
    if job_queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Job queue not initialized"})
    
    if not folder_path:
        folder_path = os.getenv("PHOTOS_FOLDER", "./photos")
    
    if not os.path.exists(folder_path):
        return JSONResponse(status_code=400, content={
            "error": f"Path does not exist: {folder_path}",
            "detail": "Please provide a valid folder path that exists on the server.",
        })
    
    if not os.path.isdir(folder_path):
        return JSONResponse(status_code=400, content={
            "error": f"Path is not a directory: {folder_path}",
            "detail": "The provided path exists but is not a folder. Please provide a directory path.",
        })
    
    try:
        # Initialize scanner and database session. Pass the Qdrant client so
        # files deleted on disk have their vectors purged (not orphaned).
        scanner = FolderScanner(qdrant_client=job_queue_manager.qdrant_client)
        session = job_queue_manager.SessionLocal()
        
        # Scan folder for changes (new, modified, deleted photos)
        photo_ids, change_count = scanner.scan_folder(folder_path, session)
        session.close()
        
        if change_count == 0:
            return JSONResponse(status_code=200, content={
                "message": "No changes detected",
                "changes_found": 0
            })
        
        # Create job for processing changed photos
        job_id = str(uuid.uuid4())
        job_created = job_queue_manager.create_job(job_id, change_count)
        
        if not job_created:
            return JSONResponse(status_code=500, content={"error": "Failed to create processing job"})
        
        # Queue photos for processing
        for photo_id in photo_ids:
            asyncio.create_task(job_queue_manager.process_photo(job_id, photo_id))
        
        return JSONResponse(status_code=202, content={
            "job_id": job_id,
            "message": "Rescan initiated",
            "changes_found": change_count,
            "photos_queued": len(photo_ids)
        })
    except Exception as e:
        logger.error("Error during rescan of '%s': %s", folder_path, e, exc_info=True)
        return JSONResponse(status_code=500, content={
            "error": "Failed to rescan folder",
            "detail": str(e),
        })


@app.get("/thumbnails/{photo_id}")
async def get_thumbnail(photo_id: int, size: int = 200):
    """Return a cached thumbnail for the given photo, generating it if needed."""
    # Reject out-of-range ids/sizes up front (a 0/negative id or a 10000px
    # "thumbnail" is a client bug, not a 404/500 on our side).
    err = validate_photo_id(photo_id) or validate_thumbnail_size(size)
    if err:
        return JSONResponse(status_code=400, content={"error": err})
    if job_queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Service not initialized"})

    session = job_queue_manager.SessionLocal()
    try:
        photo = session.query(Photo).filter(Photo.id == photo_id).first()
        if not photo:
            return JSONResponse(status_code=404, content={"error": "Photo not found"})

        if not os.path.isfile(photo.file_path):
            return JSONResponse(status_code=404, content={"error": "Photo file not found on disk"})

        # The thumbnail cache is keyed by file_hash. file_hash is nullable
        # (hashing can fail), and a None key would make every hash-less photo
        # collide onto one cache file (wrong thumbnails). Fall back to a
        # per-photo key so each still gets its own thumbnail.
        cache_key = photo.file_hash or f"photo-{photo.id}"
        thumb_path = thumbnail_service.get_thumbnail(
            photo.file_path, cache_key, size=(size, size)
        )
        return FileResponse(thumb_path, media_type="image/jpeg")
    except Exception as e:
        logger.error("Error generating thumbnail for photo %d: %s", photo_id, e, exc_info=True)
        return JSONResponse(status_code=500, content={
            "error": "Failed to generate thumbnail",
            "detail": str(e),
        })
    finally:
        session.close()


_BROWSER_NATIVE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".gif", ".webp", ".bmp", ".svg", ".ico", ".avif"}


@app.get("/photos/{photo_id}/image-info")
async def get_photo_image_info(photo_id: int):
    """Return width / height / created_date for a single photo by reading
    the file's EXIF metadata. Lazy: called by the lightbox on open, so
    /similarity-groups (the slider's hot path) doesn't have to do per-
    photo file I/O for every tick.

    Cached via _read_image_info's LRU. The first call for a given file
    pays a PIL parse (~5ms); subsequent calls are O(1).
    """
    if job_queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Service not initialized"})
    session = job_queue_manager.SessionLocal()
    try:
        photo = session.query(Photo).filter(Photo.id == photo_id).first()
        if not photo:
            return JSONResponse(status_code=404, content={"error": "Photo not found"})
        if not os.path.isfile(photo.file_path):
            return {"photo_id": photo_id, "width": None, "height": None,
                    "created_date": None}
        width, height, created = _read_image_info(photo.file_path)
        return {"photo_id": photo_id, "width": width, "height": height,
                "created_date": created}
    finally:
        session.close()


@app.get("/photos/{photo_id}/full")
async def get_full_photo(photo_id: int):
    """Serve the full-resolution photo. HEIC/HEIF and other browser-
    incompatible formats are transcoded to JPEG on the fly."""
    if job_queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Service not initialized"})
    session = job_queue_manager.SessionLocal()
    try:
        photo = session.query(Photo).filter(Photo.id == photo_id).first()
        if not photo:
            return JSONResponse(status_code=404, content={"error": "Photo not found"})
        if not os.path.isfile(photo.file_path):
            return JSONResponse(status_code=404, content={"error": "File not found on disk"})

        ext = os.path.splitext(photo.file_path)[1].lower()
        if ext in _BROWSER_NATIVE_EXTS:
            import mimetypes
            mt = mimetypes.guess_type(photo.file_path)[0] or "image/jpeg"
            return FileResponse(photo.file_path, media_type=mt)

        # Non-native format (HEIC, HEIF, TIFF, etc.) — decode with Pillow
        # and stream as high-quality JPEG.
        from PIL import Image as _PILImage
        import io as _io
        img = _PILImage.open(photo.file_path)
        img = img.convert("RGB")  # drop alpha / palette if present
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        return Response(content=buf.read(), media_type="image/jpeg")
    except Exception as e:
        logger.error("Error serving full photo %d: %s", photo_id, e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Failed to serve photo", "detail": str(e)})
    finally:
        session.close()


import functools
import numpy as np

@functools.lru_cache(maxsize=4096)
def _read_image_info(file_path: str) -> tuple:
    """Read dimensions + EXIF date from an image file. Returns a tuple
    (width, height, created_date_iso_or_None). Cached per file path."""
    width = height = None
    created_date = None
    try:
        from PIL import Image as _Img
        with _Img.open(file_path) as img:
            width = img.width
            height = img.height
            exif = img.getexif() if hasattr(img, "getexif") else {}
            for tag in (36867, 306):
                val = exif.get(tag)
                if val:
                    try:
                        created_date = datetime.strptime(str(val), "%Y:%m:%d %H:%M:%S").isoformat()
                    except Exception:
                        pass
                    break
            if not created_date:
                mtime = os.path.getmtime(file_path)
                created_date = datetime.fromtimestamp(mtime).isoformat()
    except Exception:
        pass
    return (width, height, created_date)


# --------------- Event-driven similarity index cache ---------------
#
# Built once eagerly at startup, then refreshed on changes:
#   - Additions: job_queue calls notify_embeddings_changed(photo_id) after
#     upsert. A debounce coalesces rapid additions, then folds just the new
#     photos in incrementally (_incremental_add_sync) — the UI sees fresh
#     groupings within a minute of the scan going idle, without re-scrolling
#     the whole collection.
#   - Deletions: deduplicate / auto-deduplicate / folder-delete call
#     _remove_photos_from_cache() immediately — an O(edges) in-memory filter
#     of just the removed nodes, so the UI reflects removals right away without
#     a full Qdrant re-scroll.
#
# The cache is a SPARSE EDGE index, NOT a dense N×N cosine matrix. At 60k
# photos a dense matrix is 14.4 GB; we instead keep only pairs above
# _SIM_CACHE_THRESHOLD (typically 0.7) — on real photo collections ~20 edges
# per photo. Those are stored as a score-sorted, undirected `edge_arrays`
# triple of numpy arrays (see _edges_from_adjacency / _threshold_components),
# which is both compact and O(active-edges) to cluster per request. The unit-
# normalized vectors are kept too (60k × 384 × 4 ≈ 92 MB) so reference-vs-
# member scoring stays exact even for an edge that wasn't above the floor.

_sim_cache: Dict[str, object] = {"data": None, "meta": None}
_sim_debounce_handle: Optional[asyncio.TimerHandle] = None
_sim_recompute_lock: Optional[asyncio.Lock] = None

# Connected-components are memoized PER threshold, and the memo lives ON the
# cache_data object (see _threshold_components_cached) — so when the index is
# replaced (scan adds photos, a delete removes them) the new cache_data carries
# a fresh, empty memo and a stale entry can never be served. The cap keeps
# memory bounded regardless of collection size (each entry is the dup-graph
# partition — ints only, far smaller than the group payloads it lets us skip).
from collections import OrderedDict as _OrderedDict
_COMPONENTS_MEMO_MAX = 16


def _store_sim_cache(cache_data, photo_meta) -> None:
    """Replace the cached index (and, implicitly, its components memo)."""
    _sim_cache.update(data=cache_data, meta=photo_meta)


_SIM_DEBOUNCE_SECONDS = 8.0       # fold changes in after this much quiet
_SIM_MAX_COALESCE_SECONDS = 25.0  # ...but during a continuous scan, update at
                                  # least this often so groups appear live
_SIM_SCROLL_PAGE = 2000          # Qdrant scroll page size
_SIM_CACHE_THRESHOLD = 0.70      # adjacency floor; UI thresholds are >= this
_SIM_TOP_K = 100                 # max neighbours stored per photo
_SIM_SEARCH_BATCH = 256          # Qdrant search_batch size

# Observability fields exposed via /stats.
_sim_index_info: Dict[str, object] = {
    "last_recompute_at": None,
    "last_recompute_duration_ms": None,
    "recompute_running": False,
    "vectors_in_index": 0,
    "edges_in_index": 0,
    "cache_threshold": _SIM_CACHE_THRESHOLD,
}


def _get_recompute_lock() -> asyncio.Lock:
    """Lazy-construct the lock against the running loop. A module-level
    Lock would bind to whichever loop happened to be current at import
    time, which breaks under TestClient (one loop per request)."""
    global _sim_recompute_lock
    if _sim_recompute_lock is None:
        _sim_recompute_lock = asyncio.Lock()
    return _sim_recompute_lock


async def remove_photos_from_index(deleted_pids: set) -> None:
    """Lock-serialized wrapper around _remove_photos_from_cache.

    Deletes (dedupe / folder removal) mutate _sim_cache; an incremental add
    runs on an executor thread UNDER the recompute lock. Calling the sync
    removal directly could interleave with that thread — racing the same numpy
    arrays, or being overwritten when the in-flight add writes its (pre-delete)
    snapshot back, resurrecting just-deleted photos. Taking the lock makes the
    two mutually exclusive."""
    async with _get_recompute_lock():
        _remove_photos_from_cache(deleted_pids)


def _compute_sim_cache():
    """Synchronous: scroll Qdrant, query Postgres, build a SPARSE adjacency
    of (i, j, score) triples for all pairs above _SIM_CACHE_THRESHOLD.

    Returns (cache_data, photo_meta) or (None, None). cache_data shape:
        {
          "vectors":      np.ndarray (N, D), unit-normalized,
          "photo_ids":    [int],     index i -> photo_id
          "point_ids":    [str],     index i -> Qdrant point id
          "edge_arrays":  (scores_asc, i_idx, j_idx) numpy arrays — the
                          score-sorted undirected edge index used for O(active)
                          threshold clustering (see _threshold_components),
          "max_effective_size": float,  quality-score denominator,
          "cache_threshold": float,  the floor at which edges were built
        }

    Memory at 60k photos with ~20 neighbours each: ~92 MB vectors + ~10 MB
    edges, vs 14.4 GB for the previous dense matrix. Time at 60k photos
    via Qdrant search_batch (HNSW, sub-linear per query) is seconds, not
    minutes.
    """
    qc = job_queue_manager.qdrant_client if job_queue_manager else None
    if qc is None:
        return None, None
    collection = "embeddings"

    # 1) Scroll all points (paginated — no silent truncation).
    points: list = []
    next_offset = None
    while True:
        page, next_offset = qc.scroll(
            collection_name=collection,
            limit=_SIM_SCROLL_PAGE,
            offset=next_offset,
            with_payload=True,
            with_vectors=True,
        )
        if not page:
            break
        points.extend(page)
        if next_offset is None:
            break
    if not points:
        return None, None

    # 2) Load photo metadata from Postgres.
    session = job_queue_manager.SessionLocal()
    photo_meta: Dict[int, dict] = {}
    try:
        rows = session.query(
            Photo.id, Photo.filename, Photo.file_path,
            Photo.file_size, Photo.mime_type, Photo.uploaded_at
        ).all()
        for r in rows:
            photo_meta[r[0]] = {
                "filename": r[1], "file_path": r[2], "file_size": r[3],
                "mime_type": r[4],
                "uploaded_at": r[5].isoformat() if r[5] else None,
            }
    finally:
        session.close()

    # 3) Drop Qdrant points whose Postgres row is gone (orphaned vectors).
    valid_ids = set(photo_meta.keys())
    filtered = [(p, int(p.payload.get("photo_id", 0))) for p in points
                if int(p.payload.get("photo_id", 0)) in valid_ids]
    if not filtered:
        return None, None

    point_ids = [p.id for p, _ in filtered]
    photo_ids = [pid for _, pid in filtered]
    raw_vecs = np.array([p.vector for p, _ in filtered], dtype=np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = raw_vecs / norms
    n = len(photo_ids)

    # 4) Build sparse adjacency. Use Qdrant's batched HNSW search to find
    # each vector's near-neighbours above the cache threshold.
    pid_to_idx = {pid: i for i, pid in enumerate(photo_ids)}
    adjacency: list = [[] for _ in range(n)]

    # Qdrant's SearchRequest model lives under qdrant_client.http.models.
    # Import lazily so unit tests with a fake client don't pay the import.
    from qdrant_client.http.models import SearchRequest

    for batch_start in range(0, n, _SIM_SEARCH_BATCH):
        batch_end = min(batch_start + _SIM_SEARCH_BATCH, n)
        requests = [
            SearchRequest(
                vector=vecs[i].tolist(),
                limit=_SIM_TOP_K + 1,  # +1 to absorb the self-hit
                score_threshold=_SIM_CACHE_THRESHOLD,
                with_payload=True,
            )
            for i in range(batch_start, batch_end)
        ]
        results = qc.search_batch(collection_name=collection, requests=requests)
        for offset, hits in enumerate(results):
            i = batch_start + offset
            for hit in hits:
                hit_pid = int(hit.payload.get("photo_id", 0)) if hit.payload else 0
                j = pid_to_idx.get(hit_pid)
                if j is None or j == i:
                    continue
                adjacency[i].append((j, float(hit.score)))

    cache_data = {
        "vectors": vecs,
        "photo_ids": photo_ids,
        "point_ids": point_ids,
        # Score-sorted undirected edge index — the ONLY edge structure kept in
        # steady state. We build the directed `adjacency` locally above only to
        # derive this, then let it go: at 300k photos the list-of-lists-of-
        # tuples is hundreds of MB, while these three numpy arrays are tens of
        # MB and are all the hot paths (clustering, auto-dedupe) need.
        "edge_arrays": _edges_from_adjacency(adjacency),
        # Quality-score denominator, precomputed so the slider hot path never
        # scans all N photo metas.
        "max_effective_size": max(
            (_effective_size(m) for m in photo_meta.values()), default=0.0
        ),
        "cache_threshold": _SIM_CACHE_THRESHOLD,
    }
    return cache_data, photo_meta


async def _recompute_sim_cache():
    """Recompute the sparse index (heavy work in a thread) and update the
    cache. Lock-guarded so concurrent triggers (debounce + delete) don't
    stomp; updates _sim_index_info for /stats observability."""
    async with _get_recompute_lock():
        loop = asyncio.get_running_loop()
        _sim_index_info["recompute_running"] = True
        t0 = time.time()
        try:
            cache_data, photo_meta = await loop.run_in_executor(None, _compute_sim_cache)
            _store_sim_cache(cache_data, photo_meta)
            n_vecs = len(cache_data["photo_ids"]) if cache_data else 0
            n_edges = int(cache_data["edge_arrays"][0].size) if cache_data else 0
            _sim_index_info.update(
                last_recompute_at=datetime.utcnow().isoformat(),
                last_recompute_duration_ms=int((time.time() - t0) * 1000),
                vectors_in_index=n_vecs,
                edges_in_index=n_edges,
            )
            logger.info(
                "Similarity index recomputed: %d vectors, %d edges, %dms",
                n_vecs, n_edges, _sim_index_info["last_recompute_duration_ms"],
            )
        finally:
            _sim_index_info["recompute_running"] = False


# Photo IDs embedded since the last index update, accumulated by
# notify_embeddings_changed and drained by the debounced _apply_pending_changes.
# Lets a scan's worth of additions be folded into the index incrementally
# (search only the new vectors) instead of re-scrolling all N vectors.
_pending_new_pids: set = set()


_pending_first_at: Optional[float] = None  # loop time when the current batch started


def notify_embeddings_changed(photo_id: Optional[int] = None):
    """Call after an embedding is added/updated. Debounces the index update:
    it fires after _SIM_DEBOUNCE_SECONDS of quiet, BUT no later than
    _SIM_MAX_COALESCE_SECONDS after the batch started. The max-coalesce cap is
    what makes duplicate groups appear progressively during a long, continuous
    scan (otherwise the quiet-timer keeps resetting and nothing updates until
    the scan finishes).

    Pass the photo_id so the update is INCREMENTAL — only the new photos are
    searched and merged in (O(new) Qdrant searches), keeping it cheap at 300k.
    Called with no id, it falls back to a full recompute."""
    global _sim_debounce_handle, _pending_first_at
    if photo_id is not None:
        _pending_new_pids.add(photo_id)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop (e.g. called from a sync test) — skip debounce
        return
    now = loop.time()
    if _pending_first_at is None:
        _pending_first_at = now
    # Normally wait for quiet; but if we've been accumulating for a while
    # during a continuous scan, fire promptly so results show up live.
    delay = 0.0 if (now - _pending_first_at) >= _SIM_MAX_COALESCE_SECONDS else _SIM_DEBOUNCE_SECONDS
    if _sim_debounce_handle is not None:
        _sim_debounce_handle.cancel()
    _sim_debounce_handle = loop.call_later(
        delay,
        lambda: asyncio.ensure_future(_apply_pending_changes()),
    )


async def _apply_pending_changes():
    """Debounce callback: incrementally add the photos accumulated since the
    last update, or fall back to a full recompute when there's no base index
    yet (cold start) or no specific additions were recorded."""
    global _pending_new_pids, _pending_first_at
    pending = _pending_new_pids
    _pending_new_pids = set()
    _pending_first_at = None  # restart the max-coalesce clock for the next batch

    if _sim_cache.get("data") is None or not pending:
        await _recompute_sim_cache()
        return

    need_full_fallback = False
    async with _get_recompute_lock():
        loop = asyncio.get_running_loop()
        _sim_index_info["recompute_running"] = True
        t0 = time.time()
        try:
            cache_data, photo_meta = await loop.run_in_executor(
                None, _incremental_add_sync, pending
            )
            if cache_data is None:
                need_full_fallback = True
            else:
                _store_sim_cache(cache_data, photo_meta)
                _sim_index_info.update(
                    last_recompute_at=datetime.utcnow().isoformat(),
                    last_recompute_duration_ms=int((time.time() - t0) * 1000),
                    vectors_in_index=len(cache_data["photo_ids"]),
                    edges_in_index=int(cache_data["edge_arrays"][0].size),
                )
        except Exception as e:
            logger.warning("Incremental index add failed (%s); full rebuild.", e)
            need_full_fallback = True
        finally:
            _sim_index_info["recompute_running"] = False

    # Fallback OUTSIDE the lock — _recompute_sim_cache reacquires it and the
    # asyncio lock is not reentrant.
    if need_full_fallback:
        await _recompute_sim_cache()


def _get_cached_data():
    """Return (cache_data, photo_meta) from the precomputed cache.
    If cache is empty (first call before any event), compute synchronously.
    Startup eagerly precomputes, so this fallback is only hit when the
    server skipped startup (e.g. some tests) or before any embeddings exist."""
    if _sim_cache["data"] is not None:
        return _sim_cache["data"], _sim_cache["meta"]
    cache_data, photo_meta = _compute_sim_cache()
    _store_sim_cache(cache_data, photo_meta)
    return cache_data, photo_meta


_PURE_DUPE_EPSILON = 1e-4  # float32 normalize-then-dot noise floor


def _effective_size(meta: dict) -> float:
    """File size with the same +20% bonus for universally-decodable formats
    (JPEG/PNG) that _keeper_key uses. Keeps the quality score monotonic with
    the keeper ranking, so the photo we mark "Best" is also the highest
    quality in its group."""
    size = float(meta.get("file_size") or 0)
    if meta.get("mime_type") in _PREFERRED_MIME_TYPES:
        return size * 1.2
    return size


def _quality_score(meta: dict, max_effective_size: float) -> float:
    """A cross-group-comparable photo quality in [0, 1].

    Derived from file size (a proxy for detail / low compression) normalized
    by the largest effective size in the whole collection, so the single
    highest-quality photo scores ~1.0 and everything else scales beneath it.
    This is data-driven (no magic byte threshold) and uses only fields
    already in the in-memory cache, so it stays off the slider's hot path.

    Replaces the old hard-coded 0.8 that made `min_quality` and
    `sort_by=quality` no-ops and left the frontend quality badges blank.
    """
    if max_effective_size <= 0:
        return 0.0
    return round(min(1.0, _effective_size(meta) / max_effective_size), 4)


# ---- Scalable clustering primitives (200k–300k photo collections) ----
#
# The cache's per-photo `adjacency` (list of lists of (j, score)) is great for
# the cold build but a poor shape for the per-slider-tick hot path: clustering
# from it meant allocating N empty sets and scanning every edge on EVERY tick,
# i.e. O(N + E) regardless of threshold. At 300k photos / millions of edges
# that makes the slider lag for seconds.
#
# Instead we collapse the adjacency once into a flat, score-SORTED, undirected,
# de-duplicated edge index stored as three parallel numpy arrays. Then a tick
# at threshold t is:
#   - np.searchsorted to find the active suffix (edges with score >= t)   O(logE)
#   - build adjacency only for the nodes touched by those edges + BFS    O(active)
# At the high thresholds dedup actually uses (0.9–1.0) the active set is tiny,
# so a tick is near-instant even on a 300k collection.


def _edges_from_adjacency(adjacency: list):
    """Collapse the directed per-photo adjacency into score-sorted, undirected,
    de-duplicated edge arrays. Symmetrizes (Qdrant's top-k can drop a back-edge)
    by keeping the max score seen for each unordered pair. Returns
    (scores_ascending, i_idx, j_idx) as numpy arrays. Computed once per cache."""
    pair_max: Dict[tuple, float] = {}
    for i, edges in enumerate(adjacency):
        for j, s in edges:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            prev = pair_max.get((a, b))
            if prev is None or s > prev:
                pair_max[(a, b)] = s
    if not pair_max:
        empty_f = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int32)
        return empty_f, empty_i.copy(), empty_i.copy()
    items = sorted(pair_max.items(), key=lambda kv: kv[1])  # ascending by score
    m = len(items)
    # float64 (not 32): scores are compared against the request threshold via
    # searchsorted, and float32 rounding of e.g. 0.9999 drifts just below the
    # threshold and would wrongly drop a pure-duplicate edge. The source
    # adjacency already holds float64 cosines, so this is lossless.
    scores = np.fromiter((s for _, s in items), dtype=np.float64, count=m)
    i_idx = np.fromiter((p[0] for p, _ in items), dtype=np.int32, count=m)
    j_idx = np.fromiter((p[1] for p, _ in items), dtype=np.int32, count=m)
    return scores, i_idx, j_idx


def _get_edge_arrays(cache_data: dict):
    """Return the cached (scores, i, j) edge arrays, deriving + memoizing them
    from `adjacency` on first use. Production builds them eagerly in
    _compute_sim_cache (cold path); the unit-test cache only sets `adjacency`,
    so this lazily fills them in."""
    ea = cache_data.get("edge_arrays")
    if ea is not None:
        return ea
    ea = _edges_from_adjacency(cache_data.get("adjacency") or [])
    cache_data["edge_arrays"] = ea
    return ea


def _threshold_components(cache_data: dict, threshold: float) -> list:
    """Connected components (size >= 2) of the duplicate graph at `threshold`.

    O(active edges) — independent of collection size — via a searchsorted slice
    of the precomputed score-sorted edge index. Components and their members
    are returned in a deterministic (index-sorted) order so callers that build
    user-facing plans stay reproducible."""
    scores, i_idx, j_idx = _get_edge_arrays(cache_data)
    if scores.size == 0:
        return []
    lo = int(np.searchsorted(scores, threshold, side="left"))  # first >= threshold
    if lo >= scores.size:
        return []
    # tolist() is markedly faster than per-element numpy indexing in the loop.
    us = i_idx[lo:].tolist()
    vs = j_idx[lo:].tolist()

    adj: Dict[int, list] = {}
    for u, v in zip(us, vs):
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    visited: set = set()
    components: list = []
    for start in adj:
        if start in visited:
            continue
        visited.add(start)
        comp = [start]
        stack = [start]
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    comp.append(nb)
                    stack.append(nb)
        if len(comp) >= 2:
            comp.sort()
            components.append(comp)
    components.sort(key=lambda c: c[0])
    return components


def _threshold_components_cached(cache_data: dict, threshold: float) -> list:
    """_threshold_components memoized by (index version, rounded threshold).

    The slider's hot path re-requests the same threshold constantly — every
    page change, and every time the user drags back over a value. Clustering
    ~100k edges each time is what made the slider lag; this turns a repeat into
    an O(1) dict lookup. The memo lives on cache_data, so when the index is
    replaced the entries vanish with it (no stale results), and it's LRU-capped
    so memory stays bounded no matter the collection size."""
    memo = cache_data.get("__components_memo")
    if memo is None:
        memo = _OrderedDict()
        cache_data["__components_memo"] = memo
    key = round(float(threshold), 4)
    hit = memo.get(key)
    if hit is not None:
        memo.move_to_end(key)
        return hit
    comps = _threshold_components(cache_data, threshold)
    memo[key] = comps
    while len(memo) > _COMPONENTS_MEMO_MAX:
        memo.popitem(last=False)
    return comps


def _merge_new_edges(existing_arrays, new_directed_edges):
    """Merge freshly-discovered directed edges into the score-sorted edge
    index. Used by incremental add.

    `new_directed_edges` is a list of (i, j, score) where at least one endpoint
    is a brand-new node index (>= the old node count). Because every new edge
    touches a new index and existing edges only connect old indices, the two
    sets are DISJOINT — so we just de-duplicate the new directed edges into
    undirected pairs (keeping the max score for each), concatenate, and re-sort
    ascending. Returns new (scores, i, j) arrays."""
    scores, ii, jj = existing_arrays
    pair_max: Dict[tuple, float] = {}
    for a, b, s in new_directed_edges:
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        prev = pair_max.get(key)
        if prev is None or s > prev:
            pair_max[key] = s
    if not pair_max:
        return existing_arrays
    m = len(pair_max)
    ns = np.fromiter(pair_max.values(), dtype=np.float64, count=m)
    ni = np.fromiter((k[0] for k in pair_max), dtype=np.int32, count=m)
    nj = np.fromiter((k[1] for k in pair_max), dtype=np.int32, count=m)
    all_s = np.concatenate([scores, ns])
    all_i = np.concatenate([ii, ni])
    all_j = np.concatenate([jj, nj])
    order = np.argsort(all_s, kind="stable")  # ascending, as production expects
    return all_s[order], all_i[order].astype(np.int32), all_j[order].astype(np.int32)


def _incremental_add_sync(new_pids: set):
    """Add newly-embedded photos to the in-memory index WITHOUT a full rebuild.

    Searches ONLY the new vectors against Qdrant (O(new · top_k)) and merges
    their edges in, instead of scrolling all N vectors and running N HNSW
    searches. This is what makes a scan of a few hundred new photos cheap even
    when the existing collection holds 300k.

    Returns the updated (cache_data, photo_meta), or (None, None) to tell the
    caller to fall back to a full recompute (cold cache, missing client, or any
    unexpected shape — correctness over cleverness)."""
    from app.models import Embedding as _Emb, Photo as _Photo

    base = _sim_cache.get("data")
    meta = _sim_cache.get("meta")
    qc = job_queue_manager.qdrant_client if job_queue_manager else None
    if not base or meta is None or qc is None:
        return None, None

    existing = set(base["photo_ids"])
    add_pids = [p for p in new_pids if p not in existing]
    if not add_pids:
        return base, meta  # all already indexed (e.g. duplicate notify)

    # 1) point_ids + metadata for the new photos, from Postgres.
    session = job_queue_manager.SessionLocal()
    try:
        pid_to_point = {
            pid: pt for pid, pt in (
                session.query(_Emb.photo_id, _Emb.qdrant_point_id)
                .filter(_Emb.photo_id.in_(add_pids))
                .filter(_Emb.qdrant_point_id.isnot(None))
                .all()
            )
        }
        meta_rows = (
            session.query(_Photo.id, _Photo.filename, _Photo.file_path,
                          _Photo.file_size, _Photo.mime_type, _Photo.uploaded_at)
            .filter(_Photo.id.in_(add_pids)).all()
        )
    finally:
        session.close()
    new_meta = {
        r[0]: {"filename": r[1], "file_path": r[2], "file_size": r[3],
               "mime_type": r[4], "uploaded_at": r[5].isoformat() if r[5] else None}
        for r in meta_rows
    }
    add_pids = [p for p in add_pids if p in pid_to_point and p in new_meta]
    if not add_pids:
        return base, meta

    # 2) Retrieve the new vectors from Qdrant.
    point_ids = [pid_to_point[p] for p in add_pids]
    records = qc.retrieve(collection_name="embeddings", ids=point_ids, with_vectors=True)
    vec_by_point = {rec.id: rec.vector for rec in records if rec.vector is not None}
    add_pids = [p for p in add_pids if pid_to_point[p] in vec_by_point]
    if not add_pids:
        return base, meta
    point_ids = [pid_to_point[p] for p in add_pids]
    raw = np.array([vec_by_point[pt] for pt in point_ids], dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    new_vecs = raw / norms

    # 3) Combined node arrays (old photos first, then the new ones).
    old_ids = base["photo_ids"]
    old_n = len(old_ids)
    combined_ids = list(old_ids) + add_pids
    pid_to_idx = {pid: i for i, pid in enumerate(combined_ids)}
    base_vecs = base["vectors"]
    combined_vectors = (
        np.vstack([base_vecs, new_vecs]) if getattr(base_vecs, "size", 0) else new_vecs
    )

    # 4) Search only the new vectors for neighbours above the cache floor.
    from qdrant_client.http.models import SearchRequest
    floor = base.get("cache_threshold", _SIM_CACHE_THRESHOLD)
    requests = [
        SearchRequest(vector=new_vecs[k].tolist(), limit=_SIM_TOP_K + 1,
                      score_threshold=floor, with_payload=True)
        for k in range(len(add_pids))
    ]
    new_directed = []
    for k, hits in enumerate(qc.search_batch(collection_name="embeddings", requests=requests)):
        i = old_n + k
        for hit in hits:
            hpid = int(hit.payload.get("photo_id", 0)) if hit.payload else 0
            j = pid_to_idx.get(hpid)
            if j is None or j == i:
                continue
            new_directed.append((i, j, float(hit.score)))

    new_edges = _merge_new_edges(_get_edge_arrays(base), new_directed)

    merged_meta = dict(meta)
    merged_meta.update(new_meta)
    new_max = max(
        base.get("max_effective_size", 0.0),
        max((_effective_size(m) for m in new_meta.values()), default=0.0),
    )
    new_data = {
        "vectors": combined_vectors,
        "photo_ids": combined_ids,
        "point_ids": list(base.get("point_ids") or []) + point_ids,
        "edge_arrays": new_edges,
        "max_effective_size": new_max,
        "cache_threshold": floor,
    }
    return new_data, merged_meta


def _remove_photos_from_cache(deleted_pids: set) -> None:
    """Drop photo_ids from the in-memory similarity cache WITHOUT a full
    rebuild. Deletion only removes nodes/edges, so the surviving index is a
    pure subset — we filter it in O(current edges) with numpy instead of
    scrolling every vector out of Qdrant and re-running N HNSW searches.

    That full recompute is what /deduplicate, /auto-deduplicate and
    folder-delete used to await synchronously; on a 300k-photo collection it
    blocks the request for minutes just to remove a handful of duplicates. The
    next scan-triggered recompute still rebuilds from Qdrant, so this is a fast,
    correct interim update. No-op if the cache isn't built yet."""
    data = _sim_cache.get("data")
    if not data or not deleted_pids:
        return
    photo_ids = data["photo_ids"]
    n = len(photo_ids)
    # old index -> new index (-1 = removed), as a numpy lookup for vectorized
    # edge remapping.
    old_to_new = np.full(n, -1, dtype=np.int64)
    new_photo_ids = []
    new_point_ids = []
    point_ids = data.get("point_ids") or []
    for old in range(n):
        if photo_ids[old] in deleted_pids:
            continue
        old_to_new[old] = len(new_photo_ids)
        new_photo_ids.append(photo_ids[old])
        if old < len(point_ids):
            new_point_ids.append(point_ids[old])
    if len(new_photo_ids) == n:
        return  # nothing actually removed

    vectors = data["vectors"]
    new_vectors = vectors[old_to_new >= 0] if getattr(vectors, "size", 0) else vectors

    scores, ii, jj = _get_edge_arrays(data)
    if scores.size:
        ni = old_to_new[ii]
        nj = old_to_new[jj]
        mask = (ni >= 0) & (nj >= 0)            # keep edges with both ends alive
        new_edges = (scores[mask], ni[mask].astype(np.int32), nj[mask].astype(np.int32))
    else:
        new_edges = (scores, ii, jj)

    meta = _sim_cache.get("meta") or {}
    for pid in deleted_pids:
        meta.pop(pid, None)

    _sim_cache.update(
        data={
            "vectors": new_vectors,
            "photo_ids": new_photo_ids,
            "point_ids": new_point_ids,
            "adjacency": [],  # superseded by edge_arrays
            "edge_arrays": new_edges,
            "max_effective_size": max(
                (_effective_size(m) for m in meta.values()), default=0.0
            ),
            "cache_threshold": data.get("cache_threshold", _SIM_CACHE_THRESHOLD),
        },
        meta=meta,
    )
    _sim_index_info.update(
        vectors_in_index=len(new_photo_ids),
        edges_in_index=int(new_edges[0].size),
    )


def _build_similarity_groups_from_qdrant(threshold: float):
    """Cluster photos into similarity groups from the sparse edge index.
    Reads the precomputed, score-sorted edge arrays in O(active edges) — no
    Qdrant calls per request — and clusters by connected components.

    Threshold handling:
      - threshold > 1.0 returns no groups (no cosine exceeds 1.0 — defends
        against off-by-one slider bugs).
      - At threshold ≥ 1.0 the effective filter drops by _PURE_DUPE_EPSILON.
        Reason: two byte-identical photos give the same DINOv2 vector, but
        float32 normalize-then-dot returns ~0.9999998, not exactly 1.0. A
        strict s >= 1.0 filter would silently exclude the very pairs the
        user is asking for when they slide to "pure duplicates".
      - threshold < cache_floor is clamped up to the cache floor — the
        adjacency wasn't built with those edges.
    """
    if threshold > 1.0:
        return []

    cache_data, photo_meta = _get_cached_data()
    if cache_data is None:
        return []

    vectors = cache_data["vectors"]
    photo_ids = cache_data["photo_ids"]
    cache_floor = cache_data.get("cache_threshold", _SIM_CACHE_THRESHOLD)
    if threshold >= 1.0:
        threshold = 1.0 - _PURE_DUPE_EPSILON
    effective_threshold = max(threshold, cache_floor)

    # Largest effective file size across the whole collection — the
    # denominator that makes per-photo quality scores comparable between
    # groups. Memoized on the cache: scanning all photo_meta is O(N), which
    # must NOT happen on every slider tick at 300k photos. _compute_sim_cache
    # fills it eagerly; this is the lazy fallback for hand-built test caches.
    max_effective_size = cache_data.get("max_effective_size")
    if max_effective_size is None:
        max_effective_size = max(
            (_effective_size(m) for m in (photo_meta or {}).values()),
            default=0.0,
        )
        cache_data["max_effective_size"] = max_effective_size

    # Cluster by connected components over the duplicate graph. This is
    # transitive-correct (A~B, B~C ⇒ {A,B,C}, even if A≁C) and identical to
    # what _plan_auto_dedupe uses, so the groups shown here match what
    # auto-dedupe acts on. _threshold_components is O(active edges) via a
    # sorted edge index, so this stays fast on 300k-photo collections even
    # though it runs on every threshold-slider tick.
    groups = []
    for component in _threshold_components_cached(cache_data, effective_threshold):
        g = _build_group_for_component(
            component, vectors, photo_ids, photo_meta, max_effective_size
        )
        if g is not None:
            groups.append(g)
    return groups


def _build_group_for_component(component, vectors, photo_ids, photo_meta, max_effective_size):
    """Build the full user-facing group dict for ONE component: members,
    keeper/reference, exact pairwise cosine, and the 'why this copy' reasons.
    Returns None for a singleton.

    Extracted so the list endpoint can build this (the expensive part —
    per-member dot products and string formatting) for ONLY the page being
    shown, instead of every one of the hundreds of groups a low threshold
    produces."""
    # Width / height / created_date are loaded on-demand by the lightbox via
    # /photos/{id}/image-info — kept out of this hot path.
    members = []
    for j in component:
        pid = photo_ids[j]
        meta = photo_meta.get(pid) or {}
        members.append({
            "_idx": j,
            "photo_id": pid,
            "filename": meta.get("filename") or str(pid),
            "path": f"{BACKEND_PUBLIC_URL}/thumbnails/{pid}",
            "similarity_score": 0.0,  # placeholder, recomputed below
            "quality_score": _quality_score(meta, max_effective_size),
            "file_size": meta.get("file_size"),
            "file_path": meta.get("file_path"),
            "mime_type": meta.get("mime_type"),
            "uploaded_at": meta.get("uploaded_at"),
            "width": None,
            "height": None,
            "created_date": None,
        })

    if len(members) < 2:
        return None

    # Keeper ("★ Best") = quality-first, earliest-taken tiebreak — the
    # same ranking auto-dedupe uses (see _keeper_key). Ascending sort,
    # best first.
    members.sort(key=_keeper_key)
    ref = members[0]
    others = members[1:]
    ref_pid = ref["photo_id"]

    # Score each member by exact cosine against the chosen reference.
    # Vectors are unit-normalized, so dot product == cosine. One matmul over
    # all members (vectors[idxs] @ ref_vec) instead of a Python dot per member
    # — at low thresholds a single cluster can hold hundreds of photos, and the
    # per-member loop was the dominant cost there.
    ref_idx = ref.pop("_idx")
    ref["similarity_score"] = 1.0
    other_idxs = [m.pop("_idx") for m in others]
    sims = vectors[other_idxs] @ vectors[ref_idx]
    for m, sc in zip(others, sims):
        m["similarity_score"] = float(sc)
    avg_sim = float(sims.mean()) if len(sims) else 0.0

    def _fmt_size(b):
        if b >= 1_000_000:
            return f"{b / 1_000_000:.2f} MB"
        return f"{b / 1_000:.1f} KB"

    ref_size = ref.get("file_size") or 0
    reasons = []
    if ref_size > 0 and others:
        other_sizes = [(m.get("file_size") or 0) for m in others]
        biggest_other = max(other_sizes)
        if ref_size == biggest_other:
            reasons.append(f"Identical file size: {_fmt_size(ref_size)}")
            ref_name = ref.get("filename", "")
            other_names = [m.get("filename", "") for m in others]
            has_copy_suffix = any("(" in n or "copy" in n.lower() for n in other_names)
            if has_copy_suffix and "(" not in ref_name and "copy" not in ref_name.lower():
                reasons.append(f"Filename \"{ref_name}\" appears to be the original (others have copy suffixes)")
        elif ref_size > biggest_other:
            pct = ((ref_size - biggest_other) / biggest_other * 100) if biggest_other > 0 else 0
            reasons.append(f"Largest file: {_fmt_size(ref_size)} vs next {_fmt_size(biggest_other)} (+{pct:.0f}%)")
        else:
            reasons.append(f"File size: {_fmt_size(ref_size)} (a larger file exists at {_fmt_size(biggest_other)} but its format is less universal)")
    if ref.get("mime_type"):
        ref_fmt = ref["mime_type"]
        # Coerce None/missing to "?" — Photo.mime_type is nullable in
        # Postgres, and a mixed list of None and str crashes sorted().
        other_fmts = sorted({(m.get("mime_type") or "?") for m in others})
        is_preferred = ref_fmt in _PREFERRED_MIME_TYPES
        fmt_note = "preferred (universal)" if is_preferred else "less universal"
        reasons.append(f"Format: {ref_fmt} ({fmt_note}) — others: {', '.join(other_fmts)}")
    if ref.get("uploaded_at"):
        reasons.append(f"Scanned: {ref['uploaded_at'][:10]}")
    if not reasons:
        reasons.append("First in similarity ranking")

    return {
        "group_id": f"grp-{ref_pid}",
        "similarity_score": avg_sim,
        # Group quality = the kept ("Best") photo's quality, i.e. the
        # highest in the group. min_quality filters on this.
        "quality_score": ref.get("quality_score", 0.0),
        "reference_photo": ref,
        "similar_photos": others,
        "best_reasons": reasons,
    }


def _list_groups_sync(threshold, skip, limit, min_quality, sort_by):
    """Synchronous worker for GET /similarity-groups, run in a threadpool so a
    big cluster build never blocks the event loop (and stalls /stats, the WS,
    or other slider requests).

    Hot-path optimization: when there's no global sort or quality filter (the
    common slider case), it clusters once (memoized) for the total, then builds
    the expensive per-group metadata for ONLY the requested page — not all
    hundreds of groups. With a sort/quality filter every group's score is
    needed, so it falls back to building them all."""
    if threshold is not None and threshold > 1.0:
        return {"total": 0, "skip": skip, "limit": limit, "groups": []}

    cache_data, photo_meta = _get_cached_data()
    if cache_data is None:
        return {"total": 0, "skip": skip, "limit": limit, "groups": []}

    vectors = cache_data["vectors"]
    photo_ids = cache_data["photo_ids"]
    cache_floor = cache_data.get("cache_threshold", _SIM_CACHE_THRESHOLD)
    thr = 0.85 if threshold is None else threshold
    if thr >= 1.0:
        thr = 1.0 - _PURE_DUPE_EPSILON
    effective_threshold = max(thr, cache_floor)

    max_effective_size = cache_data.get("max_effective_size")
    if max_effective_size is None:
        max_effective_size = max(
            (_effective_size(m) for m in (photo_meta or {}).values()), default=0.0
        )
        cache_data["max_effective_size"] = max_effective_size

    components = _threshold_components_cached(cache_data, effective_threshold)

    def _build(comp):
        return _build_group_for_component(
            comp, vectors, photo_ids, photo_meta, max_effective_size
        )

    needs_all = (sort_by is not None) or (min_quality is not None)
    if not needs_all:
        # Components are already size>=2, so each yields exactly one group →
        # total is just the component count, and we materialize only the page.
        total = len(components)
        page = [g for c in components[skip:skip + limit] if (g := _build(c)) is not None]
        return {"total": total, "skip": skip, "limit": limit, "groups": page}

    groups = [g for c in components if (g := _build(c)) is not None]
    if min_quality is not None:
        groups = [g for g in groups if g.get("quality_score", 0) >= min_quality]
    if sort_by == "similarity":
        groups.sort(key=lambda g: g.get("similarity_score", 0), reverse=True)
    elif sort_by == "quality":
        groups.sort(key=lambda g: g.get("quality_score", 0), reverse=True)
    return {
        "total": len(groups),
        "skip": skip,
        "limit": limit,
        "groups": groups[skip:skip + limit],
    }


@app.get("/similarity-groups")
async def list_similarity_groups(
    skip: int = 0,
    limit: int = 100,
    min_similarity: Optional[float] = None,
    min_quality: Optional[float] = None,
    sort_by: Optional[str] = None,
):
    """List similarity groups with pagination, filtering, and sorting."""
    # Reject malformed pagination / filter values instead of silently
    # mis-behaving (a negative skip slices from the end of the list; an
    # out-of-range min_similarity quietly returns nothing).
    err = validate_pagination(skip, limit)
    if err:
        raise HTTPException(status_code=400, detail=err)
    if min_similarity is not None or min_quality is not None:
        err = validate_similarity_filters(
            min_similarity if min_similarity is not None else 0.0,
            min_quality if min_quality is not None else 0.0,
        )
        if err:
            raise HTTPException(status_code=400, detail=err)
    if sort_by is not None and sort_by not in ("similarity", "quality"):
        raise HTTPException(status_code=400, detail=f"Invalid sort_by: {sort_by}")
    # min_similarity is the clustering threshold: a pair appears together
    # iff cos(a,b) >= min_similarity. We do NOT additionally filter on the
    # group's avg-to-reference similarity afterwards — that would silently
    # drop legitimate clusters whose ref differs from the seed by ε.
    #
    # Build off the event loop: clustering + metadata is CPU-bound and used to
    # run inline, so one slow slider request blocked every other request (and
    # the WS), and a flurry of slider moves serialized into a multi-second
    # backlog. run_in_executor keeps the loop free to serve the latest request.
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _list_groups_sync, min_similarity, skip, limit, min_quality, sort_by
    )


@app.get("/similarity-groups/{group_id}")
async def get_similarity_group_detail(group_id: str):
    """Get a similarity group by ID with thumbnail paths for each member.

    The in-memory similarity_group_service is not populated by the running
    app — the list endpoint builds groups on the fly from the Qdrant cache.
    So before giving up with a 404 we rebuild the live groups and look for a
    match. Without this fallback this endpoint could only ever 404 in
    production (the store is empty), even for groups the list endpoint
    happily returns. group_id is "grp-{reference_photo_id}" (see
    _build_similarity_groups_from_qdrant)."""
    group = similarity_group_service.get_group(group_id)
    if group is None:
        # Rebuild at the cache floor so the broadest grouping is searched —
        # the detail route carries no threshold of its own.
        for g in _build_similarity_groups_from_qdrant(_SIM_CACHE_THRESHOLD):
            if g.get("group_id") == group_id:
                group = g
                break
    if group is None:
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")

    # Deep copy so we don't mutate the stored group
    import copy
    result = copy.deepcopy(group)

    for member in result.get("members", []):
        file_path = member.get("file_path")
        file_hash = member.get("file_hash")
        if file_path and file_hash:
            try:
                member["thumbnail"] = thumbnail_generator.get_thumbnail(file_path, file_hash)
            except Exception:
                member["thumbnail"] = None
        else:
            member["thumbnail"] = None

    return result


@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time progress updates during photo processing.
    
    Broadcasts progress updates including percentage completion and estimated time remaining.
    """
    await websocket.accept()
    try:
        if job_queue_manager is None:
            await websocket.send_json({"error": "Job queue not initialized"})
            await websocket.close()
            return
        
        # Send progress updates every 100ms while job is active
        while True:
            if job_id in job_queue_manager.active_jobs:
                progress_data = await job_queue_manager.get_progress(job_id)
                await websocket.send_json(progress_data)
            else:
                # Job not found or completed
                await websocket.send_json({"status": "not_found"})
                break
            
            await asyncio.sleep(0.1)  # Update every 100ms
    except WebSocketDisconnect:
        pass  # Client disconnected
    except Exception as e:
        print(f"WebSocket error for job {job_id}: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# ─────────────── Serve the built React UI from FastAPI ───────────────
# Native (no-Docker) app: one process serves the API AND the static UI on the
# same origin. In Docker the UI is a separate container, so FRONTEND_DIR is
# unset and this is a no-op. Mounted LAST, so every API route defined above
# takes precedence; the static mount only catches what's left (/, /assets/…).
def _mount_frontend(app_, directory: str) -> bool:
    """Mount `directory` as a static SPA at '/' if it exists. Returns whether
    it was mounted (so callers/tests can assert)."""
    if not directory or not os.path.isdir(directory):
        return False
    from fastapi.staticfiles import StaticFiles
    app_.mount("/", StaticFiles(directory=directory, html=True), name="frontend")
    return True


_mount_frontend(app, os.getenv("FRONTEND_DIR", ""))
