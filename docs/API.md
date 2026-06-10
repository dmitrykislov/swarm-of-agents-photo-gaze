# API Documentation

The backend is FastAPI. **The authoritative, always-up-to-date reference is the
auto-generated Swagger UI** at `/docs` (and ReDoc at `/redoc`) — this page is a
hand-written summary of the real endpoints.

- **Base URL**: `http://localhost:8000` by default. The host port is
  `FASTAPI_PORT` (set at the top of `start.sh`); the path prefix is unchanged.
- **Swagger UI**: `http://localhost:8000/docs`
- **Auth**: none (single-user, self-hosted tool).

## Response & error format

All responses are JSON. Errors use a consistent envelope:

```json
{ "error": "message", "detail": "optional detail", "path": "/endpoint" }
```

## Endpoints

### System

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | Liveness. Returns `{"status": "healthy"}`. |
| `GET` | `/stats` | Counts: `photos`, `embeddings`, `completed`, `pending`, `failed`, plus a `similarity_index` sub-object (last recompute, vector/edge counts). |
| `GET` | `/metrics` | Prometheus metrics. |
| `GET` | `/job-queue/status` | Queue totals and active job ids. |
| `WS`  | `/ws/progress/{job_id}` | Streams `{percentage, processed_photos, total_photos, eta_seconds, status}` (~10/s) until the job leaves the active set. |

### Folders & scanning

| Method | Path | Notes |
|---|---|---|
| `GET` | `/folders` | List registered folders. |
| `POST` | `/folders` | Body `{"path": "..."}`. Rejects paths inside the trash dir. |
| `DELETE` | `/folders/{folder_id}` | Cascade-delete photos/embeddings/state under the folder + Qdrant points. |
| `POST` | `/folders/{folder_id}/scan` | Scan a registered folder (delegates to `/rescan`). |
| `POST` | `/rescan` | Query `folder_path` (defaults to `PHOTOS_FOLDER`). Incremental change detection; returns `{job_id, changes_found, ...}` (202). |
| `POST` | `/process-pending` | Queue all `pending` photos for embedding. |
| `POST` | `/stop-processing` | Cancel active jobs (pending photos stay pending). |
| `GET` | `/browse?path=` | List subdirectories + image count for the folder picker. |

### Similarity groups

| Method | Path | Notes |
|---|---|---|
| `GET` | `/similarity-groups` | Query: `min_similarity` (clustering threshold, default 0.85), `min_quality`, `sort_by` (`similarity`\|`quality`), `skip`, `limit`. Returns `{total, skip, limit, groups[]}`; each group has `reference_photo` ("best" kept), `similar_photos[]` (with per-photo `similarity_score` and `quality_score`), and `best_reasons[]`. |
| `GET` | `/similarity-groups/{group_id}` | One group by id (`grp-{referencePhotoId}`); falls back to the live index if not in the in-memory store. |

Thresholds map to cosine similarity; the index floor is 0.70, so values below
that are clamped. Groups are connected components of the duplicate graph.

### Deduplication & trash

| Method | Path | Notes |
|---|---|---|
| `POST` | `/deduplicate` | Body `{"photo_ids": [...]}`. Moves files to the trash dir (with a recovery manifest), purges DB rows + Qdrant points. Returns `{deleted, moved_to_trash, trash_dir, errors}`. |
| `POST` | `/auto-deduplicate` | Body `{"folder_path", "threshold"=1.0, "dry_run"=false}`. Keeps the single earliest-taken copy inside `folder_path` per duplicate cluster, trashes the rest. `dry_run` returns the plan without touching anything. |
| `GET` | `/trash` | List recoverable trashed files (path, original path, size, timestamp). |
| `GET` | `/trash/thumbnail?path=&size=240` | Thumbnail for a trashed file (path must resolve inside the trash dir). |
| `POST` | `/trash/recover` | Body `{"trash_paths": [...]}`. Moves files back and rebuilds DB rows + Qdrant points from the manifest snapshot. |

### Photos & thumbnails

| Method | Path | Notes |
|---|---|---|
| `GET` | `/thumbnails/{photo_id}?size=200` | Cached JPEG thumbnail. Validates `photo_id` and `size` (32–2048). |
| `GET` | `/photos/{photo_id}/full` | Full-resolution image; HEIC/HEIF/TIFF/etc. transcoded to JPEG on the fly. |
| `GET` | `/photos/{photo_id}/image-info` | Lazy `{width, height, created_date}` (read from the file; used by the lightbox). |

### Backup

| Method | Path | Notes |
|---|---|---|
| `POST` | `/backup/manual` | Trigger a backup (202). |
| `GET` | `/backup/status` | Recent backups + recovery options. |
| `POST` | `/backup/recover/{backup_id}` | Restore from a backup. |

## Examples

```bash
# Register and scan a folder
curl -X POST http://localhost:8000/folders \
  -H 'Content-Type: application/json' -d '{"path": "/Users/me/Pictures"}'
curl -X POST "http://localhost:8000/folders/1/scan"

# Duplicate groups at a threshold
curl -s "http://localhost:8000/similarity-groups?min_similarity=0.95" | jq '.total'

# Dry-run an auto-dedupe sweep (keeps earliest copy under the folder)
curl -X POST http://localhost:8000/auto-deduplicate \
  -H 'Content-Type: application/json' \
  -d '{"folder_path": "/Users/me/Pictures", "threshold": 1.0, "dry_run": true}'

# Recover a trashed file
curl -X POST http://localhost:8000/trash/recover \
  -H 'Content-Type: application/json' \
  -d '{"trash_paths": ["/Users/me/.photo-gaze-trash/20260610_102506_29_x.webp"]}'
```
