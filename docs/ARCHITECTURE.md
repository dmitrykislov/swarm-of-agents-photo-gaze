# Architecture Overview

A self-hosted photo deduplication system: a React/TypeScript UI talks to a
FastAPI backend that stores photo metadata in PostgreSQL, 384-dim DINOv2
embeddings in Qdrant, and finds/serves near-duplicate groups. Everything runs
locally via Docker Compose.

## System diagram

```
┌──────────────────────────────────────────────────────────────┐
│  React UI (host port REACT_PORT, default 3001)                 │
│  App.tsx · SimilarPhotosGrid · GroupDetailView ·               │
│  AutoDeduplicateModal · TrashPage · ProgressBar                │
└──────────────────────────────────────────────────────────────┘
                  ↓ HTTP + WebSocket (to FASTAPI_PORT, default 8000)
┌──────────────────────────────────────────────────────────────┐
│  FastAPI backend (app/)                                        │
│  REST + WebSocket · metrics & error middleware                 │
│  JobQueueManager · FolderScanner · EmbeddingGenerator (DINOv2, │
│  in-process) · in-memory similarity index · BackupManager      │
└──────────────────────────────────────────────────────────────┘
        ↓ SQLAlchemy        ↓ vectors          ↓ /metrics
  ┌──────────────┐    ┌──────────────┐   ┌──────────────────────┐
  │ PostgreSQL    │    │ Qdrant       │   │ Prometheus +         │
  │ (metadata)    │    │ (embeddings) │   │ Alertmanager         │
  └──────────────┘    └──────────────┘   └──────────────────────┘
```

All host ports are configurable at the top of `start.sh` (or via the same env
vars in `docker-compose.yml`). Containers talk to each other over the Docker
network on fixed internal ports (`postgres:5432`, `qdrant:6333`,
`fastapi:8000`), so changing a host port never breaks inter-service wiring.

## Backend modules (`app/`)

- **main.py** — FastAPI app, all HTTP/WebSocket endpoints, Prometheus + error
  middleware, startup hooks, and the in-memory similarity index (see below).
- **folder_scanner.py** — recursive scan with incremental change detection
  (new/modified via SHA-256 hash, deleted scoped to the scanned folder).
  Excludes the trash dir and system/VCS dirs.
- **metadata_extractor.py** — Pillow-based format validation, dimensions, and
  SHA-256 file hash (HEIC/HEIF via `pillow-heif`).
- **embedding_generator.py** — DINOv2 **ViT-S/14**, images resized to
  **224×224**, output **384-dim** L2-normalized vectors. Device auto-detect:
  **MPS** (Apple Silicon) → CUDA → CPU. No GPU is required.
- **job_queue.py** — `JobQueueManager`: async per-photo processing with a
  concurrency semaphore, per-photo progress persistence, completion detection,
  and checkpoint recovery on restart.
- **qdrant_client.py** — thin helper wrapper (the live code mostly uses the
  `qdrant-client` library directly).
- **backup_manager.py** — periodic local snapshots of Postgres/Qdrant data.
- **models.py** — SQLAlchemy models (see schema below).
- **orchestrator.py** — an alternative scan→queue→complete driver. *Not wired
  into the running HTTP app* (the live endpoints drive `JobQueueManager`
  directly); kept for reference/tests.

## Frontend (`src/`)

- **App.tsx** — top-level state, folder panel with an inline server-side
  browser, processing status + progress WebSocket, threshold slider.
- **SimilarPhotosGrid.tsx** / **useSimilaritySearch.ts** — fetch and render
  similarity groups (debounced on the threshold). Groups are global, not tied
  to a job.
- **GroupDetailView.tsx** — per-group review modal + full-res lightbox,
  keep/delete toggles, "Mark as Best".
- **AutoDeduplicateModal.tsx** — pick a source-of-truth folder, preview a
  dry-run plan, then execute.
- **TrashPage.tsx** — list and recover trashed files.
- **api.ts** — typed API client. The backend base URL is baked at build time
  from `REACT_APP_API_URL` (a docker build arg derived from `FASTAPI_PORT`).

## Data flow

```
1. Add a folder (POST /folders) and scan it (POST /folders/{id}/scan → /rescan).
   FolderScanner inserts photos + processing_state(pending) rows.
2. Each photo is processed by JobQueueManager.process_photo:
   - DINOv2 embedding (384-dim) → upserted to Qdrant
   - pointer row written to the `embeddings` table; processing_state → completed
   - progress streamed over /ws/progress/{job_id}
3. notify_embeddings_changed(photo_id) debounces an index update; the new
   photos are folded into the in-memory similarity index incrementally.
4. GET /similarity-groups clusters the index at the requested threshold and
   returns groups (reference "best" photo + similar photos + reasons).
5. Deduplicate (manual /deduplicate or sweep /auto-deduplicate): losing
   copies move to the trash dir with a recovery manifest; their Photo /
   Embedding / ProcessingState rows and Qdrant points are removed; the index
   is updated incrementally.
```

### Similarity index

Built once at startup and kept in memory:

- **Vectors** — all unit-normalized embeddings (for exact reference-vs-member
  cosine scoring).
- **Edge index** — a score-sorted, undirected, de-duplicated set of pairs
  above `_SIM_CACHE_THRESHOLD` (0.70), stored as three numpy arrays. This is a
  *sparse* index (not a dense N×N matrix).

Clustering a request at threshold *t* takes a `searchsorted` slice of the edge
arrays (edges ≥ *t*) plus a BFS over only those edges — **O(active edges)**,
independent of collection size, so the threshold slider stays responsive at
hundreds of thousands of photos. Groups are **connected components** (so
transitive near-duplicates A~B~C stay in one group); the same clustering backs
both the group view and auto-dedupe.

Index maintenance:
- **Additions** (after a scan) are folded in incrementally — only the new
  vectors are searched against Qdrant and their edges merged.
- **Deletions** (dedupe / folder removal) filter the in-memory index in
  O(edges); no full Qdrant re-scroll.
- A full rebuild (`_recompute_sim_cache`) runs at startup, on recovery, and as
  a fallback.

## Database schema (PostgreSQL)

Tables (see `app/models.py`): `folder_paths`, `photos`, `embeddings`,
`processing_state`, `user_preferences`, `job_queue`.

- **photos** — `id`, `filename`, `file_path` (unique), `file_size`,
  `mime_type`, `file_hash` (nullable), `uploaded_at`, `user_id`. Width/height
  are read on demand from the file (not stored).
- **embeddings** — pointer rows: `photo_id`, `embedding_model`,
  `vector_dimension`, `qdrant_point_id`. The actual vector lives in Qdrant.
- **processing_state** — per-photo pipeline status (`pending`/`completed`/
  `failed`) + timestamps.
- **job_queue** — scan/processing jobs: status, totals, checkpoint info.

### Qdrant

Collection **`embeddings`**, **384-dim**, **cosine** distance. Each point's
payload is `{"photo_id": N}`.

## Cross-cutting concerns

- **Fault tolerance** — `job_queue` checkpoints progress per photo; incomplete
  jobs are recovered on startup. Per-photo failures are recorded and skipped
  so a batch still completes.
- **Deletion safety** — deduplication never deletes a file outright: it moves
  it to the trash dir with a manifest (full DB + vector snapshot) and only then
  purges DB/Qdrant. If the file move fails, DB rows are left intact. Recovery
  restores the file and rebuilds the rows + Qdrant point.
- **Monitoring** — Prometheus metrics at `/metrics`
  (`fastapi_requests_total`, `fastapi_request_duration_seconds`,
  `fastapi_active_requests`, `fastapi_errors_total`); `/health` liveness.
- **CORS** — any `localhost`/`127.0.0.1` origin (any port) is allowed, so the
  UI works regardless of `REACT_PORT`; extra origins via `CORS_ORIGINS`.
- **Security note** — there is no authentication; this is a single-user,
  self-hosted tool. The server-side `/browse` endpoint can list any directory
  the backend can read. Add auth and restrict origins before any networked
  deployment.

## Deployment

Docker Compose services: `postgres` (15-alpine), `qdrant`, `fastapi`, `react`,
`prometheus`, `alertmanager`, `node-exporter`. Bring everything up with
`./start.sh`. There is no Kubernetes manifest in this repo.
