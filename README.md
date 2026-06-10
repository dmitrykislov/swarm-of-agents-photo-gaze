# DINOv2 Photo Similarity Search & Deduplication System

## This entire project was written by AI agents. No human wrote a single line.

A **swarm of autonomous AI agents** — built by [Dmitry Kislov](https://www.linkedin.com/in/dmitrykislov/) — took an empty repository and 40 task descriptions and shipped a production-grade, multi-service photo deduplication platform: **11,400 lines of code, 285 tests, 7 Docker services, 20 API endpoints — in 72 minutes for $6.31 in API costs.**

A senior engineer would need 2–3 weeks. A typical team, 8–12 weeks. The swarm did it before lunch.

### Build stats

| Metric | Value |
|---|---|
| Tasks completed | 40 / 40 (100%) |
| Rework cycles | 13 |
| LLM calls | 327 |
| **Total tokens** | **4,066,227** |
| Sum of task durations | 72.3 minutes |
| Model | claude-haiku-4-5 |
| **API cost** | **~$6.31 USD** |

Cost math: 3.51M input × $1/MTok + 0.56M output × $5/MTok = **$6.31**

### Resulting code

- **55 Python files** (FastAPI backend, services, tests)
- **33 frontend files** (26 TypeScript/TSX + 7 JavaScript/JSX — React components, API client, hooks, configs)
- **~11,400 lines of code** (excluding generated/dependency files)
- **285 tests** across 29 test files
- **20 REST/WebSocket API endpoints**
- **7 Docker Compose services** (postgres, qdrant, fastapi, react, prometheus, alertmanager, node-exporter)

### How the swarm works

Each task goes through: **investigate → implement → build → code review → merge → post-merge verification**. Code review caught 13 issues across 40 tasks — each resolved via automated rework. No human intervention at any step.

---

## What it does

A self-hosted photo deduplication tool powered by the DINOv2 vision transformer. Point it at your photo folders, and it finds visually similar images, ranks them by quality, and lets you delete duplicates — moving them to a trash folder for safety. Runs entirely on your machine via Docker Compose.

**Stack:** FastAPI + React/TypeScript + PostgreSQL + Qdrant + Prometheus/Alertmanager.

1. **Browse & register folders** from the UI — no config files needed. The server-side browser lets you navigate your filesystem and select folders.
2. **Scan folders recursively** for photos (JPEG, PNG, WebP, HEIC/HEIF, TIFF, AVIF, camera RAW: DNG, CR2, NEF, ARW, ORF, RW2, PEF — 22 formats total).
3. **Generate embeddings** via DINOv2 ViT-S/14 at 224×224. Runs on CPU at ~40 photos/min in Docker, much faster natively on Apple Silicon with MPS.
4. **Store** metadata in PostgreSQL, 384-dim vectors in Qdrant (cosine similarity).
5. **Find similar images** with a configurable similarity threshold (default 0.95 for near-exact duplicates).
6. **Rank duplicates** within each group (manual review): largest file wins (less compression = more detail), with a 20% bonus for JPEG/PNG (universally compatible formats). Ties broken by earliest upload and shortest filename (e.g. `photo.jpg` beats `photo (1).jpg`). This is the photo flagged "★ Best" in the group view, and a quality score (file size normalized across the collection) is shown per photo. Full reasoning shown in the UI.
7. **Inspect at full resolution** — click any photo for a lightbox with arrow-key navigation, loading spinner, metadata overlay (resolution, file size, type, date created, full path), and keep/delete toggle.
8. **Deduplicate** two ways: **manually** (open a group, mark photos, Delete) or **auto-deduplicate** — pick a "source of truth" folder and sweep every pure-duplicate cluster, keeping the single **earliest-taken** copy inside that folder and trashing the rest. Either way, removed files move to `~/.photo-gaze-trash/` (configurable via `TRASH_DIR`) with a manifest for recovery; their DB records and Qdrant vectors are cleaned up and the kept copy stays on disk.

## Quick start

```bash
./start.sh            # builds images on first run, brings everything up
./start.sh --logs     # same + tail aggregated logs
./start.sh --rebuild  # force-rebuild fastapi/react images
./start.sh --down     # stop and remove containers (volumes are kept)
```

Then open the UI (default **http://localhost:3001** — `start.sh` prints the exact URLs on startup; ports are configurable at the top of `start.sh`) and:

1. Click **Browse & Add** to navigate to a photo folder on your Mac
2. Click **Scan** to discover photos and start generating embeddings
3. Watch the **progress bar** fill as photos are processed
4. When done, similarity groups appear — **click any group** to inspect
5. In the detail view, review photos at full resolution, **mark duplicates**, and hit **Delete**

## UI features

- **Photo Folders panel** — browse & add folders, per-folder scan, remove (cascading delete through DB + Qdrant + trash)
- **Processing Status** — live progress bar with percentage, pulsing indicator when active, "Resume processing" button for interrupted jobs
- **Similarity threshold slider** — adjust from 0.00 to 1.00 to find looser or stricter matches
- **Clickable group rows** — hover highlights, click to open detail modal
- **Detail modal** — all photos side by side with metadata (resolution, file size, type, date created, path), "why best" explanation, color-coded KEEPING/DELETING badges, "Mark as Best" override, "Select all including best" option
- **Full-resolution lightbox** — click any photo to view at original quality. Left/right arrows cycle through the batch. Bottom overlay shows all metadata + full file path. Keep/delete toggle available inline. HEIC and other non-browser formats auto-transcoded to JPEG on-the-fly.

## Where the data lives

| What | Where | Notes |
|---|---|---|
| **Photo files (originals)** | Your filesystem | Mounted into the container at the same path. The app moves deleted duplicates to `~/.photo-gaze-trash/` but never modifies originals. |
| **Trash (deleted duplicates)** | `~/.photo-gaze-trash/` | Moved files with timestamp prefix. `*_manifest.json` records original paths for recovery. |
| **Photo metadata, folders, processing state** | Postgres (`postgres_data` volume) | Tables: `photos`, `folder_paths`, `processing_state`, `job_queue`, `embeddings` (pointer rows), `user_preferences`. |
| **Embedding vectors (384-dim floats)** | Qdrant (`qdrant_storage` volume) | Collection `embeddings`, cosine distance. Each vector has `{"photo_id": N}` payload. |
| **Thumbnails** | Inside the backend container at `/app/.thumbnail_cache/` | Keyed by file hash; regenerated on demand; safe to lose. |
| **Embedding model weights** | `torch_cache` volume | DINOv2 ViT-S/14 (~90MB). Persisted so rebuilds don't re-download. |
| **Prometheus metrics** | `prometheus_data` volume | 15 days retention. |

### Data flow

```
Photo folder on disk
        │
        ▼
    Scan (recursive, 22 formats)
        │
        ▼
  photos + processing_state (Postgres)
        │
        ▼
  DINOv2 ViT-S/14 @ 224×224 (CPU, 2 concurrent)
        │
   ┌────┴────┐
   ▼         ▼
 Qdrant    embeddings row (Postgres)
 384-dim   (photo_id → qdrant_point_id)
   │
   ▼
 /similarity-groups (on-demand grouping from Qdrant)
   │
   ▼
 UI grid → detail modal → lightbox → deduplicate → trash
```

### Inspecting data directly

```bash
# Registered folders
curl -s http://localhost:8000/folders | jq

# Processing counts
curl -s http://localhost:8000/stats | jq

# Browse server filesystem
curl -s "http://localhost:8000/browse?path=/Users" | jq

# Similarity groups
curl -s "http://localhost:8000/similarity-groups?min_similarity=0.9" | jq

# Postgres
docker exec -it postgres_db psql -U postgres -d app_db \
  -c "SELECT COUNT(*) photos, (SELECT COUNT(*) FROM embeddings) embeddings FROM photos;"

# Qdrant dashboard
open http://localhost:6333/dashboard
```

### Deleting data

- **Deduplicate from the UI** — moves files to `~/.photo-gaze-trash/`, removes DB + Qdrant records.
- **Remove a folder** — cascading delete: `folder_paths` → `photos` → `processing_state` → `embeddings` → Qdrant vectors. Original files untouched.
- **`./start.sh --down`** — stops containers, keeps volumes.
- **`docker compose down -v`** — **destroys all volumes**. Originals and trash folder on disk are unaffected.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/stats` | Photo/embedding/processing counts |
| `GET` | `/folders` | List registered folders |
| `POST` | `/folders` | Register a new folder |
| `DELETE` | `/folders/{id}` | Remove folder + cascade delete all data |
| `POST` | `/folders/{id}/scan` | Scan a specific folder |
| `GET` | `/browse?path=...` | List subdirectories for folder picker |
| `POST` | `/rescan` | Scan default folder for changes |
| `POST` | `/process-pending` | Resume embedding generation |
| `POST` | `/deduplicate` | Move selected photos to trash + clean DB/Qdrant |
| `POST` | `/auto-deduplicate` | Sweep duplicate clusters, keep earliest copy in a folder (supports `dry_run`) |
| `GET` | `/trash` | List recoverable trashed files |
| `POST` | `/trash/recover` | Restore trashed files + rebuild DB/Qdrant from manifest |
| `GET` | `/similarity-groups` | Compute groups on-demand from the similarity index |
| `GET` | `/thumbnails/{id}` | Cached thumbnail (JPEG) |
| `GET` | `/photos/{id}/full` | Full-res photo (HEIC auto-transcoded to JPEG) |
| `GET` | `/job-queue/status` | Current queue state |
| `WS` | `/ws/progress/{job_id}` | Real-time progress updates |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/backup/manual` | Trigger manual backup |
| `GET` | `/backup/status` | Backup status |
| `POST` | `/backup/recover/{id}` | Restore from backup |

## Architecture

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  React   │────▶│ FastAPI  │────▶│ Postgres │  (metadata, jobs, folders)
│ Frontend │     │ Backend  │     └──────────┘
└──────────┘     │          │     ┌──────────┐
                 │          │────▶│  Qdrant  │  (384-dim embeddings)
                 │          │     └──────────┘
                 │          │     ┌──────────┐
                 │          │────▶│ DINOv2   │  (ViT-S/14 in-process)
                 └──────────┘     └──────────┘
                       │
                       │ /metrics
                       ▼
                 ┌──────────┐     ┌──────────────┐
                 │Prometheus│────▶│ Alertmanager │
                 └──────────┘     └──────────────┘
```

## Local dev (without Docker)

```bash
# Backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/app_db \
QDRANT_URL=http://localhost:6333 \
uvicorn app.main:app --reload

# Frontend (Vite dev server)
npm install && npm run dev

# Tests
pip install -r requirements-dev.txt
pytest -v
```

---

Built by an AI agent swarm created by [Dmitry Kislov](https://www.linkedin.com/in/dmitrykislov/).
