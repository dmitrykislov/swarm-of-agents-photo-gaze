# Photo Gaze — find & remove duplicate photos

Point it at your photo folders. It finds visually similar/duplicate images
(using the DINOv2 vision model), shows them grouped, and lets you delete the
extras safely — the originals are never touched and deleted copies go to a
recoverable trash. Everything runs **locally on your machine**; nothing is
uploaded anywhere.

There are **two ways to run it**:

| | Native macOS app | Docker |
|---|---|---|
| Best for | just using it on a Mac | development / Linux / Windows |
| Install | double-click an app | `./start.sh` |
| Needs Docker? | **No** | Yes |
| Size | ~250 MB app | several GB of images |

---

## A) Native macOS app (no Docker)

### Build it
On an Apple-Silicon Mac (one-time tools: Python 3.10+, Node 18+, and `torch`
just for the model export):

```bash
./scripts/build_native_mac.sh
```

This produces two things in `dist/`:

- **`Photo Gaze.app`** — the actual application. Double-click to run.
- **`PhotoGaze.dmg`** — a disk-image “installer” that just *wraps* the same
  app, for sending/downloading. Open it and drag the app to **Applications**.

### Does it install anything?
**No background services, no admin, no system changes.** The `.app` is
self-contained — double-clicking just *runs* it. "Installing" only means
optionally dragging `Photo Gaze.app` into `/Applications` so it shows up in
Launchpad. The only thing it creates on disk is its **data folder** (below),
the first time you run it.

### Run it
Double-click **Photo Gaze.app**. It starts everything internally and opens your
browser at **http://127.0.0.1:8765**.

> The app is **unsigned** (open-source, no Apple Developer ID). The first time
> you open a copy that came from the `.dmg` or a download, macOS will warn. Do
> **right-click → Open → Open** once, or run:
> `xattr -dr com.apple.quarantine "/Applications/Photo Gaze.app"`

### Where is my data? (native)
Everything lives in **one folder**:

```
~/Library/Application Support/PhotoGaze/
├── app.db        ← database (SQLite): your folders, photo metadata, job state
├── qdrant/       ← the image “fingerprints” (embedding vectors)
├── thumbnails/   ← cached previews (safe to delete; regenerated)
├── backups/      ← automatic backups
└── trash/        ← deleted duplicates (recoverable) + recovery manifests
```

Your **original photos stay where they are** on disk and are never modified.

### Stopping & restarting
- **Quit** the app (close it / Cmd-Q) → the internal server and vector engine
  stop. **Nothing is lost** — everything is on disk in the folder above.
- **Next launch** reads that folder back, so all your registered folders,
  embeddings, and duplicate groups are exactly as you left them.
- If a scan was interrupted, the **Processing Status** panel shows
  *“Resume processing”* to finish the remaining photos.

### Uninstall (native)
1. Delete `Photo Gaze.app` (drag to Trash).
2. Delete the data folder: `rm -rf "~/Library/Application Support/PhotoGaze"`.

That's it — there's nothing else on your system. (Your original photos and
anything you recovered are untouched.)

---

## B) Docker (development / Linux / Windows)

```bash
./start.sh            # build images on first run, start everything
./start.sh --logs     # same, then tail logs
./start.sh --rebuild  # force-rebuild the app images
./start.sh --down     # stop containers (your data volumes are kept)
```

Open the UI at **http://localhost:3001** (the script prints the exact URLs).

**Ports** are set at the top of `start.sh` — change any number if it's already
in use on your machine (defaults: UI `3001`, API `8000`, Postgres `5433`,
Qdrant `6333`). Re-run `./start.sh` after editing.

### Where is my data? (Docker)
| What | Where | Survives `--down`? |
|---|---|---|
| Database (folders, metadata, jobs) | Postgres — `postgres_data` Docker volume | ✅ yes |
| Image fingerprints (vectors) | Qdrant — `qdrant_storage` Docker volume | ✅ yes |
| Deleted duplicates (trash) | `~/.photo-gaze-trash/` on your machine | ✅ yes |
| Thumbnails | inside the container | ❌ rebuilt on demand |
| Originals | your filesystem (mounted, never modified) | ✅ yes |

- **`./start.sh --down`** stops the containers but **keeps your data** — the
  next `./start.sh` brings it all back.
- **`docker compose down -v`** **deletes the data volumes** (database +
  vectors). Your original photos and the trash folder on disk are unaffected.

---

## How to use it (either mode)

1. **Add a folder** — *Photo Folders → Browse & Add*, pick a folder on your
   disk, *Select this folder*.
2. **Scan** it — click **Scan**. The app reads the photos (JPEG, PNG, WebP,
   HEIC/HEIF, TIFF, RAW… 22 formats) and generates a “fingerprint” for each.
   Watch **Processing Status** fill up. (CPU does ~40 photos/min; Apple Silicon
   native is much faster.)
3. **Review duplicates** — similarity groups appear automatically. Drag the
   **Similarity threshold** slider (e.g. `0.95` = near-identical, lower = looser
   matches). Page through all groups with the pager.
4. **Inspect a group** — click it. Each photo shows its size, type, resolution,
   quality, and similarity. The **★ Best** copy (highest quality) is kept by
   default; the rest are pre-marked for deletion. Click any photo for a
   full-screen view; toggle keep/delete; or **Mark as Best** to override.
5. **Delete** — removes the marked copies (they go to **trash**, recoverable).
6. **Auto-deduplicate** (optional, fast) — set the slider to `1.00`, click
   **Auto-deduplicate**, choose a *“source of truth”* folder, preview the plan,
   and sweep every pure-duplicate cluster in one go.
7. **Recover** — the **Trash** page lists everything you deleted; select and
   **Recover** to restore files *and* their database/index entries.

**How “which copy to keep” is decided:** highest quality wins (largest file =
least compression; +20% for universal JPEG/PNG). Ties break by earliest-taken
(the likely original), then shortest filename. The same rule drives both the
manual “★ Best” and auto-dedupe, so they always agree.

**Nothing destructive is permanent:** deletes move files to trash with a
manifest; recovery rebuilds everything. Removing a *folder* asks for
confirmation, then clears its photos + fingerprints from the app (originals on
disk untouched).

---

## Under the hood

- **Model:** DINOv2 ViT-S/14 → 384-dim vectors, cosine similarity. In Docker it
  runs via PyTorch; the native app runs the same model via **ONNX Runtime**
  (no PyTorch → ~250 MB instead of multiple GB; verified numerically identical).
- **Vector search:** Qdrant (HNSW) — fast even at 200k–300k photos.
- **Database:** PostgreSQL (Docker) or SQLite (native).
- **UI:** React/TypeScript, served by FastAPI.

```
React UI ──HTTP/WS──▶ FastAPI ──▶ Postgres/SQLite  (metadata, jobs, folders)
                          ├──────▶ Qdrant           (384-dim vectors)
                          └──────▶ DINOv2 (torch or ONNX)  (in-process)
```

### Inspect things directly (Docker; default API port 8000)
```bash
curl -s http://localhost:8000/stats | jq                       # counts
curl -s "http://localhost:8000/similarity-groups?min_similarity=0.9" | jq '.total'
open http://localhost:6333/dashboard                           # Qdrant dashboard
```

### Key API endpoints
| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | liveness |
| `GET` | `/stats` | photo / embedding / processing counts |
| `GET`·`POST`·`DELETE` | `/folders` … `/folders/{id}` | list / add / remove (cascade) folders |
| `POST` | `/folders/{id}/scan`, `/rescan`, `/process-pending` | scan / resume |
| `GET` | `/similarity-groups` | duplicate groups (params: `min_similarity`, `skip`, `limit`) |
| `POST` | `/deduplicate`, `/auto-deduplicate` | delete selected / sweep clusters |
| `GET`·`POST` | `/trash`, `/trash/recover` | list / restore trashed files |
| `GET` | `/photos/{id}/full`, `/thumbnails/{id}` | full image / thumbnail |
| `WS` | `/ws/progress/{job_id}` | live progress |

Full, always-current API reference: the Swagger UI at `/docs`.

## Develop (without Docker or packaging)
```bash
# Backend (Postgres + Qdrant reachable via env)
pip install -r requirements.txt
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/app_db \
QDRANT_URL=http://localhost:6333 uvicorn app.main:app --reload
# Frontend dev server
npm install && npm run dev          # http://localhost:5173
# Tests
pip install -r requirements-dev.txt && pytest -v
```
See `docs/NATIVE_BUILD.md` for the server-less / native build details.

---

## Origin

This project was built end-to-end by a **swarm of autonomous AI agents** —
created by [Dmitry Kislov](https://www.linkedin.com/in/dmitrykislov/) — from an
empty repository and 40 task descriptions: a multi-service photo-deduplication
platform shipped in ~72 minutes for ~$6.31 in API cost (40/40 tasks, 327 LLM
calls, model `claude-haiku-4-5`). It has since been extended and hardened
(scalability to 300k photos, the native macOS app, bug fixes) with tests.
