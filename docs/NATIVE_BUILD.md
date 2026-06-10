# Native macOS app (no Docker)

Photo Gaze can run as a single, unsigned, double-click macOS app — no Docker,
no Postgres, no separate Qdrant server. It bundles:

- **SQLite** for metadata (instead of Postgres),
- a **Qdrant binary** run as a local sidecar (keeps HNSW → 300k-photo scale),
- the **DINOv2 ViT-S/14 model as ONNX** + **ONNX Runtime** (no PyTorch → the
  bundle is ~250 MB instead of multiple GB),
- the **React UI served by FastAPI** on the same origin.

`app/native.py` is the launcher: it picks a free port for the Qdrant sidecar,
starts uvicorn on a fixed loopback port (`PHOTO_GAZE_PORT`, default 8765),
opens your browser, and shuts the sidecar down on exit. Data lives in
`~/Library/Application Support/PhotoGaze/`.

## Build it

On an Apple-Silicon Mac (Python 3.10+, Node 18+; `torch` only needed once, to
export the ONNX model):

```bash
./scripts/build_native_mac.sh
```

That script:
1. builds the UI same-origin (`REACT_APP_API_URL="" npm run build`),
2. exports `models/dinov2_vits14.onnx` if missing (verifies torch↔ONNX parity),
3. downloads a pinned Qdrant macOS-arm64 binary into `vendor/`,
4. runs PyInstaller (`photogaze.spec`),
5. produces **`dist/Photo Gaze.app`** and **`dist/PhotoGaze.dmg`**.

The result is **unsigned** (open-source; no Apple Developer ID): the first
launch needs **right-click → Open** to get past Gatekeeper.

## Run the server-less stack without packaging (for development)

```bash
pip install -r requirements-native.txt          # onnxruntime, no torch
python scripts/export_dinov2_onnx.py             # one-time, needs torch
PHOTO_GAZE_PORT=8765 QDRANT_BINARY=vendor/qdrant python -m app.native
```

Or point it at an already-running Qdrant by leaving `QDRANT_BINARY` unset and
setting `QDRANT_URL`.

## Notes / knobs

- `PHOTO_GAZE_DATA_DIR` — override the data directory.
- `EMBEDDING_ONNX_COREML=1` — try Apple's CoreML execution provider (CPU stays
  the fallback).
- Windows later: the same design (SQLite + ONNX + Qdrant sidecar + static UI)
  ports over — swap the Qdrant binary and use a PyInstaller `.exe`/MSI.
