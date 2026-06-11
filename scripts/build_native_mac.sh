#!/usr/bin/env bash
# Build the native, unsigned "Photo Gaze.app" (+ .dmg) for macOS / Apple Silicon.
# No Docker at runtime: SQLite + bundled ONNX model + bundled Qdrant sidecar +
# FastAPI serving the React build, all wrapped by app/native.py.
#
# Run on an Apple-Silicon Mac. Produces dist/Photo Gaze.app and dist/PhotoGaze.dmg.
# The result is UNSIGNED — first launch needs right-click → Open (Gatekeeper).
#
# Prereqs: Python 3.10+, Node 18+, curl/tar, and (for the one-time ONNX export)
# torch installed. Re-run any time; cached steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

QDRANT_VERSION="${QDRANT_VERSION:-v1.7.4}"
QDRANT_ASSET="qdrant-aarch64-apple-darwin.tar.gz"   # macOS arm64

echo "[1/5] Build the React UI (same-origin: relative API + WS-from-location)…"
REACT_APP_API_URL="" REACT_APP_WS_URL="" npm install --no-audit --no-fund --silent
REACT_APP_API_URL="" REACT_APP_WS_URL="" npm run build

echo "[2/5] Ensure the ONNX model exists (export needs torch; one-time)…"
if [ ! -f models/dinov2_vits14.onnx ]; then
  python scripts/export_dinov2_onnx.py models/dinov2_vits14.onnx
else
  echo "      models/dinov2_vits14.onnx present — skipping export."
fi

echo "[3/5] Fetch the Qdrant binary ($QDRANT_VERSION, macOS arm64)…"
mkdir -p vendor
if [ ! -x vendor/qdrant ]; then
  url="https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/${QDRANT_ASSET}"
  echo "      $url"
  curl -fsSL "$url" -o vendor/qdrant.tar.gz
  tar -xzf vendor/qdrant.tar.gz -C vendor
  chmod +x vendor/qdrant
  rm -f vendor/qdrant.tar.gz
else
  echo "      vendor/qdrant present — skipping download."
fi

echo "[4/6] PyInstaller bundle…"
python -m pip install --quiet --upgrade pyinstaller
python -m pip install --quiet -r requirements-native.txt
# --workpath MUST NOT be ./build (PyInstaller's default) — that's the React
# build dir, which the spec bundles; a collision would pack PyInstaller's work
# files into the app.
pyinstaller --noconfirm --workpath .pyi-work photogaze.spec

echo "[5/6] Deep ad-hoc re-sign…"
# PyInstaller's per-file signing leaves an inconsistent seal on a .app, which
# macOS reports as "damaged" (right-click→Open can't bypass that). A single
# deep ad-hoc signature over the assembled bundle makes the seal valid, so the
# app gets the normal "unidentified developer" treatment instead.
# (Do NOT launch the app before this step — running it can write files into the
#  bundle and break the seal again; thumbnails/data go to ~/Library, not here.)
codesign --force --deep --sign - "dist/Photo Gaze.app"
codesign --verify --deep --strict "dist/Photo Gaze.app" \
  && echo "      signature valid" || { echo "      signature INVALID"; exit 1; }

echo "[6/6] Create the .dmg…"
rm -f "dist/PhotoGaze.dmg"
hdiutil create -volname "Photo Gaze" -srcfolder "dist/Photo Gaze.app" \
  -ov -format UDZO "dist/PhotoGaze.dmg"

cat <<'NOTE'

Done:
  app: dist/Photo Gaze.app
  dmg: dist/PhotoGaze.dmg

The app is UNSIGNED (ad-hoc). To run it the first time, EITHER:
  • right-click the app → Open → Open, OR
  • clear the download quarantine after copying it out of the .dmg:
        xattr -dr com.apple.quarantine "/Applications/Photo Gaze.app"
NOTE
