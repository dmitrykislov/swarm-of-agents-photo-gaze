# PyInstaller spec for the native "Photo Gaze" macOS app.
#   Build with:  scripts/build_native_mac.sh   (assembles the bundled assets
#   first, then runs `pyinstaller --noconfirm photogaze.spec`).
#
# Entry point is app/native.py, which starts the Qdrant sidecar + uvicorn,
# serves the bundled React build, uses SQLite, and runs the ONNX model.
import os

root = os.path.abspath(os.getcwd())

# Bundled assets. Paths inside the bundle are resolved by app.native.bundled()
# relative to sys._MEIPASS:
#   models/dinov2_vits14.onnx, build/ (React), and ./qdrant (sidecar binary).
datas = [
    (os.path.join(root, "models", "dinov2_vits14.onnx"), "models"),
    (os.path.join(root, "build"), "build"),
]
binaries = [(os.path.join(root, "vendor", "qdrant"), ".")]

# uvicorn loads its protocol/loop impls dynamically — pull them in explicitly.
hiddenimports = [
    "app.main",
    "onnxruntime",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on",
    "pillow_heif",
]

a = Analysis(
    ["app/native.py"],
    pathex=[root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=["torch", "torchvision", "timm", "psycopg2", "pg8000"],  # keep it small
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="photogaze", console=False,
    bootloader_ignore_signals=False, strip=False, upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="photogaze")
app = BUNDLE(
    coll,
    name="Photo Gaze.app",
    icon=None,
    bundle_identifier="ai.photogaze.app",
    info_plist={"LSBackgroundOnly": False, "CFBundleDisplayName": "Photo Gaze"},
)
