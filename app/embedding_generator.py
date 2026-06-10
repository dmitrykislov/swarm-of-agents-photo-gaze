"""DINOv2 ViT-S/14 embedding generator (384-dim).

Two interchangeable backends with an identical public API:
  - "onnx"  — ONNX Runtime + a pre-exported dinov2_vits14.onnx. Tiny footprint
              (no PyTorch), used by the packaged native app. Numerically
              identical to torch (verified: cosine ~1.0, max|Δ| ~3e-7).
  - "torch" — torch.hub DINOv2, used by the Docker stack.

Backend selection (EmbeddingGenerator.__init__ / EMBEDDING_BACKEND env):
  - explicit "onnx"/"torch", or
  - auto (default): use ONNX if the .onnx model AND onnxruntime are available,
    otherwise fall back to torch.

torch is imported lazily inside the torch backend so this module (and the
native app) load without PyTorch installed.
"""
import io
import os
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

# Register HEIF/HEIC decoder so Image.open handles .heic files (both backends).
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# ImageNet normalization constants (shared by both backends).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_DEFAULT_ONNX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "dinov2_vits14.onnx"
)


class EmbeddingGenerator:
    """Generate 384-dimensional, L2-normalized DINOv2 ViT-S/14 embeddings.

    Images are resized to 224x224 (faster than the 518 default, plenty for
    similarity). The keeper ranking and all thresholds are backend-agnostic
    because the two backends produce the same vectors.
    """

    def __init__(
        self,
        device: str = None,
        backend: Optional[str] = None,
        onnx_path: Optional[str] = None,
        session=None,
    ):
        self.model_name = "dinov2_vits14"
        self.embedding_dim = 384
        self._onnx_path = onnx_path or os.getenv("DINOV2_ONNX_PATH", _DEFAULT_ONNX_PATH)

        backend = backend or os.getenv("EMBEDDING_BACKEND")
        if session is not None:
            backend = "onnx"  # injected session (tests) implies ONNX path
        if backend is None:
            backend = "onnx" if self._onnx_available() else "torch"
        self.backend = backend

        if backend == "onnx":
            self.device = "onnx"
            if session is not None:
                self._session = session
            else:
                import onnxruntime as ort
                self._session = ort.InferenceSession(
                    self._onnx_path, providers=self._onnx_providers()
                )
            self._input_name = self._session.get_inputs()[0].name
        else:
            import torch
            self._torch = torch
            self.device = device or self._detect_device()
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.model = self.model.to(self.device)
            self.model.eval()

    # ----------------------------- backend setup -----------------------------

    def _onnx_available(self) -> bool:
        if not os.path.isfile(self._onnx_path):
            return False
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _onnx_providers() -> List[str]:
        """CPU by default. Set EMBEDDING_ONNX_COREML=1 to try Apple's CoreML
        execution provider first (faster on Apple Silicon; CPU stays as the
        fallback)."""
        if os.getenv("EMBEDDING_ONNX_COREML") == "1":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _detect_device(self) -> str:
        """Detect optimal torch device: MPS (Apple) > CUDA > CPU."""
        torch = self._torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # ----------------------------- inference ----------------------------------

    def _preprocess(self, image_data: bytes) -> np.ndarray:
        """Bytes -> (1, 3, 224, 224) float32, ImageNet-normalized. Pure
        Pillow + numpy (no torch), identical math to the original torch path."""
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0   # HWC
        arr = arr.transpose(2, 0, 1)                          # CHW
        arr = (arr - _MEAN) / _STD
        return arr[np.newaxis, ...].astype(np.float32)        # (1, 3, 224, 224)

    def _embed_raw(self, batch: np.ndarray) -> np.ndarray:
        """Run the model on a (B, 3, 224, 224) batch → (B, 384) raw (un-
        normalized) embeddings, as numpy. Backend-specific."""
        if self.backend == "onnx":
            out = self._session.run(None, {self._input_name: batch.astype(np.float32)})[0]
            return np.asarray(out, dtype=np.float32)
        torch = self._torch
        with torch.no_grad():
            t = torch.from_numpy(batch)
            if self.device != "cpu":
                t = t.to(self.device)
            out = self.model(t)
            return out.detach().cpu().numpy()

    @staticmethod
    def _normalize(raw: np.ndarray) -> Tuple[List[float], float]:
        """(384,) raw → (unit-normalized list, confidence). Confidence is the
        L2 norm before normalization (stronger feature activation = higher)."""
        confidence = float(np.linalg.norm(raw))
        emb = raw / (confidence if confidence else 1.0)
        return emb.astype(np.float32).tolist(), confidence

    def generate_embedding(self, image_data: bytes) -> Tuple[List[float], float]:
        """Embedding for a single image → (unit vector as list, confidence)."""
        raw = self._embed_raw(self._preprocess(image_data))[0]
        return self._normalize(raw)

    def generate_embeddings_batch(
        self, image_data_list: List[bytes]
    ) -> List[Tuple[List[float], float]]:
        """Embeddings for many images in one forward pass."""
        if not image_data_list:
            return []
        batch = np.concatenate([self._preprocess(b) for b in image_data_list], axis=0)
        raws = self._embed_raw(batch)
        return [self._normalize(raw) for raw in raws]

    async def generate(self, file_path: str) -> List[float]:
        """Async wrapper used by the job-queue worker: read the file and return
        its embedding vector. Heavy work runs in a thread so the event loop is
        not blocked."""
        import asyncio

        def _run() -> List[float]:
            with open(file_path, "rb") as f:
                image_data = f.read()
            embedding, _confidence = self.generate_embedding(image_data)
            return embedding

        return await asyncio.to_thread(_run)

    def get_model_info(self) -> Dict[str, object]:
        """Metadata about the embedding model + active backend."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "backend": self.backend,
        }
