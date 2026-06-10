"""Unit tests for the ONNX backend of EmbeddingGenerator.

These run WITHOUT torch or onnxruntime installed: a fake InferenceSession is
injected, so they exercise the preprocessing → inference → normalization path
(the contract the native app relies on) on any Python. The torch backend is
covered separately in test_embedding_generator.py (requires torch)."""
import io

import numpy as np
import pytest
from PIL import Image

from app.embedding_generator import EmbeddingGenerator


def _jpeg_bytes(color=(123, 50, 200), size=(40, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, "JPEG")
    return buf.getvalue()


class _FakeSession:
    """Mimics onnxruntime.InferenceSession: a single input, returns a fixed
    raw (un-normalized) 384-dim embedding per item, scaled so we can assert
    the confidence (= L2 norm before normalization)."""

    def __init__(self, fill=2.0):
        self.fill = fill
        self.last_shape = None

    def get_inputs(self):
        import types
        return [types.SimpleNamespace(name="input")]

    def run(self, _outputs, feed):
        x = feed["input"]
        self.last_shape = x.shape
        b = x.shape[0]
        return [np.full((b, 384), self.fill, dtype=np.float32)]


def test_onnx_backend_preprocess_shape_and_normalization():
    fake = _FakeSession(fill=2.0)
    g = EmbeddingGenerator(session=fake)
    assert g.backend == "onnx"
    assert g.get_model_info()["backend"] == "onnx"

    emb, conf = g.generate_embedding(_jpeg_bytes())

    # Preprocessing fed a single (1, 3, 224, 224) tensor to the session.
    assert fake.last_shape == (1, 3, 224, 224)
    # 384-dim, unit-normalized output.
    assert len(emb) == 384
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5
    # Confidence is the L2 norm of the raw (pre-normalization) vector:
    # sqrt(384 * 2^2) = 2 * sqrt(384).
    assert abs(conf - 2.0 * np.sqrt(384)) < 1e-3


def test_onnx_backend_batch():
    fake = _FakeSession(fill=1.5)
    g = EmbeddingGenerator(session=fake)
    results = g.generate_embeddings_batch([_jpeg_bytes(), _jpeg_bytes((10, 20, 30))])
    assert fake.last_shape == (2, 3, 224, 224)
    assert len(results) == 2
    for emb, conf in results:
        assert len(emb) == 384
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_onnx_backend_empty_batch_is_noop():
    g = EmbeddingGenerator(session=_FakeSession())
    assert g.generate_embeddings_batch([]) == []


def test_preprocess_matches_imagenet_normalization():
    """The numpy preprocessing must match the original torch pipeline:
    resize→/255→(x-mean)/std, channels-first."""
    g = EmbeddingGenerator(session=_FakeSession())
    arr = g._preprocess(_jpeg_bytes(color=(255, 255, 255)))
    assert arr.shape == (1, 3, 224, 224)
    assert arr.dtype == np.float32
    # A pure-white image → (1.0 - mean)/std per channel.
    expected_r = (1.0 - 0.485) / 0.229
    assert abs(float(arr[0, 0].mean()) - expected_r) < 1e-2
