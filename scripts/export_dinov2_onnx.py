#!/usr/bin/env python3
"""Export DINOv2 ViT-S/14 to ONNX and verify parity against PyTorch.

Run at BUILD time (needs torch + onnx + onnxruntime; Python 3.10+). The
resulting ``models/dinov2_vits14.onnx`` is bundled into the native app, which
runs it with ONNX Runtime ONLY — no torch shipped. The model file itself is
gitignored (~88 MB); regenerate it with this script.

    python scripts/export_dinov2_onnx.py [output_path]

Exits non-zero if torch↔ONNX parity is not within tolerance, so a bad export
can't slip into a build.
"""
import os
import sys

import numpy as np
import torch


class _Wrap(torch.nn.Module):
    """DINOv2's forward is ``forward(x, masks=None)``; the optional ``masks``
    leaks into the ONNX graph as a required input. Wrapping exposes only x."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "models", "dinov2_vits14.onnx"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    torch.manual_seed(0)
    base = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()
    model = _Wrap(base).eval()
    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=14,
    )
    size_mb = round(os.path.getsize(out_path) / 1e6, 1)
    print(f"Exported {out_path} ({size_mb} MB)")

    # Parity check: torch vs ONNX Runtime on several random inputs.
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    worst_cos, worst_abs = 1.0, 0.0
    with torch.no_grad():
        for _ in range(8):
            x = torch.randn(1, 3, 224, 224)
            t = model(x).squeeze(0).numpy()
            o = sess.run(None, {"input": x.numpy()})[0].squeeze(0)
            tn, on = t / np.linalg.norm(t), o / np.linalg.norm(o)
            worst_cos = min(worst_cos, float(np.dot(tn, on)))
            worst_abs = max(worst_abs, float(np.max(np.abs(tn - on))))
    print(f"parity: worst cosine={worst_cos:.8f}  worst max|Δ|={worst_abs:.2e}")

    if worst_cos > 0.9999 and worst_abs < 1e-3:
        print("PARITY OK")
        return 0
    print("PARITY FAIL — ONNX export does not match torch", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
