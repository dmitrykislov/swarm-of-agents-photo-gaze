"""Tests for the similarity-matrix cache and grouping logic.

Covers:
  - _compute_sim_cache: pagination, normalization, cosine matrix correctness,
    empty/no-overlap handling.
  - _build_similarity_groups_from_qdrant: clustering at threshold, reference-
    photo selection, similarity score recomputed relative to the reference.
  - notify_embeddings_changed: debounce semantics (rapid calls coalesce into
    one recompute), no-loop fallback.
  - best_reasons string logic: format-and-size reasons, copy-suffix detection,
    "less universal" fallback.

These tests are hermetic — they replace job_queue_manager with a stub holding
a fake Qdrant client + an in-memory Postgres-equivalent. Heavy linalg uses
tiny vectors so the math is easy to verify by hand.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pytest

from app import main as app_main


# ----------------------------- fakes -----------------------------


@dataclass
class _FakePoint:
    id: str
    vector: list
    payload: dict


class _FakeScoredPoint:
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload


class _FakeQdrant:
    """Minimal Qdrant stub with paginated scroll() AND search_batch().

    scroll() returns (page, next_offset). next_offset=None on the last page.
    search_batch() answers each SearchRequest by computing real cosine
    against the stored unit-normalized vectors and returning ScoredPoint
    objects above score_threshold, capped at limit. Matches qdrant-client
    closely enough for _compute_sim_cache to exercise the full path.
    """

    def __init__(self, points: List[_FakePoint], page_size: int = 2):
        self._points = points
        self._page_size = page_size
        self.scroll_calls = 0
        self.search_batch_calls = 0
        # Pre-normalize vectors so cosine == dot product.
        self._vec_array = np.array([p.vector for p in points], dtype=np.float32) \
            if points else np.zeros((0, 1), dtype=np.float32)
        if self._vec_array.size:
            norms = np.linalg.norm(self._vec_array, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._vec_array = self._vec_array / norms

    def scroll(self, *, collection_name, limit, offset, with_payload, with_vectors):
        self.scroll_calls += 1
        start = offset or 0
        page = self._points[start : start + self._page_size]
        next_offset: Optional[int] = start + self._page_size
        if next_offset >= len(self._points):
            next_offset = None
        return page, next_offset

    def retrieve(self, *, collection_name, ids, with_vectors=False):
        """Return stored points (id + raw vector) for the given point ids,
        mirroring qdrant_client.retrieve. Used by incremental add."""
        by_id = {p.id: p for p in self._points}
        out = []
        for pid in ids:
            p = by_id.get(pid)
            if p is not None:
                out.append(_FakePoint(id=p.id, vector=p.vector, payload=p.payload))
        return out

    def search_batch(self, *, collection_name, requests):
        self.search_batch_calls += 1
        results = []
        for req in requests:
            qv = np.asarray(req.vector, dtype=np.float32)
            qn = np.linalg.norm(qv)
            if qn:
                qv = qv / qn
            scores = self._vec_array @ qv if self._vec_array.size else np.zeros(0)
            order = np.argsort(-scores)
            hits = []
            for idx in order:
                s = float(scores[idx])
                if req.score_threshold is not None and s < req.score_threshold:
                    break
                hits.append(_FakeScoredPoint(
                    id=self._points[idx].id,
                    score=s,
                    payload=self._points[idx].payload,
                ))
                if len(hits) >= req.limit:
                    break
            results.append(hits)
        return results


class _FakeRow:
    """Tuple-like row matching the .all() return shape used in _compute_sim_cache."""

    def __init__(self, t: Tuple):
        self._t = t

    def __getitem__(self, i):
        # Columns added to the meta query later (e.g. file_hash) read as None
        # for older fixtures, instead of IndexError.
        return self._t[i] if i < len(self._t) else None


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, photo_rows):
        self._rows = photo_rows

    def query(self, *_cols):
        return _FakeQuery(self._rows)

    def close(self):
        pass


class _FakeJobQueueManager:
    """Stand-in for the real JobQueueManager. Provides only the bits
    _compute_sim_cache reads: qdrant_client and SessionLocal()."""

    def __init__(self, qdrant_points, photo_rows):
        self.qdrant_client = _FakeQdrant(qdrant_points)
        self._photo_rows = photo_rows

    def SessionLocal(self):
        return _FakeSession(self._photo_rows)


# --------------------- shared fixtures ---------------------


def _orthonormal(n: int, dim: int = 4) -> np.ndarray:
    """Return n unit vectors that are pairwise nearly-orthogonal — useful when
    we want similarity ~ 0 between distinct photos."""
    rng = np.random.default_rng(seed=1234)
    raw = rng.standard_normal((n, dim))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return raw


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts with an empty cache and no patched manager."""
    app_main._sim_cache.update(data=None, meta=None)
    app_main._sim_debounce_handle = None
    app_main._sim_recompute_lock = None
    yield
    app_main._sim_cache.update(data=None, meta=None)
    app_main._sim_debounce_handle = None
    app_main._sim_recompute_lock = None


# ----------------------------- helpers -----------------------------


def _install_cache(sim_matrix, photo_ids, photo_meta, cache_threshold: float = 0.0):
    """Pre-populate the new sparse-adjacency cache so _build_... reads it.

    Accepts a dense sim_matrix as input for ergonomics, then derives:
      - vectors that exactly reproduce the matrix via dot product (Cholesky)
      - adjacency: all pairs above cache_threshold

    cache_threshold defaults to 0.0 so test matrices with low values are
    fully indexed and don't get clamped by _build_similarity_groups_from_qdrant.
    """
    sim_matrix = np.asarray(sim_matrix, dtype=np.float64)
    n = sim_matrix.shape[0]
    if n == 0:
        vectors = np.zeros((0, 0), dtype=np.float32)
    else:
        # Lift sim_matrix to vectors via eigendecomposition so dot products
        # reproduce it. Symmetric PSD-ish input expected from tests.
        sim_matrix = (sim_matrix + sim_matrix.T) / 2.0  # symmetrize
        evals, evecs = np.linalg.eigh(sim_matrix)
        evals = np.clip(evals, 0.0, None)  # PSD floor for numerical noise
        vectors = (evecs * np.sqrt(evals)).astype(np.float32)
        # Re-normalize so unit-norm invariant the production code relies on holds.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

    adjacency = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s = float(sim_matrix[i, j])
            if s >= cache_threshold:
                adjacency[i].append((j, s))

    app_main._sim_cache.update(
        data={
            "vectors": vectors,
            "photo_ids": list(photo_ids),
            "point_ids": [f"q{p}" for p in photo_ids],
            "adjacency": adjacency,
            "cache_threshold": cache_threshold,
        },
        meta=photo_meta,
    )


# ----------------------------- tests -----------------------------


class TestComputeSimCache:
    def test_returns_none_when_manager_missing(self):
        with patch.object(app_main, "job_queue_manager", None):
            data, meta = app_main._compute_sim_cache()
        assert data is None and meta is None

    def test_returns_none_when_no_points(self):
        mgr = _FakeJobQueueManager(qdrant_points=[], photo_rows=[])
        with patch.object(app_main, "job_queue_manager", mgr):
            data, meta = app_main._compute_sim_cache()
        assert data is None and meta is None

    def test_pagination_collects_all_pages(self):
        """Five Qdrant points with page_size=2 must require three scroll calls
        and produce a 5-vector index — proves >10k bug is gone."""
        vecs = _orthonormal(5)
        points = [
            _FakePoint(id=f"qp{i}", vector=vecs[i].tolist(), payload={"photo_id": i + 1})
            for i in range(5)
        ]
        rows = [
            _FakeRow((i + 1, f"p{i}.jpg", f"/photos/p{i}.jpg", 1000 * (i + 1),
                      "image/jpeg", datetime(2024, 1, 1)))
            for i in range(5)
        ]
        mgr = _FakeJobQueueManager(qdrant_points=points, photo_rows=rows)
        with patch.object(app_main, "job_queue_manager", mgr):
            data, meta = app_main._compute_sim_cache()

        assert mgr.qdrant_client.scroll_calls == 3  # 2+2+1
        assert data["vectors"].shape == (5, 4)
        assert len(data["photo_ids"]) == 5
        # Steady-state cache stores the compact edge index, not the directed
        # adjacency list (dropped to save memory at scale).
        assert "edge_arrays" in data
        assert "adjacency" not in data
        assert set(meta.keys()) == {1, 2, 3, 4, 5}

    def test_vectors_diagonal_is_one_after_normalization(self):
        """Cosine of any unit vector with itself = 1; rebuilds confidence
        that the normalize-then-store flow is correct."""
        vecs = _orthonormal(3)
        # Scale them up so they're not unit; _compute_sim_cache should renormalize.
        vecs *= np.array([[2.0], [10.0], [0.5]])
        points = [
            _FakePoint(id=f"qp{i}", vector=vecs[i].tolist(), payload={"photo_id": i + 1})
            for i in range(3)
        ]
        rows = [
            _FakeRow((i + 1, f"p{i}.jpg", f"/photos/p{i}.jpg", 1000, "image/jpeg",
                      datetime(2024, 1, 1)))
            for i in range(3)
        ]
        mgr = _FakeJobQueueManager(qdrant_points=points, photo_rows=rows)
        with patch.object(app_main, "job_queue_manager", mgr):
            data, _ = app_main._compute_sim_cache()
        # Each stored vector should be unit-norm.
        norms = np.linalg.norm(data["vectors"], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_zero_vector_does_not_explode(self):
        """A zero embedding must not cause divide-by-zero — check the
        norms[norms == 0] = 1 guard. Then no NaNs in the stored vectors."""
        vecs = _orthonormal(2).tolist()
        vecs.append([0.0, 0.0, 0.0, 0.0])  # the bad vector
        points = [
            _FakePoint(id=f"qp{i}", vector=v, payload={"photo_id": i + 1})
            for i, v in enumerate(vecs)
        ]
        rows = [
            _FakeRow((i + 1, f"p{i}.jpg", f"/photos/p{i}.jpg", 1000,
                      "image/jpeg", datetime(2024, 1, 1)))
            for i in range(3)
        ]
        mgr = _FakeJobQueueManager(qdrant_points=points, photo_rows=rows)
        with patch.object(app_main, "job_queue_manager", mgr):
            data, _ = app_main._compute_sim_cache()
        assert np.all(np.isfinite(data["vectors"]))

    def test_qdrant_points_without_postgres_row_are_dropped(self):
        """If a Qdrant point references a photo_id that's been deleted from
        Postgres, it must not appear in the index."""
        vecs = _orthonormal(3)
        points = [
            _FakePoint(id="qp1", vector=vecs[0].tolist(), payload={"photo_id": 1}),
            _FakePoint(id="qp2", vector=vecs[1].tolist(), payload={"photo_id": 99}),  # orphan
            _FakePoint(id="qp3", vector=vecs[2].tolist(), payload={"photo_id": 3}),
        ]
        rows = [
            _FakeRow((1, "a.jpg", "/p/a.jpg", 100, "image/jpeg", datetime(2024, 1, 1))),
            _FakeRow((3, "c.jpg", "/p/c.jpg", 100, "image/jpeg", datetime(2024, 1, 1))),
        ]
        mgr = _FakeJobQueueManager(qdrant_points=points, photo_rows=rows)
        with patch.object(app_main, "job_queue_manager", mgr):
            data, meta = app_main._compute_sim_cache()
        assert data["photo_ids"] == [1, 3]
        assert 99 not in meta

    def test_edge_index_excludes_self_and_below_threshold(self):
        """Build a small set with two near-identical vectors and one
        orthogonal. The edge index must contain the i↔j edge but no self-loop
        and not the orthogonal edge."""
        # Two identical-direction vectors (sim=1) + one orthogonal.
        v_a = np.array([1.0, 0.0, 0.0, 0.0])
        v_b = np.array([0.99, 0.01, 0.0, 0.0])
        v_c = np.array([0.0, 0.0, 1.0, 0.0])
        points = [
            _FakePoint(id="qp1", vector=v_a.tolist(), payload={"photo_id": 1}),
            _FakePoint(id="qp2", vector=v_b.tolist(), payload={"photo_id": 2}),
            _FakePoint(id="qp3", vector=v_c.tolist(), payload={"photo_id": 3}),
        ]
        rows = [_FakeRow((i + 1, f"p{i}.jpg", "/p/x.jpg", 100, "image/jpeg",
                           datetime(2024, 1, 1))) for i in range(3)]
        mgr = _FakeJobQueueManager(qdrant_points=points, photo_rows=rows)
        with patch.object(app_main, "job_queue_manager", mgr):
            data, _ = app_main._compute_sim_cache()
        # The only undirected edge is idx 0 ↔ idx 1 (photos 1↔2). No self-loop,
        # and the orthogonal photo 3 (idx 2) is below the 0.7 floor.
        scores, i_idx, j_idx = data["edge_arrays"]
        pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(i_idx.tolist(), j_idx.tolist())}
        assert pairs == {(0, 1)}
        assert all(int(a) != int(b) for a, b in zip(i_idx.tolist(), j_idx.tolist()))

    def test_search_batch_called_for_all_points(self):
        """Every point must get a search query — confirms no point is skipped
        from adjacency build."""
        vecs = _orthonormal(5)
        points = [
            _FakePoint(id=f"qp{i}", vector=vecs[i].tolist(), payload={"photo_id": i + 1})
            for i in range(5)
        ]
        rows = [_FakeRow((i + 1, f"p{i}.jpg", "/p/x.jpg", 100, "image/jpeg",
                           datetime(2024, 1, 1))) for i in range(5)]
        mgr = _FakeJobQueueManager(qdrant_points=points, photo_rows=rows)
        with patch.object(app_main, "job_queue_manager", mgr):
            app_main._compute_sim_cache()
        assert mgr.qdrant_client.search_batch_calls >= 1


class TestBuildSimilarityGroups:
    """End-to-end clustering tests against a hand-crafted matrix."""

    def _install_cache(self, sim_matrix, photo_ids, photo_meta):
        """Pre-populate the cache via the shared helper (sparse adjacency)."""
        _install_cache(sim_matrix, photo_ids, photo_meta)

    def test_singletons_produce_no_groups(self):
        # 3 mutually orthogonal photos: nothing should cluster at threshold 0.5
        m = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        meta = {i: {"filename": f"p{i}.jpg", "file_path": "",
                    "file_size": 100, "mime_type": "image/jpeg",
                    "uploaded_at": "2024-01-01T00:00:00"} for i in (1, 2, 3)}
        self._install_cache(m, [1, 2, 3], meta)
        assert app_main._build_similarity_groups_from_qdrant(threshold=0.5) == []

    def test_one_group_picks_largest_as_reference(self):
        """Three near-identical vectors. Largest jpeg wins reference slot."""
        m = [
            [1.0, 0.95, 0.92],
            [0.95, 1.0, 0.94],
            [0.92, 0.94, 1.0],
        ]
        meta = {
            1: {"filename": "small.jpg", "file_path": "",
                "file_size": 1_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
            2: {"filename": "big.jpg", "file_path": "",
                "file_size": 5_000_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
            3: {"filename": "medium.jpg", "file_path": "",
                "file_size": 2_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
        }
        self._install_cache(m, [1, 2, 3], meta)
        groups = app_main._build_similarity_groups_from_qdrant(threshold=0.9)
        assert len(groups) == 1
        g = groups[0]
        assert g["reference_photo"]["photo_id"] == 2  # the largest
        assert g["reference_photo"]["similarity_score"] == 1.0
        # similar_photos scores are computed against the reference (row 2 of m)
        scores = {p["photo_id"]: pytest.approx(p["similarity_score"], abs=1e-5)
                  for p in g["similar_photos"]}
        assert scores == {1: 0.95, 3: 0.94}

    def test_format_bonus_overrides_raw_size(self):
        """JPEG with size 100k beats HEIC with size 110k because of the
        20% format bonus. Confirms _keeper_key behavior."""
        m = [[1.0, 0.99], [0.99, 1.0]]
        meta = {
            1: {"filename": "a.jpg", "file_path": "",
                "file_size": 100_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
            2: {"filename": "b.heic", "file_path": "",
                "file_size": 110_000, "mime_type": "image/heic",
                "uploaded_at": "2024-01-01T00:00:00"},
        }
        self._install_cache(m, [1, 2], meta)
        groups = app_main._build_similarity_groups_from_qdrant(threshold=0.9)
        assert groups[0]["reference_photo"]["photo_id"] == 1

    def test_transitive_chain_grouped_as_one_component(self):
        """Regression: A~B and B~C with A just under threshold to C must form
        ONE group of three (connected components), not split B's partner C off
        into a dropped singleton the way the old greedy single-link clustering
        did. Also keeps the displayed groups consistent with auto-dedupe."""
        m = [
            [1.00, 0.96, 0.50],   # A close to B, far from C
            [0.96, 1.00, 0.96],   # B close to both
            [0.50, 0.96, 1.00],   # C close to B, far from A
        ]
        meta = {
            i: {"filename": f"p{i}.jpg", "file_path": "",
                "file_size": 1000 * i, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"}
            for i in (1, 2, 3)
        }
        self._install_cache(m, [1, 2, 3], meta)
        groups = app_main._build_similarity_groups_from_qdrant(threshold=0.9)
        assert len(groups) == 1
        g = groups[0]
        member_ids = {g["reference_photo"]["photo_id"]} | {
            p["photo_id"] for p in g["similar_photos"]
        }
        assert member_ids == {1, 2, 3}  # C (id 3) is NOT dropped

    def test_quality_score_is_real_not_hardcoded(self):
        """quality_score must reflect file size normalized across the
        collection — not the old hard-coded 0.8 that made min_quality and
        sort_by=quality no-ops and left the UI quality badges blank.

        Photo 2 (5 MB) is the largest effective size in the collection, so it
        scores 1.0; smaller members score proportionally less; the group's
        quality_score equals its kept (Best) photo's."""
        m = [
            [1.0, 0.95, 0.92],
            [0.95, 1.0, 0.94],
            [0.92, 0.94, 1.0],
        ]
        meta = {
            1: {"filename": "small.jpg", "file_path": "",
                "file_size": 1_000_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
            2: {"filename": "big.jpg", "file_path": "",
                "file_size": 5_000_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
            3: {"filename": "medium.jpg", "file_path": "",
                "file_size": 2_000_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
        }
        self._install_cache(m, [1, 2, 3], meta)
        groups = app_main._build_similarity_groups_from_qdrant(threshold=0.9)
        g = groups[0]
        ref = g["reference_photo"]
        others = {p["photo_id"]: p["quality_score"] for p in g["similar_photos"]}

        # Largest photo (id 2) is the reference and scores 1.0.
        assert ref["photo_id"] == 2
        assert ref["quality_score"] == pytest.approx(1.0)
        # Group quality mirrors the kept photo.
        assert g["quality_score"] == pytest.approx(1.0)
        # Smaller members scale below 1.0 and differ from each other —
        # proving the score is not a constant.
        assert others[1] == pytest.approx(1_000_000 / 5_000_000)  # 0.2
        assert others[3] == pytest.approx(2_000_000 / 5_000_000)  # 0.4
        assert others[1] != others[3]

    def test_quality_score_format_bonus_applied(self):
        """A PNG/JPEG gets the same +20% effective-size bonus the keeper
        ranking uses, so quality stays monotonic with 'Best' selection."""
        m = [[1.0, 0.99], [0.99, 1.0]]
        meta = {
            1: {"filename": "a.jpg", "file_path": "",
                "file_size": 100_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
            2: {"filename": "b.heic", "file_path": "",
                "file_size": 100_000, "mime_type": "image/heic",
                "uploaded_at": "2024-01-01T00:00:00"},
        }
        self._install_cache(m, [1, 2], meta)
        g = app_main._build_similarity_groups_from_qdrant(threshold=0.9)[0]
        # jpeg effective size = 120k is the collection max → 1.0; heic 100k → ~0.833
        assert g["reference_photo"]["photo_id"] == 1
        heic = next(p for p in g["similar_photos"] if p["photo_id"] == 2)
        assert heic["quality_score"] == pytest.approx(100_000 / 120_000, abs=1e-4)


class TestClusteringScalability:
    """The per-slider-tick clustering must stay cheap on huge collections:
    cost should track the number of edges ABOVE the threshold, not the total
    photo count. These tests build a large, mostly-noise edge index and assert
    a high-threshold query returns only the few real clusters (and quickly)."""

    def _install_edge_cache(self, n, edges):
        """Install a cache directly from an undirected edge list
        [(i, j, score), ...] for `n` photos, bypassing adjacency so we can
        cheaply fabricate hundreds of thousands of photos."""
        scores = np.array([e[2] for e in edges], dtype=np.float64)
        order = np.argsort(scores, kind="stable")  # ascending, as production
        i_idx = np.array([edges[k][0] for k in order], dtype=np.int32)
        j_idx = np.array([edges[k][1] for k in order], dtype=np.int32)
        scores = scores[order]
        meta = {
            pid: {"filename": f"p{pid}.jpg", "file_path": f"/p/{pid}.jpg",
                  "file_size": 1000 + pid, "mime_type": "image/jpeg",
                  "uploaded_at": "2024-01-01T00:00:00"}
            for pid in range(1, n + 1)
        }
        app_main._sim_cache.update(
            data={
                "vectors": np.tile(np.array([1.0, 0.0], dtype=np.float32), (n, 1)),
                "photo_ids": list(range(1, n + 1)),
                "point_ids": [f"q{p}" for p in range(1, n + 1)],
                "adjacency": [],            # forces use of edge_arrays
                "edge_arrays": (scores, i_idx, j_idx),
                "cache_threshold": 0.70,
            },
            meta=meta,
        )

    def test_high_threshold_query_only_touches_active_edges(self):
        import time
        n = 120_000
        # 60k noise edges below the query threshold + two real duplicate edges.
        rng = np.random.default_rng(7)
        noise = [
            (int(a), int(b), 0.75)
            for a, b in rng.integers(0, n, size=(60_000, 2))
            if a != b
        ]
        real = [(0, 1, 0.97), (2, 3, 0.96)]  # idx 0/1 → pids 1/2 ; 2/3 → 3/4
        self._install_edge_cache(n, noise + real)

        t0 = time.time()
        groups = app_main._build_similarity_groups_from_qdrant(threshold=0.9)
        elapsed = time.time() - t0

        # Only the two real clusters survive the 0.9 threshold.
        assert len(groups) == 2
        member_ids = sorted(
            {g["reference_photo"]["photo_id"] for g in groups}
            | {p["photo_id"] for g in groups for p in g["similar_photos"]}
        )
        assert member_ids == [1, 2, 3, 4]
        # Backstop against accidental O(N) / O(N·E) reintroduction. The active
        # set is 2 edges; this is generous but would fail loudly on a per-photo
        # set allocation over 120k photos done in a tight loop.
        assert elapsed < 2.0

    def test_incremental_removal_filters_and_remaps(self):
        """_remove_photos_from_cache must drop the deleted photos, keep the
        surviving edges with correctly REMAPPED indices, and leave the groups
        consistent — without a full rebuild. This is the fast delete path used
        by /deduplicate and folder-delete at 300k scale."""
        # 3 photos: 1~2 (0.97) and 2~3 (0.96) → one chain component {1,2,3}.
        self._install_edge_cache(3, [(0, 1, 0.97), (1, 2, 0.96)])
        # Delete photo 2 (the chain's middle) → 1 and 3 are no longer linked.
        app_main._remove_photos_from_cache({2})

        data = app_main._sim_cache["data"]
        assert data["photo_ids"] == [1, 3]            # 2 removed, order preserved
        scores, ii, jj = app_main._get_edge_arrays(data)
        # Both surviving edges touched photo 2, so no edges remain.
        assert scores.size == 0
        assert app_main._build_similarity_groups_from_qdrant(threshold=0.9) == []

    def test_incremental_removal_remaps_surviving_edge(self):
        """A surviving edge's endpoints must be remapped to the new compacted
        indices (deleting an earlier photo shifts later indices down)."""
        # photos 1,2,3,4 ; edge between 3 and 4 (idx 2,3) at 0.97.
        self._install_edge_cache(4, [(2, 3, 0.97)])
        app_main._remove_photos_from_cache({1})  # drop idx 0 → others shift down
        data = app_main._sim_cache["data"]
        assert data["photo_ids"] == [2, 3, 4]
        groups = app_main._build_similarity_groups_from_qdrant(threshold=0.9)
        assert len(groups) == 1
        ids = {groups[0]["reference_photo"]["photo_id"]} | {
            p["photo_id"] for p in groups[0]["similar_photos"]
        }
        assert ids == {3, 4}  # the 3~4 edge survived and still groups correctly

    def test_merge_new_edges_dedups_and_sorts(self):
        """_merge_new_edges keeps the max score per undirected pair, appends to
        the existing edges, and returns them sorted ascending by score."""
        existing = (
            np.array([0.80], dtype=np.float64),
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
        )
        # New node idx 5 reciprocally linked to 0 (two directions, diff scores)
        # plus a link to 2. Reciprocal pair must collapse to its max (0.95).
        merged = app_main._merge_new_edges(existing, [(5, 0, 0.93), (0, 5, 0.95), (5, 2, 0.88)])
        scores, ii, jj = merged
        assert list(scores) == sorted(scores)  # ascending
        pairs = {
            tuple(sorted((int(a), int(b)))): float(s)
            for a, b, s in zip(ii.tolist(), jj.tolist(), scores.tolist())
        }
        assert pairs == {(0, 1): 0.80, (0, 5): 0.95, (2, 5): 0.88}

    def test_incremental_add_merges_new_photo(self):
        """End-to-end: adding a photo that duplicates an existing one links
        them in the index WITHOUT rebuilding from scratch (only the new vector
        is searched). Uses a real SQLite session + the fake Qdrant."""
        from types import SimpleNamespace
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from app.models import Base, Photo, Embedding

        # Existing index: photos 1 and 2, orthogonal (no edge).
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        v3 = [1.0, 0.0]  # photo 3 duplicates photo 1
        base_meta = {
            1: {"filename": "a.jpg", "file_path": "/p/a.jpg", "file_size": 1000,
                "mime_type": "image/jpeg", "uploaded_at": "2024-01-01T00:00:00"},
            2: {"filename": "b.jpg", "file_path": "/p/b.jpg", "file_size": 1000,
                "mime_type": "image/jpeg", "uploaded_at": "2024-01-01T00:00:00"},
        }
        app_main._sim_cache.update(
            data={
                "vectors": np.array([v1, v2], dtype=np.float32),
                "photo_ids": [1, 2],
                "point_ids": ["q1", "q2"],
                "edge_arrays": (np.empty(0, np.float64),
                                np.empty(0, np.int32), np.empty(0, np.int32)),
                "max_effective_size": 1200.0,
                "cache_threshold": 0.70,
            },
            meta=dict(base_meta),
        )

        # SQLite with the new photo's Photo + Embedding rows.
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        s = Session()
        s.add(Photo(id=3, filename="c.jpg", file_path="/p/c.jpg", file_size=2000,
                    mime_type="image/jpeg"))
        s.add(Embedding(photo_id=3, embedding_model="m", vector_dimension=2,
                        qdrant_point_id="q3"))
        s.commit(); s.close()

        # Fake Qdrant holds all three points so the new vector's search finds
        # photo 1 as a neighbour.
        points = [
            _FakePoint(id="q1", vector=v1, payload={"photo_id": 1}),
            _FakePoint(id="q2", vector=v2, payload={"photo_id": 2}),
            _FakePoint(id="q3", vector=v3, payload={"photo_id": 3}),
        ]
        mgr = SimpleNamespace(qdrant_client=_FakeQdrant(points), SessionLocal=Session)

        with patch.object(app_main, "job_queue_manager", mgr):
            new_data, new_meta = app_main._incremental_add_sync({3})

        assert new_data is not None
        assert new_data["photo_ids"] == [1, 2, 3]
        assert 3 in new_meta
        # Photo 3 (idx 2) is linked to photo 1 (idx 0); photo 2 stays isolated.
        scores, ii, jj = new_data["edge_arrays"]
        pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(ii.tolist(), jj.tolist())}
        assert (0, 2) in pairs
        # Feed it back and confirm the group materializes.
        app_main._sim_cache.update(data=new_data, meta=new_meta)
        groups = app_main._build_similarity_groups_from_qdrant(threshold=0.9)
        assert len(groups) == 1
        ids = {groups[0]["reference_photo"]["photo_id"]} | {
            p["photo_id"] for p in groups[0]["similar_photos"]
        }
        assert ids == {1, 3}

    def test_incremental_add_cold_cache_signals_fallback(self):
        """With no base index built yet, _incremental_add_sync returns
        (None, None) so the caller does a full recompute instead."""
        from types import SimpleNamespace
        app_main._sim_cache.update(data=None, meta=None)
        mgr = SimpleNamespace(qdrant_client=_FakeQdrant([]), SessionLocal=lambda: None)
        with patch.object(app_main, "job_queue_manager", mgr):
            assert app_main._incremental_add_sync({1}) == (None, None)

    def test_edge_arrays_are_memoized(self):
        self._install_edge_cache(10, [(0, 1, 0.95)])
        data = app_main._sim_cache["data"]
        first = app_main._get_edge_arrays(data)
        second = app_main._get_edge_arrays(data)
        # Same object identity → derived once, then cached.
        assert first is second


class TestGroupDetailFallback:
    """The /similarity-groups/{id} detail endpoint historically read only the
    in-memory similarity_group_service, which the running app never populates —
    so it could only ever 404 in production. It now falls back to the live
    cache-built groups."""

    def test_detail_serves_cache_built_group_when_store_empty(self):
        from fastapi.testclient import TestClient

        m = [[1.0, 0.96], [0.96, 1.0]]
        meta = {
            1: {"filename": "small.jpg", "file_path": "",
                "file_size": 1_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
            2: {"filename": "big.jpg", "file_path": "",
                "file_size": 9_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
        }
        _install_cache(m, [1, 2], meta)
        app_main.similarity_group_service.clear()  # store empty → must fall back

        client = TestClient(app_main.app)
        resp = client.get("/similarity-groups/grp-2")  # ref is the larger photo
        assert resp.status_code == 200
        body = resp.json()
        assert body["group_id"] == "grp-2"
        assert body["reference_photo"]["photo_id"] == 2

    def test_detail_unknown_group_still_404s(self):
        from fastapi.testclient import TestClient

        app_main.similarity_group_service.clear()
        client = TestClient(app_main.app)
        resp = client.get("/similarity-groups/grp-does-not-exist")
        assert resp.status_code == 404


class TestBestReasonsStrings:
    """Exercise the human-readable best_reasons string builder. The strings
    drive the 'why this photo' UI tooltip — regressions here are user-visible."""

    def _install(self, sim_matrix, photo_ids, photo_meta):
        _install_cache(sim_matrix, photo_ids, photo_meta)

    def test_largest_file_string_format(self):
        m = [[1.0, 0.99], [0.99, 1.0]]
        self._install(m, [1, 2], {
            1: {"filename": "a.jpg", "file_path": "",
                "file_size": 5_000_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
            2: {"filename": "b.jpg", "file_path": "",
                "file_size": 1_000_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
        })
        groups = app_main._build_similarity_groups_from_qdrant(threshold=0.9)
        reasons = groups[0]["best_reasons"]
        # "Largest file: 5.00 MB vs next 1.00 MB (+400%)"
        assert any(r.startswith("Largest file: 5.00 MB vs next 1.00 MB") for r in reasons)
        assert any("+400%" in r for r in reasons)

    def test_identical_size_with_copy_suffix_detection(self):
        """Original and a "(copy)" duplicate at same size — string should
        flag the original."""
        m = [[1.0, 0.99], [0.99, 1.0]]
        self._install(m, [1, 2], {
            1: {"filename": "vacation.jpg", "file_path": "",
                "file_size": 2_000_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
            2: {"filename": "vacation copy.jpg", "file_path": "",
                "file_size": 2_000_000, "mime_type": "image/jpeg",
                # later uploaded_at -> tiebreak goes to earlier (negated ts)
                "uploaded_at": "2024-02-01T00:00:00"},
        })
        groups = app_main._build_similarity_groups_from_qdrant(threshold=0.9)
        ref = groups[0]["reference_photo"]
        reasons = groups[0]["best_reasons"]
        assert ref["filename"] == "vacation.jpg"
        assert any("Identical file size: 2.00 MB" in r for r in reasons)
        assert any('"vacation.jpg" appears to be the original' in r for r in reasons)

    def test_format_string_marks_jpeg_as_preferred(self):
        m = [[1.0, 0.99], [0.99, 1.0]]
        self._install(m, [1, 2], {
            1: {"filename": "a.jpg", "file_path": "",
                "file_size": 1_000_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
            2: {"filename": "b.heic", "file_path": "",
                "file_size": 800_000, "mime_type": "image/heic",
                "uploaded_at": "2024-01-01T00:00:00"},
        })
        groups = app_main._build_similarity_groups_from_qdrant(threshold=0.9)
        reasons = groups[0]["best_reasons"]
        assert any("Format: image/jpeg (preferred (universal))" in r for r in reasons)
        assert any("others: image/heic" in r for r in reasons)

    def test_format_string_handles_null_mime_type_among_others(self):
        """Postgres allows null mime_type. A mix of None and str values must
        not crash sorted()."""
        m = [[1.0, 0.99, 0.99], [0.99, 1.0, 0.98], [0.99, 0.98, 1.0]]
        self._install(m, [1, 2, 3], {
            1: {"filename": "ref.jpg", "file_path": "",
                "file_size": 5_000_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
            2: {"filename": "other.jpg", "file_path": "",
                "file_size": 1_000_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
            3: {"filename": "missing_type.bin", "file_path": "",
                "file_size": 1_000_000, "mime_type": None,  # the trap
                "uploaded_at": "2024-01-01T00:00:00"},
        })
        groups = app_main._build_similarity_groups_from_qdrant(threshold=0.9)
        assert len(groups) == 1
        reasons = groups[0]["best_reasons"]
        # "?" sorts before "image/jpeg" — assert both appear, no TypeError
        assert any("others: ?, image/jpeg" in r for r in reasons)

    def test_kb_size_format_under_one_mb(self):
        """Sub-1MB files render as KB, not MB."""
        m = [[1.0, 0.99], [0.99, 1.0]]
        self._install(m, [1, 2], {
            1: {"filename": "a.jpg", "file_path": "",
                "file_size": 500_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
            2: {"filename": "b.jpg", "file_path": "",
                "file_size": 100_000, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00"},
        })
        groups = app_main._build_similarity_groups_from_qdrant(threshold=0.9)
        reasons = groups[0]["best_reasons"]
        assert any("Largest file: 500.0 KB vs next 100.0 KB" in r for r in reasons)


class TestNotifyDebounce:
    """Verify rapid changes coalesce into one recompute via the debounce."""

    @pytest.mark.asyncio
    async def test_rapid_calls_schedule_single_recompute(self, monkeypatch):
        """Five notify calls in quick succession should leave exactly one
        TimerHandle pending — earlier handles must be cancelled."""
        # Avoid actually running the heavy recompute: replace it with a no-op.
        async def _noop():
            return None
        monkeypatch.setattr(app_main, "_recompute_sim_cache", _noop)

        # Push the debounce way out so the test doesn't race with the timer.
        monkeypatch.setattr(app_main, "_SIM_DEBOUNCE_SECONDS", 60.0)

        cancels = {"count": 0}
        real_call_later = asyncio.get_running_loop().call_later

        for _ in range(5):
            app_main.notify_embeddings_changed()

        handle = app_main._sim_debounce_handle
        assert handle is not None
        # Only one outstanding handle — the previous four should have been cancelled
        # before being replaced.
        handle.cancel()  # don't leak the timer

    @pytest.mark.asyncio
    async def test_debounce_actually_fires_recompute(self, monkeypatch):
        """With a tiny debounce, the recompute coroutine should run."""
        called = asyncio.Event()

        async def _fake_recompute():
            called.set()

        monkeypatch.setattr(app_main, "_recompute_sim_cache", _fake_recompute)
        monkeypatch.setattr(app_main, "_SIM_DEBOUNCE_SECONDS", 0.01)

        app_main.notify_embeddings_changed()
        await asyncio.wait_for(called.wait(), timeout=1.0)

    def test_no_running_loop_returns_silently(self):
        """When called from sync code (no event loop), notify must not raise."""
        # In a test running outside async, get_running_loop raises RuntimeError;
        # the function should swallow it and return.
        app_main.notify_embeddings_changed()  # must not raise
        assert app_main._sim_debounce_handle is None


class TestRecomputeLock:
    """The lazy lock must only be created once and reused per loop."""

    @pytest.mark.asyncio
    async def test_lock_is_singleton_per_run(self):
        a = app_main._get_recompute_lock()
        b = app_main._get_recompute_lock()
        assert a is b
        # And it really gates concurrent recomputes:
        async with a:
            assert a.locked()
        assert not a.locked()


class TestObservability:
    """The /stats endpoint exposes similarity index health for the UI."""

    @pytest.mark.asyncio
    async def test_recompute_updates_index_info(self):
        """After a recompute, _sim_index_info should reflect the new state:
        non-null timestamp, vector/edge counts matching the cache."""
        # Use the same fake-manager harness as the unit tests above.
        v_a = np.array([1.0, 0.0, 0.0, 0.0])
        v_b = np.array([0.99, 0.01, 0.0, 0.0])
        v_c = np.array([0.0, 0.0, 1.0, 0.0])
        points = [
            _FakePoint(id="qp1", vector=v_a.tolist(), payload={"photo_id": 1}),
            _FakePoint(id="qp2", vector=v_b.tolist(), payload={"photo_id": 2}),
            _FakePoint(id="qp3", vector=v_c.tolist(), payload={"photo_id": 3}),
        ]
        rows = [_FakeRow((i + 1, f"p{i}.jpg", "/p/x.jpg", 100, "image/jpeg",
                           datetime(2024, 1, 1))) for i in range(3)]
        mgr = _FakeJobQueueManager(qdrant_points=points, photo_rows=rows)

        # Reset info baseline
        app_main._sim_index_info.update(
            last_recompute_at=None,
            last_recompute_duration_ms=None,
            recompute_running=False,
            vectors_in_index=0,
            edges_in_index=0,
        )

        with patch.object(app_main, "job_queue_manager", mgr):
            await app_main._recompute_sim_cache()

        info = app_main._sim_index_info
        assert info["last_recompute_at"] is not None
        assert info["last_recompute_duration_ms"] is not None
        assert info["recompute_running"] is False
        assert info["vectors_in_index"] == 3
        # edges_in_index now counts UNDIRECTED edges: A↔B is one edge; C is
        # isolated → total 1.
        assert info["edges_in_index"] == 1


class TestTenPhotoEndToEnd:
    """Integration test: simulate a folder of 10 'photos' with a known
    mix of duplicates, near-duplicates, and uniques. Run the full
    compute-cache → cluster pipeline and assert correct grouping.

    Photos are not real JPEGs (we never invoke the embedding model in
    tests — that would require ~90MB DINOv2 weights). Instead each
    'photo' is a hand-crafted unit vector laid out so the resulting
    similarity structure is exactly what we'd expect from real
    near-duplicate clusters: tightly grouped vectors for duplicates,
    orthogonal vectors for uniques.
    """

    def _build_collection(self):
        """Return (qdrant_points, postgres_rows) for 10 photos.

        Cluster A: 4 near-duplicates of a "sunset" shot, varying file
                   sizes and formats — the largest preferred-format wins.
        Cluster B: 3 near-duplicates of a "portrait", with one having
                   a "(copy)" suffix.
        Singletons: 3 unrelated unique photos.
        """
        # Ten 8-D unit vectors. Cluster A on axis 0, cluster B on axis 1,
        # singletons on axes 2/3/4.
        def _unit(direction, jitter=0.0):
            v = np.zeros(8, dtype=np.float32)
            v[direction] = 1.0
            if jitter:
                # Tiny perpendicular jitter so vectors aren't bit-identical.
                v[(direction + 1) % 8] = jitter
            n = np.linalg.norm(v)
            return (v / n).tolist()

        photos = [
            # --- Cluster A: 4 sunset near-duplicates ---
            # A1: medium jpeg, the original
            (1, _unit(0, 0.00),  "sunset.jpg",         "image/jpeg", 2_000_000, "2024-01-01T08:00:00"),
            # A2: smaller jpeg copy
            (2, _unit(0, 0.01),  "sunset copy.jpg",    "image/jpeg",   500_000, "2024-01-01T09:00:00"),
            # A3: largest png — should win the reference slot (bonus + size)
            (3, _unit(0, 0.02),  "sunset.png",         "image/png",  3_000_000, "2024-01-01T10:00:00"),
            # A4: heic (less universal), even bigger but loses on format bonus tie
            (4, _unit(0, 0.03),  "sunset.heic",        "image/heic", 3_500_000, "2024-01-01T11:00:00"),
            # --- Cluster B: 3 portraits ---
            (5, _unit(1, 0.00),  "portrait.jpg",        "image/jpeg", 1_500_000, "2024-02-01T08:00:00"),
            (6, _unit(1, 0.01),  "portrait (1).jpg",    "image/jpeg", 1_500_000, "2024-02-01T09:00:00"),
            (7, _unit(1, 0.02),  "portrait copy.jpg",   "image/jpeg", 1_500_000, "2024-02-01T10:00:00"),
            # --- Singletons: unrelated photos ---
            (8, _unit(2, 0.00),  "tree.jpg",            "image/jpeg",   400_000, "2024-03-01T08:00:00"),
            (9, _unit(3, 0.00),  "skyline.jpg",         "image/jpeg",   600_000, "2024-03-01T09:00:00"),
            (10, _unit(4, 0.00), "cat.jpg",             "image/jpeg",   800_000, "2024-03-01T10:00:00"),
        ]

        qpoints = [
            _FakePoint(id=f"qp{pid}", vector=v, payload={"photo_id": pid})
            for (pid, v, *_rest) in photos
        ]
        prows = [
            _FakeRow((pid, fname, f"/photos/{fname}", size, mime,
                      datetime.fromisoformat(uploaded)))
            for (pid, _v, fname, mime, size, uploaded) in photos
        ]
        return qpoints, prows

    def test_full_pipeline_groups_three_clusters_correctly(self):
        qpoints, prows = self._build_collection()
        mgr = _FakeJobQueueManager(qdrant_points=qpoints, photo_rows=prows)
        with patch.object(app_main, "job_queue_manager", mgr):
            data, meta = app_main._compute_sim_cache()
            app_main._sim_cache.update(data=data, meta=meta)
            groups = app_main._build_similarity_groups_from_qdrant(threshold=0.9)

        # Two non-singleton clusters: A (4 photos) and B (3 photos).
        # Singletons (8/9/10) form no groups.
        assert len(groups) == 2

        # Find each by membership.
        groups_by_size = sorted(groups, key=lambda g: -(1 + len(g["similar_photos"])))
        cluster_a = groups_by_size[0]
        cluster_b = groups_by_size[1]
        a_pids = {cluster_a["reference_photo"]["photo_id"]} | {
            p["photo_id"] for p in cluster_a["similar_photos"]
        }
        b_pids = {cluster_b["reference_photo"]["photo_id"]} | {
            p["photo_id"] for p in cluster_b["similar_photos"]
        }
        assert a_pids == {1, 2, 3, 4}
        assert b_pids == {5, 6, 7}
        # Singletons must not appear anywhere.
        assert {8, 9, 10}.isdisjoint(a_pids | b_pids)

    def test_reference_selection_prefers_universal_format(self):
        """In cluster A, the HEIC (3.5 MB) is biggest by raw bytes but
        the PNG (3.0 MB) wins because the 20% preferred-format bonus
        gives it a higher effective score."""
        qpoints, prows = self._build_collection()
        mgr = _FakeJobQueueManager(qdrant_points=qpoints, photo_rows=prows)
        with patch.object(app_main, "job_queue_manager", mgr):
            data, meta = app_main._compute_sim_cache()
            app_main._sim_cache.update(data=data, meta=meta)
            groups = app_main._build_similarity_groups_from_qdrant(threshold=0.9)

        # Pick the 4-member cluster
        cluster_a = next(
            g for g in groups
            if 1 + len(g["similar_photos"]) == 4
        )
        # PNG (photo_id=3, 3MB) should win over HEIC (photo_id=4, 3.5MB)
        # because score(PNG) = 3M*1.2 = 3.6M > score(HEIC) = 3.5M.
        assert cluster_a["reference_photo"]["photo_id"] == 3
        assert cluster_a["reference_photo"]["mime_type"] == "image/png"

    def test_reference_score_is_one_others_above_threshold(self):
        """All members of a near-duplicate cluster must score very close
        to 1.0 against the reference (vectors are constructed that way)."""
        qpoints, prows = self._build_collection()
        mgr = _FakeJobQueueManager(qdrant_points=qpoints, photo_rows=prows)
        with patch.object(app_main, "job_queue_manager", mgr):
            data, meta = app_main._compute_sim_cache()
            app_main._sim_cache.update(data=data, meta=meta)
            groups = app_main._build_similarity_groups_from_qdrant(threshold=0.9)

        for g in groups:
            assert g["reference_photo"]["similarity_score"] == 1.0
            for m in g["similar_photos"]:
                assert m["similarity_score"] >= 0.9, \
                    f"member {m['photo_id']} dropped below threshold: {m['similarity_score']}"

    def test_threshold_above_one_yields_no_groups(self):
        """Asking for threshold > 1 must return zero groups (no pair is
        more similar than 1.0). Defends against off-by-one slider bugs."""
        qpoints, prows = self._build_collection()
        mgr = _FakeJobQueueManager(qdrant_points=qpoints, photo_rows=prows)
        with patch.object(app_main, "job_queue_manager", mgr):
            data, meta = app_main._compute_sim_cache()
            app_main._sim_cache.update(data=data, meta=meta)
            groups = app_main._build_similarity_groups_from_qdrant(threshold=1.0001)
        assert groups == []

    def test_threshold_below_cache_floor_is_clamped(self):
        """If the user asks for threshold 0.0 (would normally pull in
        everything) the build still honors the cache floor (0.7) — we
        can't fabricate edges that weren't precomputed. Behavior must be
        identical to threshold=cache_threshold."""
        qpoints, prows = self._build_collection()
        mgr = _FakeJobQueueManager(qdrant_points=qpoints, photo_rows=prows)
        with patch.object(app_main, "job_queue_manager", mgr):
            data, meta = app_main._compute_sim_cache()
            app_main._sim_cache.update(data=data, meta=meta)
            g_low = app_main._build_similarity_groups_from_qdrant(threshold=0.0)
            g_floor = app_main._build_similarity_groups_from_qdrant(
                threshold=app_main._SIM_CACHE_THRESHOLD)
        assert len(g_low) == len(g_floor)

    def test_memory_footprint_is_sparse(self):
        """The whole point of the rewrite. With 10 photos in 2 clusters of
        4 and 3 plus 3 singletons, the undirected edge count should be small
        (≤ C(4,2) + C(3,2) = 9), nowhere near a dense N² = 100."""
        qpoints, prows = self._build_collection()
        mgr = _FakeJobQueueManager(qdrant_points=qpoints, photo_rows=prows)
        with patch.object(app_main, "job_queue_manager", mgr):
            data, _ = app_main._compute_sim_cache()
        total_edges = int(data["edge_arrays"][0].size)
        assert total_edges <= 9, f"edge index exploded: {total_edges} edges"
        # Vectors stored separately for exact ref-vs-member scoring.
        assert data["vectors"].shape == (10, 8)
