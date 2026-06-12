"""Regression tests for the similarity-slider performance path.

Covers the mechanisms added to make the threshold slider fast — the per-index
connected-components memo, page-only group building, and vectorized cosine —
which the endpoint behavior tests exercise only incidentally.
"""
import numpy as np
import pytest

import app.main as m


@pytest.fixture(autouse=True)
def _restore_cache():
    """Each test sets the in-memory index directly; reset it afterwards so we
    never leak a hand-built cache (or its components memo) into other tests."""
    saved = dict(m._sim_cache)
    yield
    m._sim_cache.clear()
    m._sim_cache.update(saved)


def _make_cache(n_pairs):
    """Build an index of n_pairs disjoint duplicate pairs: photos (2k, 2k+1)
    share direction k, so at any threshold <= 1.0 there are exactly n_pairs
    groups of size 2."""
    n = 2 * n_pairs
    vectors = np.zeros((n, n_pairs), dtype=np.float32)
    photo_ids, scores, ii, jj = [], [], [], []
    meta = {}
    for k in range(n_pairs):
        a, b = 2 * k, 2 * k + 1
        vectors[a, k] = 1.0
        vectors[b, k] = 1.0
        scores.append(1.0); ii.append(a); jj.append(b)
        for idx in (a, b):
            pid = idx + 1
            photo_ids.append(pid)
            meta[pid] = {
                "filename": f"p{pid}.jpg", "file_path": f"/x/p{pid}.jpg",
                "file_size": 1000 + idx, "mime_type": "image/jpeg",
                "uploaded_at": "2024-01-01T00:00:00",
            }
    order = np.argsort(np.array(scores, dtype=np.float64))
    edge_arrays = (
        np.array(scores, dtype=np.float64)[order],
        np.array(ii, dtype=np.int32)[order],
        np.array(jj, dtype=np.int32)[order],
    )
    max_eff = max((m._effective_size(v) for v in meta.values()), default=1.0)
    cache_data = {
        "vectors": vectors, "photo_ids": photo_ids,
        "point_ids": [str(p) for p in photo_ids],
        "adjacency": [], "edge_arrays": edge_arrays,
        "cache_threshold": 0.7, "max_effective_size": max_eff,
    }
    return cache_data, meta


# --------------------------------------------------------------------------- #
# Components memo
# --------------------------------------------------------------------------- #
def test_components_memo_serves_repeat_from_cache():
    cache, _ = _make_cache(5)
    c1 = m._threshold_components_cached(cache, 0.9)
    c2 = m._threshold_components_cached(cache, 0.9)
    assert c1 is c2          # identical object → served from the memo, not recomputed
    assert len(c1) == 5


def test_components_memo_is_per_index_object_never_stale():
    """The memo lives ON cache_data, so a replaced index (scan/delete builds a
    NEW object) gets a fresh memo — a stale partition can never be served. This
    is the exact bug class that broke results mid-development."""
    cache_a, _ = _make_cache(3)
    cache_b, _ = _make_cache(6)
    a = m._threshold_components_cached(cache_a, 0.9)
    b = m._threshold_components_cached(cache_b, 0.9)
    assert len(a) == 3 and len(b) == 6
    assert cache_a["__components_memo"] is not cache_b["__components_memo"]


def test_components_memo_is_lru_bounded():
    cache, _ = _make_cache(3)
    for i in range(m._COMPONENTS_MEMO_MAX + 8):
        m._threshold_components_cached(cache, 0.70 + i * 0.001)
    assert len(cache["__components_memo"]) <= m._COMPONENTS_MEMO_MAX


# --------------------------------------------------------------------------- #
# Page-only pagination (the hot path: no sort / no quality filter)
# --------------------------------------------------------------------------- #
def test_page_only_pagination_is_complete_and_consistent():
    cache, meta = _make_cache(25)
    m._sim_cache["data"], m._sim_cache["meta"] = cache, meta

    full = m._list_groups_sync(0.9, 0, 1000, None, None)
    ids_all = [g["group_id"] for g in full["groups"]]
    assert full["total"] == 25 and len(ids_all) == 25

    collected = []
    for skip in range(0, full["total"], 10):
        page = m._list_groups_sync(0.9, skip, 10, None, None)
        assert page["total"] == 25            # total is stable across pages
        collected.extend(g["group_id"] for g in page["groups"])

    assert collected == ids_all               # same order, every group, in order
    assert len(set(collected)) == 25          # no duplicates, no omissions


def test_page_only_past_end_is_empty_but_total_correct():
    cache, meta = _make_cache(4)
    m._sim_cache["data"], m._sim_cache["meta"] = cache, meta
    res = m._list_groups_sync(0.9, 1000, 10, None, None)
    assert res["total"] == 4 and res["groups"] == []


def test_sort_by_quality_still_orders_all_groups():
    """The needs-all path (sort/min_quality) must still consider every group,
    not just a page — guards against the page-only optimization leaking in."""
    cache, meta = _make_cache(8)
    m._sim_cache["data"], m._sim_cache["meta"] = cache, meta
    res = m._list_groups_sync(0.9, 0, 100, None, "quality")
    qs = [g["quality_score"] for g in res["groups"]]
    assert res["total"] == 8
    assert qs == sorted(qs, reverse=True)     # globally sorted, descending


# --------------------------------------------------------------------------- #
# Vectorized cosine == scalar dot
# --------------------------------------------------------------------------- #
def test_vectorized_member_scores_match_scalar_dot():
    # A 3-photo cluster with known, distinct unit vectors.
    vectors = np.array(
        [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.6, 0.0, 0.8]], dtype=np.float32
    )
    photo_ids = [10, 11, 12]
    meta = {
        10: {"filename": "a.jpg", "file_path": "/a.jpg", "file_size": 9000, "mime_type": "image/jpeg"},
        11: {"filename": "b.jpg", "file_path": "/b.jpg", "file_size": 2000, "mime_type": "image/jpeg"},
        12: {"filename": "c.jpg", "file_path": "/c.jpg", "file_size": 1000, "mime_type": "image/jpeg"},
    }
    max_eff = max(m._effective_size(v) for v in meta.values())
    g = m._build_group_for_component([0, 1, 2], vectors, photo_ids, meta, max_eff)

    ref_idx = photo_ids.index(g["reference_photo"]["photo_id"])
    assert g["reference_photo"]["similarity_score"] == 1.0
    for member in g["similar_photos"]:
        midx = photo_ids.index(member["photo_id"])
        expected = float(np.dot(vectors[ref_idx], vectors[midx]))
        assert member["similarity_score"] == pytest.approx(expected, abs=1e-6)
