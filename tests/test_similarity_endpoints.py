"""Tests for the similarity-group API endpoints against the CURRENT
architecture.

The list endpoint (`GET /similarity-groups`) builds groups on the fly from the
sparse Qdrant similarity cache — it does NOT read the in-memory
similarity_group_service (which the running app never populates). The detail
endpoint (`GET /similarity-groups/{id}`) falls back to the same cache-built
groups. These tests install a hand-crafted cache and exercise pagination,
filtering, sorting, and detail lookup through it.

(The previous version of this file tested a long-removed design where the list
endpoint served the in-memory store; it asserted behavior the endpoint no
longer has.)
"""
import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from tests.test_similarity_matrix import _install_cache


# Two duplicate pairs, weakly cross-correlated, plus distinct file sizes so the
# quality score (file-size normalized across the collection) varies per group.
#   p1+p2 cluster at ~0.97 ; p3+p4 cluster at ~0.91 ; cross-pair ~0.10
_SIM_MATRIX = [
    [1.00, 0.97, 0.10, 0.10],
    [0.97, 1.00, 0.10, 0.10],
    [0.10, 0.10, 1.00, 0.91],
    [0.10, 0.10, 0.91, 1.00],
]
_PHOTO_IDS = [1, 2, 3, 4]
_META = {
    1: {"filename": "p1.jpg", "file_path": "/photos/p1.jpg",
        "file_size": 10_000_000, "mime_type": "image/jpeg",   # collection max
        "uploaded_at": "2024-01-01T00:00:00"},
    2: {"filename": "p2.jpg", "file_path": "/photos/p2.jpg",
        "file_size": 1_000_000, "mime_type": "image/jpeg",
        "uploaded_at": "2024-01-02T00:00:00"},
    3: {"filename": "p3.jpg", "file_path": "/photos/p3.jpg",
        "file_size": 2_000_000, "mime_type": "image/jpeg",
        "uploaded_at": "2024-01-03T00:00:00"},
    4: {"filename": "p4.jpg", "file_path": "/photos/p4.jpg",
        "file_size": 2_000_000, "mime_type": "image/jpeg",
        "uploaded_at": "2024-01-04T00:00:00"},
}


@pytest.fixture(autouse=True)
def _install_cache_fixture():
    app_main._sim_cache.update(data=None, meta=None)
    _install_cache(_SIM_MATRIX, _PHOTO_IDS, _META)
    yield
    app_main._sim_cache.update(data=None, meta=None)


@pytest.fixture
def client():
    return TestClient(app_main.app)


class TestListSimilarityGroups:
    def test_list_returns_both_groups(self, client):
        data = client.get("/similarity-groups?min_similarity=0.85").json()
        assert data["total"] == 2
        ids = {g["group_id"] for g in data["groups"]}
        assert ids == {"grp-1", "grp-3"}  # ref = largest photo of each pair

    def test_reference_is_largest_photo(self, client):
        data = client.get("/similarity-groups?min_similarity=0.85").json()
        by_id = {g["group_id"]: g for g in data["groups"]}
        assert by_id["grp-1"]["reference_photo"]["photo_id"] == 1

    def test_min_similarity_splits_weaker_cluster(self, client):
        # At 0.95 the p3~p4 edge (0.91) drops out, leaving only the p1~p2 group.
        data = client.get("/similarity-groups?min_similarity=0.95").json()
        assert data["total"] == 1
        assert data["groups"][0]["group_id"] == "grp-1"

    def test_min_quality_filters_on_real_score(self, client):
        # grp-1 quality = 1.0 (contains the largest photo); grp-3 ~0.2.
        data = client.get("/similarity-groups?min_similarity=0.85&min_quality=0.5").json()
        assert data["total"] == 1
        assert data["groups"][0]["group_id"] == "grp-1"

    def test_sort_by_quality(self, client):
        groups = client.get(
            "/similarity-groups?min_similarity=0.85&sort_by=quality"
        ).json()["groups"]
        scores = [g["quality_score"] for g in groups]
        assert scores == sorted(scores, reverse=True)
        assert groups[0]["group_id"] == "grp-1"  # highest quality first

    def test_sort_by_similarity(self, client):
        groups = client.get(
            "/similarity-groups?min_similarity=0.85&sort_by=similarity"
        ).json()["groups"]
        scores = [g["similarity_score"] for g in groups]
        assert scores == sorted(scores, reverse=True)

    def test_invalid_sort_by_returns_400(self, client):
        resp = client.get("/similarity-groups?sort_by=invalid")
        assert resp.status_code == 400
        assert "Invalid sort_by" in resp.text

    def test_pagination_skip_and_limit(self, client):
        data = client.get("/similarity-groups?min_similarity=0.85&skip=1&limit=1").json()
        assert data["total"] == 2
        assert data["skip"] == 1 and data["limit"] == 1
        assert len(data["groups"]) == 1

    def test_pagination_beyond_results_is_empty(self, client):
        data = client.get("/similarity-groups?min_similarity=0.85&skip=100").json()
        assert data["total"] == 2
        assert data["groups"] == []

    def test_high_threshold_returns_nothing(self, client):
        data = client.get("/similarity-groups?min_similarity=0.999").json()
        assert data["total"] == 0
        assert data["groups"] == []


class TestGetSimilarityGroupDetail:
    def test_detail_served_from_cache(self, client):
        resp = client.get("/similarity-groups/grp-1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["group_id"] == "grp-1"
        assert body["reference_photo"]["photo_id"] == 1
        member_ids = {body["reference_photo"]["photo_id"]} | {
            p["photo_id"] for p in body["similar_photos"]
        }
        assert member_ids == {1, 2}

    def test_unknown_group_returns_404(self, client):
        resp = client.get("/similarity-groups/grp-does-not-exist")
        assert resp.status_code == 404
