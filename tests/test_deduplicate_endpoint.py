"""Tests for POST /deduplicate endpoint."""
import os
import tempfile
import shutil
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app, job_queue_manager
from app.models import Photo
from app.database import SessionLocal


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def temp_photo_dir():
    """Create a temporary directory with test photos."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_photos(temp_photo_dir):
    """Create sample photos in the database and filesystem."""
    session = SessionLocal()
    try:
        # Create test files
        photo_files = []
        for i in range(3):
            file_path = os.path.join(temp_photo_dir, f"photo_{i}.jpg")
            Path(file_path).write_text(f"fake image data {i}")
            photo_files.append(file_path)
        
        # Create Photo records
        photos = []
        for i, file_path in enumerate(photo_files):
            photo = Photo(
                filename=f"photo_{i}.jpg",
                file_path=file_path,
                file_size_bytes=100,
                modified_date=__import__('datetime').datetime.utcnow(),
                quality_score=0.9 - (i * 0.1),  # Decreasing quality
            )
            session.add(photo)
            photos.append(photo)
        
        session.commit()
        photo_ids = [p.id for p in photos]
        yield photo_ids, temp_photo_dir
    finally:
        session.close()


class TestDeduplicateEndpoint:
    """Test suite for deduplication endpoint."""
    
    def test_deduplicate_moves_files_to_trash(self, client, sample_photos):
        """Files should be moved to .trash/ folder, not deleted."""
        photo_ids, temp_dir = sample_photos
        
        # Move first two photos to trash
        response = client.post("/deduplicate", json={"photo_ids": photo_ids[:2]})
        assert response.status_code == 200
        data = response.json()
        assert data["moved_count"] == 2
        
        # Verify files were moved to .trash/
        trash_dir = os.path.join(temp_dir, ".trash")
        assert os.path.exists(trash_dir)
        assert os.path.exists(os.path.join(trash_dir, "photo_0.jpg"))
        assert os.path.exists(os.path.join(trash_dir, "photo_1.jpg"))
        
        # Original files should not exist
        assert not os.path.exists(os.path.join(temp_dir, "photo_0.jpg"))
        assert not os.path.exists(os.path.join(temp_dir, "photo_1.jpg"))
        
        # Third photo should still be in original location
        assert os.path.exists(os.path.join(temp_dir, "photo_2.jpg"))
    
    def test_deduplicate_removes_from_database(self, client, sample_photos):
        """Photos should be removed from database after deduplication."""
        photo_ids, _ = sample_photos
        session = SessionLocal()
        try:
            # Verify photos exist before deduplication
            photos_before = session.query(Photo).filter(Photo.id.in_(photo_ids[:2])).all()
            assert len(photos_before) == 2
            
            # Deduplicate
            response = client.post("/deduplicate", json={"photo_ids": photo_ids[:2]})
            assert response.status_code == 200
            
            # Verify photos are removed from database
            photos_after = session.query(Photo).filter(Photo.id.in_(photo_ids[:2])).all()
            assert len(photos_after) == 0
            
            # Third photo should still exist
            photo_3 = session.query(Photo).filter(Photo.id == photo_ids[2]).first()
            assert photo_3 is not None
        finally:
            session.close()
    
    def test_deduplicate_empty_photo_ids(self, client):
        """Empty photo_ids list should return 400 error."""
        response = client.post("/deduplicate", json={"photo_ids": []})
        assert response.status_code == 400
        assert "No photo IDs provided" in response.json()["error"]
    
    def test_deduplicate_nonexistent_photos(self, client):
        """Nonexistent photo IDs should return 404 error."""
        response = client.post("/deduplicate", json={"photo_ids": [99999]})
        assert response.status_code == 404
        assert "No photos found" in response.json()["error"]
    
    def test_deduplicate_preserves_path_structure(self, client, temp_photo_dir):
        """Deduplication should preserve relative path structure in .trash/."""
        session = SessionLocal()
        try:
            # Create nested directory structure
            nested_dir = os.path.join(temp_photo_dir, "vacation", "2024")
            os.makedirs(nested_dir, exist_ok=True)
            file_path = os.path.join(nested_dir, "IMG_001.jpg")
            Path(file_path).write_text("fake image data")
            
            # Create Photo record
            photo = Photo(
                filename="IMG_001.jpg",
                file_path=file_path,
                file_size_bytes=100,
                modified_date=__import__('datetime').datetime.utcnow(),
                quality_score=0.8,
            )
            session.add(photo)
            session.commit()
            photo_id = photo.id
            
            # Deduplicate
            response = client.post("/deduplicate", json={"photo_ids": [photo_id]})
            assert response.status_code == 200
            
            # Verify file was moved to .trash/ in the same directory
            trash_path = os.path.join(nested_dir, ".trash", "IMG_001.jpg")
            assert os.path.exists(trash_path)
            assert not os.path.exists(file_path)
        finally:
            session.close()
    
    def test_deduplicate_response_structure(self, client, sample_photos):
        """Response should include moved_count and failed_moves."""
        photo_ids, _ = sample_photos
        response = client.post("/deduplicate", json={"photo_ids": photo_ids[:1]})
        assert response.status_code == 200
        data = response.json()
        assert "moved_count" in data
        assert "failed_moves" in data
        assert "message" in data
        assert data["moved_count"] == 1
        assert isinstance(data["failed_moves"], list)
