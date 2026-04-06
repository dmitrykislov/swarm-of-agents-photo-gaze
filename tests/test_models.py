import pytest
from datetime import datetime
from app import create_app, db
from app.models import Photo

class TestPhotoModel:
    """Tests for Photo model."""
    
    @pytest.fixture
    def app(self):
        """Create test app with in-memory database."""
        app = create_app('testing')
        return app
    
    def test_photo_creation(self, app):
        """Test creating a Photo record."""
        with app.app_context():
            photo = Photo(
                filename='test.jpg',
                file_path='/path/to/test.jpg',
                file_size_bytes=1024000,
                width=1920,
                height=1080,
                megapixels=2.07,
                created_date=datetime(2023, 6, 15, 14, 30, 45),
                modified_date=datetime(2023, 6, 15, 14, 30, 45),
                camera_info='Canon EOS 5D Mark IV',
                quality_score=2070000.0,
                similarity_group_id=1
            )
            db.session.add(photo)
            db.session.commit()
            
            assert photo.id is not None
            assert photo.filename == 'test.jpg'
            assert photo.width == 1920
            assert photo.height == 1080
    
    def test_photo_to_dict(self, app):
        """Test Photo.to_dict() serialization."""
        with app.app_context():
            photo = Photo(
                filename='test.jpg',
                file_path='/path/to/test.jpg',
                file_size_bytes=1024000,
                width=1920,
                height=1080,
                megapixels=2.07,
                created_date=datetime(2023, 6, 15, 14, 30, 45),
                modified_date=datetime(2023, 6, 15, 14, 30, 45)
            )
            
            photo_dict = photo.to_dict()
            assert photo_dict['filename'] == 'test.jpg'
            assert photo_dict['width'] == 1920
            assert photo_dict['megapixels'] == 2.07
            assert 'created_date' in photo_dict
