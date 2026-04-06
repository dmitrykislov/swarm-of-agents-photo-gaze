import pytest
from datetime import datetime, timedelta
from app import create_app, db
from app.models import Photo
from app.quality_scorer import QualityScorer

class TestQualityScorer:
    """Tests for photo quality scoring algorithm."""
    
    @pytest.fixture
    def app(self):
        """Create test app with in-memory database."""
        app = create_app('testing')
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()
    
    def test_quality_score_calculation(self, app):
        """Test that quality score is calculated correctly."""
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
            
            score = QualityScorer.calculate_quality_score(photo)
            assert isinstance(score, float)
            assert score > 0
    
    def test_quality_score_deterministic(self, app):
        """Test that quality score is deterministic."""
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
            
            score1 = QualityScorer.calculate_quality_score(photo)
            score2 = QualityScorer.calculate_quality_score(photo)
            assert score1 == score2
    
    def test_rank_by_resolution(self, app):
        """Test that photos are ranked by resolution (megapixels) first."""
        with app.app_context():
            photo_low = Photo(
                filename='low.jpg',
                file_path='/path/to/low.jpg',
                file_size_bytes=512000,
                width=1280,
                height=720,
                megapixels=0.92,
                created_date=datetime(2023, 1, 1),
                modified_date=datetime(2023, 1, 1)
            )
            photo_high = Photo(
                filename='high.jpg',
                file_path='/path/to/high.jpg',
                file_size_bytes=2048000,
                width=3840,
                height=2160,
                megapixels=8.29,
                created_date=datetime(2023, 1, 1),
                modified_date=datetime(2023, 1, 1)
            )
            
            ranked = QualityScorer.rank_similarity_group([photo_low, photo_high])
            assert ranked[0][0].filename == 'high.jpg'
            assert ranked[1][0].filename == 'low.jpg'
    
    def test_rank_by_creation_date_tiebreaker(self, app):
        """Test that creation date is used as tiebreaker when resolution is equal."""
        with app.app_context():
            photo_old = Photo(
                filename='old.jpg',
                file_path='/path/to/old.jpg',
                file_size_bytes=1024000,
                width=1920,
                height=1080,
                megapixels=2.07,
                created_date=datetime(2023, 1, 1),
                modified_date=datetime(2023, 1, 1)
            )
            photo_new = Photo(
                filename='new.jpg',
                file_path='/path/to/new.jpg',
                file_size_bytes=1024000,
                width=1920,
                height=1080,
                megapixels=2.07,
                created_date=datetime(2023, 6, 15),
                modified_date=datetime(2023, 6, 15)
            )
            
            ranked = QualityScorer.rank_similarity_group([photo_new, photo_old])
            assert ranked[0][0].filename == 'old.jpg'
            assert ranked[1][0].filename == 'new.jpg'
    
    def test_get_best_photo(self, app):
        """Test that get_best_photo returns the highest-ranked photo."""
        with app.app_context():
            photo1 = Photo(
                filename='photo1.jpg',
                file_path='/path/to/photo1.jpg',
                file_size_bytes=1024000,
                width=1920,
                height=1080,
                megapixels=2.07,
                created_date=datetime(2023, 6, 15),
                modified_date=datetime(2023, 6, 15)
            )
            photo2 = Photo(
                filename='photo2.jpg',
                file_path='/path/to/photo2.jpg',
                file_size_bytes=2048000,
                width=3840,
                height=2160,
                megapixels=8.29,
                created_date=datetime(2023, 6, 15),
                modified_date=datetime(2023, 6, 15)
            )
            
            best = QualityScorer.get_best_photo([photo1, photo2])
            assert best.filename == 'photo2.jpg'
    
    def test_get_best_photo_empty_list(self, app):
        """Test that get_best_photo raises error for empty list."""
        with app.app_context():
            with pytest.raises(ValueError):
                QualityScorer.get_best_photo([])
