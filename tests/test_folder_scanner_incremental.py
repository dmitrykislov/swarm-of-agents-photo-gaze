"""Integration tests for incremental folder scanning with hash-based change detection."""
import pytest
import os
import tempfile
import shutil
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base, Photo
from app.folder_scanner import FolderScanner


class TestFolderScannerIncremental:
    """Test incremental folder scanning with change detection."""
    
    @pytest.fixture
    def temp_folder(self):
        """Create a temporary folder for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def db_session(self):
        """Create an in-memory SQLite database session for testing."""
        engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()
    
    @pytest.mark.integration
    def test_detect_new_photos(self, temp_folder, db_session):
        """Verify new photos are detected and queued for processing."""
        img_path = os.path.join(temp_folder, 'test.jpg')
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img_path, 'JPEG')
        
        scanner = FolderScanner()
        photo_ids, count = scanner.scan_folder(temp_folder, db_session)
        
        assert count == 1
        assert len(photo_ids) == 1
        photo = db_session.query(Photo).filter(Photo.id == photo_ids[0]).first()
        assert photo is not None
        assert photo.filename == 'test.jpg'
        assert photo.file_hash is not None
        assert len(photo.file_hash) == 64
    
    @pytest.mark.integration
    def test_detect_changed_photos(self, temp_folder, db_session):
        """Verify changed photos are detected and queued for reprocessing."""
        img_path = os.path.join(temp_folder, 'test.jpg')
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img_path, 'JPEG')
        
        scanner = FolderScanner()
        photo_ids_1, count_1 = scanner.scan_folder(temp_folder, db_session)
        assert count_1 == 1
        original_hash = db_session.query(Photo).filter(Photo.id == photo_ids_1[0]).first().file_hash
        
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(img_path, 'JPEG')
        
        photo_ids_2, count_2 = scanner.scan_folder(temp_folder, db_session)
        
        assert count_2 == 1
        assert photo_ids_2[0] == photo_ids_1[0]
        updated_photo = db_session.query(Photo).filter(Photo.id == photo_ids_1[0]).first()
        assert updated_photo.file_hash != original_hash
    
    @pytest.mark.integration
    def test_detect_deleted_photos(self, temp_folder, db_session):
        """Verify deleted photos are removed from database."""
        img_path_1 = os.path.join(temp_folder, 'test1.jpg')
        img_path_2 = os.path.join(temp_folder, 'test2.jpg')
        
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img_path_1, 'JPEG')
        img.save(img_path_2, 'JPEG')
        
        scanner = FolderScanner()
        photo_ids_1, count_1 = scanner.scan_folder(temp_folder, db_session)
        assert count_1 == 2
        
        photos = db_session.query(Photo).all()
        assert len(photos) == 2
        
        os.unlink(img_path_1)
        
        photo_ids_2, count_2 = scanner.scan_folder(temp_folder, db_session)
        
        photos = db_session.query(Photo).all()
        assert len(photos) == 1
        assert photos[0].filename == 'test2.jpg'
    
    @pytest.mark.integration
    def test_scan_does_not_delete_photos_in_other_folders(self, db_session):
        """Regression: scanning folder A must NOT delete photos belonging to
        a different registered folder B.

        The deleted-photo cleanup compares disk contents against the DB. It
        must be scoped to the folder being scanned; otherwise scanning A
        (whose scanned_paths never include B's files) wipes every Photo row
        under B — silent, cascading data loss across the user's other
        registered folders."""
        folder_a = tempfile.mkdtemp()
        folder_b = tempfile.mkdtemp()
        try:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(os.path.join(folder_a, 'a.jpg'), 'JPEG')
            img.save(os.path.join(folder_b, 'b.jpg'), 'JPEG')

            scanner = FolderScanner()
            scanner.scan_folder(folder_a, db_session)
            scanner.scan_folder(folder_b, db_session)
            assert db_session.query(Photo).count() == 2

            # Re-scan A. B's photo is untouched (it lives in a folder we
            # didn't walk), and A's photo survives because it's still on disk.
            scanner.scan_folder(folder_a, db_session)

            paths = {p.file_path for p in db_session.query(Photo).all()}
            assert os.path.join(folder_a, 'a.jpg') in paths
            assert os.path.join(folder_b, 'b.jpg') in paths, (
                "scanning folder A deleted folder B's photo — cleanup is not "
                "scoped to the scanned folder"
            )
        finally:
            shutil.rmtree(folder_a)
            shutil.rmtree(folder_b)

    @pytest.mark.integration
    def test_no_reprocessing_unchanged_photos(self, temp_folder, db_session):
        """Verify unchanged photos are not queued for reprocessing."""
        img_path = os.path.join(temp_folder, 'test.jpg')
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img_path, 'JPEG')
        
        scanner = FolderScanner()
        photo_ids_1, count_1 = scanner.scan_folder(temp_folder, db_session)
        assert count_1 == 1
        
        photo_ids_2, count_2 = scanner.scan_folder(temp_folder, db_session)
        
        assert count_2 == 0
        assert len(photo_ids_2) == 0
