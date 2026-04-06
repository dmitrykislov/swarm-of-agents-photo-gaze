import pytest
import os
import tempfile
from datetime import datetime
from PIL import Image
import piexif
from app.metadata_extractor import MetadataExtractor

class TestMetadataExtractor:
    """Tests for image metadata extraction."""
    
    @pytest.fixture
    def sample_image_path(self):
        """Create a temporary sample image for testing."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = Image.new('RGB', (1920, 1080), color='red')
            img.save(tmp.name, 'JPEG')
            yield tmp.name
        os.unlink(tmp.name)
    
    @pytest.fixture
    def sample_image_with_exif_path(self):
        """Create a temporary image with EXIF data for testing."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = Image.new('RGB', (3840, 2160), color='blue')
            img.save(tmp.name, 'JPEG')
            
            exif_dict = {
                '0th': {
                    piexif.ImageIFD.DateTime: b'2023:06:15 14:30:45',
                    piexif.ImageIFD.Model: b'Canon EOS 5D Mark IV\x00'
                }
            }
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, tmp.name)
            
            yield tmp.name
        os.unlink(tmp.name)
    
    def test_extract_basic_metadata(self, sample_image_path):
        """Test extraction of basic image metadata."""
        metadata = MetadataExtractor.extract_metadata(sample_image_path)
        
        assert metadata['filename'] == os.path.basename(sample_image_path)
        assert metadata['file_path'] == os.path.abspath(sample_image_path)
        assert metadata['file_size_bytes'] > 0
        assert metadata['width'] == 1920
        assert metadata['height'] == 1080
        assert metadata['megapixels'] == 2.07
        assert metadata['modified_date'] is not None
        assert isinstance(metadata['modified_date'], datetime)
    
    def test_extract_metadata_with_exif(self, sample_image_with_exif_path):
        """Test extraction of metadata including EXIF data."""
        metadata = MetadataExtractor.extract_metadata(sample_image_with_exif_path)
        
        assert metadata['width'] == 3840
        assert metadata['height'] == 2160
        assert metadata['megapixels'] == 8.29
        assert metadata['created_date'] is not None
        assert metadata['created_date'].year == 2023
        assert metadata['created_date'].month == 6
        assert metadata['created_date'].day == 15
        assert 'Canon' in metadata['camera_info']
    
    def test_extract_metadata_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError):
            MetadataExtractor.extract_metadata('/nonexistent/path/image.jpg')
    
    def test_metadata_deterministic(self, sample_image_path):
        """Test that metadata extraction is deterministic."""
        metadata1 = MetadataExtractor.extract_metadata(sample_image_path)
        metadata2 = MetadataExtractor.extract_metadata(sample_image_path)
        
        assert metadata1['width'] == metadata2['width']
        assert metadata1['height'] == metadata2['height']
        assert metadata1['megapixels'] == metadata2['megapixels']
        assert metadata1['filename'] == metadata2['filename']
