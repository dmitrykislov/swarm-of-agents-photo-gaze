"""Unit tests for image metadata extraction and format validation."""
import pytest
import os
import tempfile
import hashlib
from datetime import datetime
from PIL import Image
from app.metadata_extractor import (
    extract_metadata,
    validate_image_format,
    ImageMetadata,
    SUPPORTED_FORMATS,
)


class TestImageMetadataDataclass:
    """Unit tests for ImageMetadata dataclass."""
    
    @pytest.mark.unit
    def test_image_metadata_creation(self):
        """Verify ImageMetadata dataclass can be instantiated with all fields."""
        metadata = ImageMetadata(
            filename="test.jpg",
            file_path="/path/to/test.jpg",
            file_size=1024,
            width=800,
            height=600,
            format="JPEG",
            creation_timestamp=1234567890.0,
            file_hash="abc123def456",
        )
        assert metadata.filename == "test.jpg"
        assert metadata.width == 800
        assert metadata.height == 600
        assert metadata.format == "JPEG"


class TestExtractMetadataJPEG:
    """Unit tests for JPEG image metadata extraction."""
    
    @pytest.fixture
    def jpeg_file(self):
        """Create a temporary JPEG file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Create a simple 100x100 RGB image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(f.name, 'JPEG')
            yield f.name
        os.unlink(f.name)
    
    @pytest.mark.unit
    def test_extract_jpeg_metadata(self, jpeg_file):
        """Verify JPEG metadata extraction returns correct dimensions and format."""
        metadata = extract_metadata(jpeg_file)
        assert metadata.format == 'JPEG'
        assert metadata.width == 100
        assert metadata.height == 100
        assert metadata.file_size > 0
        assert os.path.basename(jpeg_file) == metadata.filename
    
    @pytest.mark.unit
    def test_extract_jpeg_hash(self, jpeg_file):
        """Verify JPEG file hash is computed correctly."""
        metadata = extract_metadata(jpeg_file)
        # Verify hash is a valid hex string of correct length (SHA256 = 64 chars)
        assert len(metadata.file_hash) == 64
        assert all(c in '0123456789abcdef' for c in metadata.file_hash)
    
    @pytest.mark.unit
    def test_extract_jpeg_timestamp(self, jpeg_file):
        """Verify JPEG creation timestamp is extracted."""
        metadata = extract_metadata(jpeg_file)
        assert isinstance(metadata.creation_timestamp, float)
        assert metadata.creation_timestamp > 0


class TestExtractMetadataPNG:
    """Unit tests for PNG image metadata extraction."""
    
    @pytest.fixture
    def png_file(self):
        """Create a temporary PNG file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create a simple 200x150 RGB image
            img = Image.new('RGB', (200, 150), color='blue')
            img.save(f.name, 'PNG')
            yield f.name
        os.unlink(f.name)
    
    @pytest.mark.unit
    def test_extract_png_metadata(self, png_file):
        """Verify PNG metadata extraction returns correct dimensions and format."""
        metadata = extract_metadata(png_file)
        assert metadata.format == 'PNG'
        assert metadata.width == 200
        assert metadata.height == 150
        assert metadata.file_size > 0
    
    @pytest.mark.unit
    def test_extract_png_hash(self, png_file):
        """Verify PNG file hash is computed and is deterministic."""
        metadata1 = extract_metadata(png_file)
        metadata2 = extract_metadata(png_file)
        assert metadata1.file_hash == metadata2.file_hash


class TestExtractMetadataWebP:
    """Unit tests for WebP image metadata extraction."""
    
    @pytest.fixture
    def webp_file(self):
        """Create a temporary WebP file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
            # Create a simple 300x200 RGB image
            img = Image.new('RGB', (300, 200), color='green')
            img.save(f.name, 'WEBP')
            yield f.name
        os.unlink(f.name)
    
    @pytest.mark.unit
    def test_extract_webp_metadata(self, webp_file):
        """Verify WebP metadata extraction returns correct dimensions and format."""
        metadata = extract_metadata(webp_file)
        assert metadata.format == 'WEBP'
        assert metadata.width == 300
        assert metadata.height == 200


class TestValidateImageFormat:
    """Unit tests for image format validation."""
    
    @pytest.fixture
    def jpeg_file(self):
        """Create a temporary JPEG file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(f.name, 'JPEG')
            yield f.name
        os.unlink(f.name)
    
    @pytest.mark.unit
    def test_validate_supported_format(self, jpeg_file):
        """Verify validation returns True for supported JPEG format."""
        assert validate_image_format(jpeg_file) is True
    
    @pytest.mark.unit
    def test_validate_nonexistent_file(self):
        """Verify validation returns False for nonexistent file."""
        assert validate_image_format("/nonexistent/path/file.jpg") is False
    
    @pytest.mark.unit
    def test_validate_text_file(self):
        """Verify validation returns False for non-image file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"This is not an image")
            f.flush()
            result = validate_image_format(f.name)
        os.unlink(f.name)
        assert result is False


class TestExtractMetadataErrorHandling:
    """Unit tests for error handling in metadata extraction."""
    
    @pytest.mark.unit
    def test_extract_nonexistent_file(self):
        """Verify FileNotFoundError is raised for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            extract_metadata("/nonexistent/path/file.jpg")
    
    @pytest.mark.unit
    def test_extract_invalid_image_file(self):
        """Verify ValueError is raised for invalid image file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"This is not a valid JPEG")
            f.flush()
            with pytest.raises(ValueError):
                extract_metadata(f.name)
        os.unlink(f.name)
    
    @pytest.mark.unit
    def test_extract_unsupported_format(self):
        """Verify ValueError is raised for an unsupported image format.

        BMP/GIF/etc. are now first-class supported formats (the app handles 22
        formats), so use PPM — a format Pillow can write but the pipeline does
        not accept — to exercise the rejection path."""
        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(f.name, 'PPM')
            with pytest.raises(ValueError, match="Unsupported image format"):
                extract_metadata(f.name)
        os.unlink(f.name)


class TestSupportedFormats:
    """Unit tests for supported formats constant."""

    @pytest.mark.unit
    def test_supported_formats_contains_required_formats(self):
        """The core formats must always be supported. SUPPORTED_FORMATS has
        since grown to cover 22 formats (HEIF, AVIF, RAW variants, …); assert
        the required ones are a subset rather than pinning the exact set."""
        required = {'JPEG', 'PNG', 'WEBP', 'RAW'}
        assert required.issubset(SUPPORTED_FORMATS)


class TestMetadataIntegration:
    """Integration tests for metadata extraction workflow."""
    
    @pytest.mark.unit
    def test_extract_and_validate_workflow(self):
        """Verify complete workflow: validate format then extract metadata."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new('RGB', (150, 150), color='yellow')
            img.save(f.name, 'PNG')
            
            # Validate format first
            assert validate_image_format(f.name) is True
            
            # Then extract metadata
            metadata = extract_metadata(f.name)
            assert metadata.format == 'PNG'
            assert metadata.width == 150
            assert metadata.height == 150
            assert len(metadata.file_hash) == 64
        
        os.unlink(f.name)

