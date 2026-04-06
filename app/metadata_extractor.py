import os
from datetime import datetime
from PIL import Image
import piexif
from typing import Optional, Dict, Any

class MetadataExtractor:
    """Extracts comprehensive metadata from image files for deduplication decisions."""
    
    @staticmethod
    def extract_metadata(file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from an image file.
        
        Args:
            file_path: Full path to the image file
        
        Returns:
            Dictionary containing extracted metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found: {file_path}')
        
        metadata = {
            'filename': os.path.basename(file_path),
            'file_path': os.path.abspath(file_path),
            'file_size_bytes': os.path.getsize(file_path),
            'modified_date': datetime.fromtimestamp(os.path.getmtime(file_path)),
            'width': None,
            'height': None,
            'megapixels': None,
            'created_date': None,
            'camera_info': None
        }
        
        try:
            with Image.open(file_path) as img:
                metadata['width'] = img.width
                metadata['height'] = img.height
                if metadata['width'] and metadata['height']:
                    metadata['megapixels'] = round(
                        (metadata['width'] * metadata['height']) / 1_000_000, 2
                    )
        except Exception as e:
            pass
        
        exif_date = MetadataExtractor._extract_exif_date(file_path)
        metadata['created_date'] = exif_date if exif_date else metadata['modified_date']
        
        metadata['camera_info'] = MetadataExtractor._extract_camera_info(file_path)
        
        return metadata
    
    @staticmethod
    def _extract_exif_date(file_path: str) -> Optional[datetime]:
        """
        Extract DateTimeOriginal from EXIF data.
        Falls back to filesystem ctime if EXIF data is unavailable.
        """
        try:
            exif_dict = piexif.load(file_path)
            if '0th' in exif_dict:
                exif_0th = exif_dict['0th']
                if piexif.ImageIFD.DateTime in exif_0th:
                    dt_bytes = exif_0th[piexif.ImageIFD.DateTime]
                    if isinstance(dt_bytes, bytes):
                        dt_str = dt_bytes.decode('utf-8')
                    else:
                        dt_str = dt_bytes
                    return datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def _extract_camera_info(file_path: str) -> Optional[str]:
        """
        Extract camera/device model from EXIF data.
        """
        try:
            exif_dict = piexif.load(file_path)
            if '0th' in exif_dict:
                exif_0th = exif_dict['0th']
                model_key = piexif.ImageIFD.Model
                if model_key in exif_0th:
                    model_bytes = exif_0th[model_key]
                    if isinstance(model_bytes, bytes):
                        return model_bytes.decode('utf-8').rstrip('\x00')
                    return str(model_bytes)
        except Exception:
            pass
        
        return None
