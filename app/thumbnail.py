"""Thumbnail generation service with disk-based caching keyed by file_hash.

Cache invalidation is automatic: when a photo is modified, its file_hash changes,
so the old cache key is never looked up again. Stale thumbnails are cleaned up
lazily or can be purged via clear_cache().
"""
import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

# Default cache directory lives alongside the app (Docker). The native app
# MUST override this (THUMBNAILS_DIR) to a writable user dir — writing inside
# the .app bundle is read-only in /Applications AND breaks the code signature.
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".thumbnail_cache")


class ThumbnailService:
    """Generate and cache JPEG thumbnails on disk, keyed by (file_hash, size)."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = os.path.abspath(
            cache_dir or os.getenv("THUMBNAILS_DIR") or DEFAULT_CACHE_DIR
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_key(self, file_hash: str, size: Tuple[int, int]) -> str:
        """Deterministic cache filename from hash + dimensions."""
        raw = f"{file_hash}_{size[0]}x{size[1]}"
        return hashlib.md5(raw.encode()).hexdigest() + ".jpg"

    def _cache_path(self, file_hash: str, size: Tuple[int, int]) -> str:
        key = self._cache_key(file_hash, size)
        return os.path.join(self.cache_dir, key)

    def get_thumbnail(
        self,
        file_path: str,
        file_hash: str,
        size: Tuple[int, int] = (200, 200),
    ) -> str:
        """Return path to a cached thumbnail, generating it if absent.

        Because the cache key includes file_hash, a modified photo (with a new
        hash) will automatically miss the cache and get a fresh thumbnail.
        """
        cached = self._cache_path(file_hash, size)
        if os.path.isfile(cached):
            return cached
        return self._generate(file_path, cached, size)

    def _generate(
        self, file_path: str, output_path: str, size: Tuple[int, int]
    ) -> str:
        """Generate a JPEG thumbnail using Pillow's efficient thumbnail()."""
        with Image.open(file_path) as img:
            img.thumbnail(size, Image.LANCZOS)
            # Convert to RGB in case source is RGBA/palette
            if img.mode not in ("RGB",):
                img = img.convert("RGB")
            img.save(output_path, "JPEG", quality=85)
        return output_path

    def is_cached(self, file_hash: str, size: Tuple[int, int] = (200, 200)) -> bool:
        """Check whether a thumbnail is already cached for this hash+size."""
        return os.path.isfile(self._cache_path(file_hash, size))

    def invalidate(self, file_hash: str, size: Tuple[int, int] = (200, 200)) -> bool:
        """Explicitly remove a cached thumbnail. Returns True if file was deleted."""
        path = self._cache_path(file_hash, size)
        if os.path.isfile(path):
            os.remove(path)
            return True
        return False

    def clear_cache(self) -> int:
        """Remove all cached thumbnails. Returns count of files deleted."""
        count = 0
        for f in os.listdir(self.cache_dir):
            fp = os.path.join(self.cache_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)
                count += 1
        return count
