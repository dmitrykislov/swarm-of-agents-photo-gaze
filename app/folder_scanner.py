"""Folder scanner for recursively discovering and queuing photos for processing."""
import os
from pathlib import Path
from typing import List, Tuple, Set
from datetime import datetime
from sqlalchemy.orm import Session
from app.models import Photo, ProcessingState
from app.metadata_extractor import extract_metadata


# Directory names that must NEVER be recursed into during a scan, regardless
# of where they appear in the user's tree. The dedupe trash dir lives here:
# if a user registers /Users/me as a photo folder and we trash a duplicate,
# the file lands at /Users/me/.photo-gaze-trash/... — recursing back into
# that on the next scan would re-ingest the just-deleted file and rebuild
# its embedding. macOS system trash dirs are excluded for the same reason.
_EXCLUDED_DIR_NAMES: Set[str] = {
    ".photo-gaze-trash",   # this app's trash (default + most overrides)
    ".Trash",              # macOS user Trash
    ".Trashes",            # macOS volume-level Trash
    ".fseventsd",          # macOS Spotlight metadata
    ".Spotlight-V100",     # macOS Spotlight index
    ".DocumentRevisions-V100",  # macOS file-versioning store
    ".TemporaryItems",     # macOS staging dir
    "__MACOSX",            # zip/tar metadata sidecars
    "$RECYCLE.BIN",        # Windows recycle bin (network drives)
    "System Volume Information",  # Windows
    ".git", ".svn", ".hg", # VCS internals
}


def _trash_dir_abs() -> str:
    """Resolve the configured trash directory once. Read at call time so
    overriding TRASH_DIR via env in tests/dev takes effect immediately."""
    raw = os.getenv("TRASH_DIR", os.path.expanduser("~/.photo-gaze-trash"))
    return os.path.realpath(os.path.abspath(raw))


def _is_excluded(dir_path: str, dir_name: str, trash_abs: str) -> bool:
    """True if this directory must be skipped by the scanner."""
    if dir_name in _EXCLUDED_DIR_NAMES:
        return True
    # Catch-all for hidden dirs — photos shouldn't live under them and a
    # custom TRASH_DIR is normally hidden.
    if dir_name.startswith("."):
        return True
    # Belt-and-suspenders: even if a non-hidden custom TRASH_DIR is set
    # and lives inside a registered folder, an absolute-path match
    # excludes it.
    try:
        return os.path.realpath(os.path.abspath(dir_path)) == trash_abs
    except OSError:
        return False


class FolderScanner:
    """Scans folders recursively and queues photos for processing."""
    
    SUPPORTED_FORMATS = {
        ".jpg", ".jpeg", ".jfif",   # JPEG variants
        ".png",                      # PNG
        ".gif",                      # GIF
        ".bmp",                      # Bitmap
        ".webp",                     # WebP
        ".heic", ".heif",            # Apple HEIC/HEIF
        ".tiff", ".tif",             # TIFF
        ".avif",                     # AV1 Image
        ".ico",                      # Icon
        ".dng",                      # Adobe RAW
        ".cr2", ".nef", ".arw",      # Canon/Nikon/Sony RAW
        ".orf", ".rw2", ".pef",      # Olympus/Panasonic/Pentax RAW
    }
    
    def __init__(self, qdrant_client=None, qdrant_collection: str = "embeddings"):
        """Initialize folder scanner.

        Args:
            qdrant_client: optional Qdrant client. When provided, photos whose
                files were deleted on disk have their vectors purged from Qdrant
                during cleanup — otherwise those vectors are orphaned forever
                (the DB row + cascade go, but the vector lingers and can still
                match new photos until a full index rebuild).
            qdrant_collection: collection holding the embedding vectors.
        """
        self.qdrant_client = qdrant_client
        self.qdrant_collection = qdrant_collection
    
    def scan_folder(self, folder_path: str, session: Session) -> Tuple[List[int], int]:
        """Recursively scan folder and queue photos with incremental change detection.
        
        Detects new files, changed files (via hash comparison), and deleted files.
        Only new or changed photos are queued for reprocessing.
        
        Args:
            folder_path: Root folder path to scan
            session: SQLAlchemy session for database operations
        
        Returns:
            Tuple of (list of photo IDs to process, total count of new/changed photos)
        """
        photo_ids = []
        total_count = 0
        scanned_paths: Set[str] = set()  # Track files found on disk
        
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Recursively walk through folder. Mutate `dirs` in-place so
        # os.walk doesn't descend into the trash, hidden dirs, or system
        # metadata directories (see _is_excluded). This is the only place
        # the trash exclusion is enforced — trash files never enter Photo
        # / Embedding / Qdrant.
        trash_abs = _trash_dir_abs()
        for root, dirs, files in os.walk(folder_path):
            dirs[:] = [
                d for d in dirs
                if not _is_excluded(os.path.join(root, d), d, trash_abs)
            ]
            for filename in files:
                file_path = os.path.join(root, filename)
                file_ext = Path(filename).suffix.lower()
                
                # Check if file is a supported image format
                if file_ext not in self.SUPPORTED_FORMATS:
                    continue
                
                scanned_paths.add(file_path)
                
                try:
                    # Get file size and MIME type
                    file_size = os.path.getsize(file_path)
                    mime_type = self._get_mime_type(file_ext)
                    
                    # Compute file hash for change detection
                    file_hash = None
                    try:
                        metadata = extract_metadata(file_path)
                        file_hash = metadata.file_hash
                    except Exception as e:
                        print(f"Warning: Could not compute hash for {file_path}: {e}")
                    
                    # Check if photo already exists in database
                    existing = session.query(Photo).filter(
                        Photo.file_path == file_path
                    ).first()
                    
                    if existing:
                        # File exists in DB; check if it changed
                        if file_hash and existing.file_hash and file_hash != existing.file_hash:
                            # File was modified; update hash and mark for reprocessing
                            existing.file_hash = file_hash
                            existing.file_size = file_size
                            session.add(existing)
                            session.flush()
                            photo_ids.append(existing.id)
                            total_count += 1
                        elif not existing.file_hash and file_hash:
                            # Hash was missing; store it now
                            existing.file_hash = file_hash
                            session.add(existing)
                            session.flush()
                        continue
                    
                    # Create new photo record
                    photo = Photo(
                        filename=filename,
                        file_path=file_path,
                        file_size=file_size,
                        mime_type=mime_type,
                        file_hash=file_hash,
                        uploaded_at=datetime.utcnow()
                    )
                    session.add(photo)
                    session.flush()  # Get the ID without committing
                    
                    # Create processing state record
                    processing_state = ProcessingState(
                        photo_id=photo.id,
                        status="pending",
                        extraction_status="pending",
                        embedding_status="pending"
                    )
                    session.add(processing_state)
                    session.flush()
                    
                    photo_ids.append(photo.id)
                    total_count += 1
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue
        
        # Detect and remove deleted photos. Scoped to THIS folder — see
        # _cleanup_deleted_photos for why passing folder_path is mandatory.
        deleted_count = self._cleanup_deleted_photos(session, scanned_paths, folder_path)
        if deleted_count > 0:
            print(f"Removed {deleted_count} deleted photos from database")
        
        session.commit()
        return photo_ids, total_count
    
    def _cleanup_deleted_photos(
        self, session: Session, scanned_paths: Set[str], folder_path: str
    ) -> int:
        """Remove photos that live UNDER ``folder_path`` but were not found
        on disk during this scan (i.e. the user deleted them).

        Scoping to ``folder_path`` is essential. ``scanned_paths`` only
        contains files discovered under the folder we just walked. If we
        compared every Photo row in the database against it — as an earlier
        version did — then scanning one registered folder would delete the
        Photo (and, via cascade, ProcessingState + Embedding) rows of every
        OTHER registered folder, because their paths legitimately aren't in
        this scan's results. That is silent data loss: the user's other
        folders vanish from the index until re-scanned. See the regression
        test test_scan_does_not_delete_photos_in_other_folders.

        Args:
            session: SQLAlchemy session for database operations
            scanned_paths: Set of file paths found during this folder scan
            folder_path: Root of the folder that was just scanned

        Returns:
            Count of deleted photo records
        """
        folder_abs = os.path.abspath(folder_path)
        # Trailing separator so /photos/a is not treated as a prefix of
        # /photos/abc.
        folder_prefix = folder_abs.rstrip(os.sep) + os.sep

        to_delete = []
        for photo in session.query(Photo).all():
            photo_abs = os.path.abspath(photo.file_path)
            under_folder = photo_abs == folder_abs or photo_abs.startswith(folder_prefix)
            if not under_folder:
                continue  # belongs to a different folder — never touch it
            if photo.file_path not in scanned_paths:
                to_delete.append(photo)

        if not to_delete:
            return 0

        # Purge the vectors from Qdrant FIRST. session.delete cascades to the
        # Embedding rows, but Qdrant is a separate store — without this the
        # vectors are orphaned: they survive, keep matching new photos in the
        # incremental index, and bloat storage. Delete vectors before the DB
        # rows so a Qdrant failure leaves the (recoverable) DB rows intact.
        from app.models import Embedding
        point_ids = [
            pid for (pid,) in session.query(Embedding.qdrant_point_id)
            .filter(Embedding.photo_id.in_([p.id for p in to_delete]))
            .filter(Embedding.qdrant_point_id.isnot(None))
            .all()
        ]
        if point_ids and self.qdrant_client is not None:
            try:
                self.qdrant_client.delete(
                    collection_name=self.qdrant_collection,
                    points_selector=point_ids,
                )
            except Exception as e:
                print(f"WARNING: failed to delete {len(point_ids)} orphaned Qdrant points: {e}")

        for photo in to_delete:
            session.delete(photo)

        return len(to_delete)
    
    def _get_mime_type(self, file_ext: str) -> str:
        """Get MIME type for file extension.
        
        Args:
            file_ext: File extension (e.g., '.jpg')
        
        Returns:
            MIME type string
        """
        mime_types = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".jfif": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
            ".heic": "image/heic", ".heif": "image/heif",
            ".tiff": "image/tiff", ".tif": "image/tiff",
            ".avif": "image/avif",
            ".ico": "image/x-icon",
            ".dng": "image/x-adobe-dng",
            ".cr2": "image/x-canon-cr2", ".nef": "image/x-nikon-nef",
            ".arw": "image/x-sony-arw", ".orf": "image/x-olympus-orf",
            ".rw2": "image/x-panasonic-rw2", ".pef": "image/x-pentax-pef",
        }
        return mime_types.get(file_ext, "image/unknown")

