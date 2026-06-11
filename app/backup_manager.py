"""Automated backup and disaster recovery for PostgreSQL and Qdrant."""
import os
import asyncio
import logging
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)


class BackupManager:
    """Manage automated backups and recovery for PostgreSQL and Qdrant."""
    
    def __init__(self, backup_dir: str = None, backup_interval_hours: int = 24):
        """Initialize backup manager with backup directory and schedule.
        
        Args:
            backup_dir: Directory to store backups (default: ./backups)
            backup_interval_hours: Hours between automated backups (default: 24)
        """
        self.backup_dir = Path(backup_dir or os.getenv("BACKUP_DIR", "./backups"))
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backup_interval_hours = backup_interval_hours
        self.last_backup_time = None
        self.backup_task = None
        
        # Database configuration
        self.database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/app_db"
        )
        
        # Qdrant configuration
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_collection = "embeddings"
        
    async def schedule_automated_backups(self):
        """Start background task for automated backups at regular intervals."""
        if self.backup_task is not None:
            return  # Already scheduled
        
        self.backup_task = asyncio.create_task(self._backup_loop())
        logger.info("Automated backup scheduler started (interval: %d hours)", self.backup_interval_hours)
        
    async def _backup_loop(self):
        """Background loop that creates backups at regular intervals."""
        while True:
            try:
                await asyncio.sleep(self.backup_interval_hours * 3600)
                await self.create_backup()
            except Exception as e:
                logger.error("Error in backup loop: %s", e, exc_info=True)
        
    async def create_backup(self) -> str:
        """Create backup of PostgreSQL and Qdrant data.
        
        Returns:
            backup_id: Unique identifier for this backup
        """
        backup_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Backup PostgreSQL
            await self._backup_postgresql(backup_path)
            
            # Backup Qdrant and capture collection metadata
            qdrant_metadata = await self._backup_qdrant(backup_path)
            
            # Create metadata file
            metadata = {
                "backup_id": backup_id,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "completed",
                "postgresql": "success",
                "qdrant": "success",
                "qdrant_collection_metadata": qdrant_metadata
            }
            metadata_file = backup_path / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            self.last_backup_time = datetime.utcnow()
            logger.info("Backup %s completed successfully", backup_id)
            return backup_id
        except Exception as e:
            logger.error("Backup %s failed: %s", backup_id, e, exc_info=True)
            raise
        
    async def _backup_postgresql(self, backup_path: Path):
        """Backup PostgreSQL database using pg_dump.
        
        Args:
            backup_path: Directory to store backup files
        """
        # Parse database URL
        db_url = self.database_url
        if db_url.startswith("sqlite"):
            db_file = db_url.replace("sqlite:///", "")
            if not db_file:
                logger.warning("SQLite database path is empty after URL parsing")
                return
            if os.path.exists(db_file):
                # Use SQLite's online backup API, NOT a file copy: under WAL
                # mode recent commits live in the -wal sidecar and a plain copy
                # of just the .db misses them. .backup() produces a single,
                # consistent snapshot file with all committed data folded in.
                import sqlite3
                src = sqlite3.connect(db_file)
                try:
                    dst = sqlite3.connect(str(backup_path / "database.db"))
                    try:
                        src.backup(dst)
                    finally:
                        dst.close()
                finally:
                    src.close()
            return
        
        # Extract PostgreSQL connection parameters
        # Format: postgresql://user:password@host:port/dbname
        try:
            from urllib.parse import urlparse
            parsed = urlparse(db_url)
            user = parsed.username or "postgres"
            password = parsed.password or ""
            host = parsed.hostname or "localhost"
            port = parsed.port or 5432
            dbname = parsed.path.lstrip("/") or "app_db"
            
            # Create pg_dump command
            dump_file = backup_path / "database.sql"
            env = os.environ.copy()
            if password:
                env["PGPASSWORD"] = password
            
            cmd = [
                "pg_dump",
                "-h", host,
                "-p", str(port),
                "-U", user,
                "-d", dbname,
                "-F", "p",  # Plain text format
                "-f", str(dump_file)
            ]
            
            # Run pg_dump
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"pg_dump failed: {result.stderr}")
            
            logger.info("PostgreSQL backup saved to %s", dump_file)
        except Exception as e:
            logger.error("PostgreSQL backup failed: %s", e, exc_info=True)
            raise
        
    async def _backup_qdrant(self, backup_path: Path) -> Dict:
        """Backup Qdrant collection using snapshots.
        
        Args:
            backup_path: Directory to store backup files
            
        Returns:
            Dictionary with collection metadata (vector_size, distance)
        """
        try:
            client = QdrantClient(url=self.qdrant_url)
            
            # Get all points from collection
            try:
                collection_info = client.get_collection(self.qdrant_collection)
            except Exception:
                logger.warning("Qdrant collection '%s' not found, skipping backup", self.qdrant_collection)
                return {}
            
            # Scroll through all points and save to JSON
            points_data = []
            offset = 0
            limit = 100
            
            while True:
                try:
                    points, next_offset = client.scroll(
                        collection_name=self.qdrant_collection,
                        limit=limit,
                        offset=offset,
                        # qdrant-client defaults with_vectors=False — without
                        # this every point serializes as "vector": null and the
                        # restore rebuilds an empty index. The vectors ARE the
                        # backup.
                        with_vectors=True,
                        with_payload=True,
                    )
                    
                    if not points:
                        break
                    
                    for point in points:
                        points_data.append({
                            "id": point.id,
                            "vector": point.vector,
                            "payload": point.payload
                        })
                    
                    offset = next_offset
                    if next_offset is None:
                        break
                except Exception as e:
                    logger.error("Error scrolling Qdrant points: %s", e)
                    break
            
            # Save points to JSON file
            qdrant_file = backup_path / "qdrant_points.json"
            with open(qdrant_file, "w") as f:
                json.dump(points_data, f, indent=2)
            
            # Extract and return collection metadata for backup
            vector_size = collection_info.config.params.vectors.size
            distance = str(collection_info.config.params.distance)
            metadata = {
                "vector_size": vector_size,
                "distance": distance
            }
            logger.info("Qdrant backup saved to %s (%d points)", qdrant_file, len(points_data))
            return metadata
        except Exception as e:
            logger.error("Qdrant backup failed: %s", e, exc_info=True)
            raise
        
    async def restore_backup(self, backup_id: str) -> bool:
        """Restore PostgreSQL and Qdrant data from a backup.
        
        Args:
            backup_id: Backup identifier to restore from
            
        Returns:
            True if restore succeeded, False otherwise
        """
        backup_path = self.backup_dir / backup_id
        if not backup_path.exists():
            logger.error("Backup %s not found", backup_id)
            return False
        
        try:
            # Restore PostgreSQL
            await self._restore_postgresql(backup_path)
            
            # Restore Qdrant
            await self._restore_qdrant(backup_path)
            
            logger.info("Backup %s restored successfully", backup_id)
            return True
        except Exception as e:
            logger.error("Restore from backup %s failed: %s", backup_id, e, exc_info=True)
            return False
        
    async def _restore_postgresql(self, backup_path: Path):
        """Restore PostgreSQL database from backup.
        
        Args:
            backup_path: Directory containing backup files
        """
        dump_file = backup_path / "database.sql"
        if not dump_file.exists():
            logger.warning("PostgreSQL dump file not found in backup")
            return
        
        db_url = self.database_url
        if db_url.startswith("sqlite"):
            # For SQLite, restore from copied file
            db_file = db_url.replace("sqlite:///", "")
            if not db_file:
                logger.warning("SQLite database path is empty after URL parsing")
                return
            import shutil
            try:
                shutil.copy(backup_path / "database.db", db_file)
                # Drop any stale WAL sidecars from the previous database — left
                # in place they'd be replayed over the freshly restored file,
                # reintroducing the very data we just rolled back.
                for sidecar in (db_file + "-wal", db_file + "-shm"):
                    if os.path.exists(sidecar):
                        os.remove(sidecar)
                logger.info("SQLite restored from backup")
            except FileNotFoundError:
                logger.error("SQLite backup file not found: %s", backup_path / "database.db")
                raise
            except PermissionError:
                logger.error("Permission denied restoring SQLite to %s", db_file)
                raise
            return
        
        try:
            from urllib.parse import urlparse
            parsed = urlparse(db_url)
            user = parsed.username or "postgres"
            password = parsed.password or ""
            host = parsed.hostname or "localhost"
            port = parsed.port or 5432
            dbname = parsed.path.lstrip("/") or "app_db"
            
            # Create psql command to restore
            env = os.environ.copy()
            if password:
                env["PGPASSWORD"] = password
            
            cmd = [
                "psql",
                "-h", host,
                "-p", str(port),
                "-U", user,
                "-d", dbname,
                "-f", str(dump_file)
            ]
            
            # Run psql to restore
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"psql restore failed: {result.stderr}")
            
            logger.info("PostgreSQL restored from %s", dump_file)
        except Exception as e:
            logger.error("PostgreSQL restore failed: %s", e, exc_info=True)
            raise
        
    async def _restore_qdrant(self, backup_path: Path):
        """Restore Qdrant collection from backup.
        
        Args:
            backup_path: Directory containing backup files
        """
        qdrant_file = backup_path / "qdrant_points.json"
        if not qdrant_file.exists():
            logger.warning("Qdrant backup file not found in backup")
            return
        
        try:
            client = QdrantClient(url=self.qdrant_url)
            
            # Load points from backup
            with open(qdrant_file, "r") as f:
                points_data = json.load(f)
            
            if not points_data:
                logger.info("No points to restore in Qdrant backup")
                return
            
            # Load collection metadata from backup
            metadata_file = backup_path / "metadata.json"
            vector_size = 1024  # Default fallback
            distance_str = "COSINE"  # Default fallback
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    backup_metadata = json.load(f)
                    qdrant_meta = backup_metadata.get("qdrant_collection_metadata", {})
                    vector_size = qdrant_meta.get("vector_size", 1024)
                    distance_str = qdrant_meta.get("distance", "COSINE")
            
            # Delete existing collection
            try:
                client.delete_collection(self.qdrant_collection)
                logger.info("Deleted existing Qdrant collection '%s'", self.qdrant_collection)
            except Exception:
                pass  # Collection may not exist
            
            # Recreate collection with backed-up parameters
            from qdrant_client.models import Distance, VectorParams
            distance_enum = Distance.COSINE
            if "EUCLID" in distance_str.upper():
                distance_enum = Distance.EUCLID
            elif "MANHATTAN" in distance_str.upper():
                distance_enum = Distance.MANHATTAN
            client.create_collection(
                collection_name=self.qdrant_collection,
                vectors_config=VectorParams(size=vector_size, distance=distance_enum)
            )
            
            # Restore points in batches
            batch_size = 100
            for i in range(0, len(points_data), batch_size):
                batch = points_data[i:i + batch_size]
                points = [
                    PointStruct(
                        id=p["id"],
                        vector=p["vector"],
                        payload=p["payload"]
                    )
                    for p in batch
                ]
                client.upsert(
                    collection_name=self.qdrant_collection,
                    points=points
                )
            
            logger.info("Qdrant restored from %s (%d points)", qdrant_file, len(points_data))
        except Exception as e:
            logger.error("Qdrant restore failed: %s", e, exc_info=True)
            raise
        
    async def get_backup_status(self) -> Dict:
        """Get status of available backups and recovery options.
        
        Returns:
            Dictionary with backup list and status information
        """
        backups = []
        
        # List all backup directories
        if self.backup_dir.exists():
            for backup_path in sorted(self.backup_dir.iterdir(), reverse=True):
                if backup_path.is_dir():
                    metadata_file = backup_path / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        backups.append(metadata)
        
        return {
            "total_backups": len(backups),
            "last_backup_time": self.last_backup_time.isoformat() if self.last_backup_time else None,
            "backup_interval_hours": self.backup_interval_hours,
            "backups": backups[:10]  # Return last 10 backups
        }
    
    def verify_backup_integrity(self, backup_id: str) -> bool:
        """Verify that a backup contains all required files.
        
        Args:
            backup_id: Backup identifier to verify
            
        Returns:
            True if backup is valid, False otherwise
        """
        backup_path = self.backup_dir / backup_id
        if not backup_path.exists():
            return False
        
        # Check for metadata file
        metadata_file = backup_path / "metadata.json"
        if not metadata_file.exists():
            return False
        
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            if metadata.get("status") != "completed":
                return False
        except Exception:
            return False
        
        # Verify data files exist
        db_url = self.database_url
        if db_url.startswith("sqlite"):
            db_file = backup_path / "database.db"
        else:
            db_file = backup_path / "database.sql"
        
        qdrant_file = backup_path / "qdrant_points.json"
        
        # Both data files must exist for valid backup
        return db_file.exists() and qdrant_file.exists()
