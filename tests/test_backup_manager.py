"""Unit tests for backup and disaster recovery functionality."""
import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock
from app.backup_manager import BackupManager


class _StatefulFakeQdrant:
    """A tiny in-memory Qdrant faithful enough to round-trip a backup:
    scroll returns vectors ONLY when with_vectors=True (like the real client),
    and upsert/create/delete actually mutate the store — so a backup→restore
    cycle can assert the vectors genuinely survived."""

    def __init__(self, vector_size=512):
        self.points = {}            # id -> (vector, payload)
        self.vector_size = vector_size
        self.created_with = None

    def seed(self, items):
        for pid, vec, payload in items:
            self.points[pid] = (vec, payload)

    def get_collection(self, name):
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(
            vectors=SimpleNamespace(size=self.vector_size), distance="Cosine")))

    def scroll(self, collection_name=None, limit=100, offset=None,
               with_vectors=False, with_payload=True):
        items = list(self.points.items())
        start = offset or 0
        page = items[start:start + limit]
        pts = [
            SimpleNamespace(
                id=pid,
                vector=(list(vec) if with_vectors else None),
                payload=(payload if with_payload else None),
            )
            for pid, (vec, payload) in page
        ]
        nxt = start + limit if start + limit < len(items) else None
        return pts, nxt

    def delete_collection(self, name):
        self.points = {}

    def create_collection(self, collection_name=None, vectors_config=None):
        self.created_with = vectors_config
        self.points = {}

    def upsert(self, collection_name=None, points=None):
        for p in (points or []):
            self.points[p.id] = (p.vector, p.payload)


class TestBackupManager:
    """Test backup creation, recovery, and data integrity verification."""
    
    @pytest.fixture
    def backup_manager(self):
        """Create BackupManager with temporary backup directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BackupManager(backup_dir=tmpdir, backup_interval_hours=1)
            yield manager
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_backup_creates_directory(self, backup_manager):
        """Verify backup creation initializes backup directory structure."""
        with patch.object(backup_manager, '_backup_postgresql', new_callable=AsyncMock):
            with patch.object(backup_manager, '_backup_qdrant', AsyncMock(return_value={})):
                backup_id = await backup_manager.create_backup()
                
                backup_path = backup_manager.backup_dir / backup_id
                assert backup_path.exists()
                assert (backup_path / "metadata.json").exists()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_backup_metadata_contains_timestamp(self, backup_manager):
        """Verify backup metadata includes timestamp and status."""
        with patch.object(backup_manager, '_backup_postgresql', new_callable=AsyncMock):
            with patch.object(backup_manager, '_backup_qdrant', AsyncMock(return_value={})):
                backup_id = await backup_manager.create_backup()
                
                metadata_file = backup_manager.backup_dir / backup_id / "metadata.json"
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                assert metadata["backup_id"] == backup_id
                assert metadata["status"] == "completed"
                assert "timestamp" in metadata
                assert metadata["postgresql"] == "success"
                assert metadata["qdrant"] == "success"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_backup_postgresql_creates_dump_file(self, backup_manager):
        """Verify PostgreSQL backup creates SQL dump file."""
        # The test DB URL is sqlite (conftest); force the Postgres branch.
        backup_manager.database_url = "postgresql://u:p@localhost:5432/db"
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_path = Path(tmpdir)

            # Mock subprocess for pg_dump
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="")
                
                await backup_manager._backup_postgresql(backup_path)
                
                # Verify pg_dump was called
                mock_run.assert_called_once()
                call_args = mock_run.call_args[0][0]
                assert "pg_dump" in call_args
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_backup_qdrant_saves_points_json(self, backup_manager):
        """Verify Qdrant backup saves points to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_path = Path(tmpdir)
            
            # Mock Qdrant client
            mock_client = MagicMock()
            mock_client.get_collection.return_value = MagicMock()
            mock_client.scroll.return_value = ([], None)  # No points
            
            with patch('app.backup_manager.QdrantClient', return_value=mock_client):
                await backup_manager._backup_qdrant(backup_path)
                
                # Verify JSON file was created
                qdrant_file = backup_path / "qdrant_points.json"
                assert qdrant_file.exists()
                
                with open(qdrant_file, "r") as f:
                    points = json.load(f)
                assert isinstance(points, list)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_restore_backup_returns_false_for_missing_backup(self, backup_manager):
        """Verify restore returns False when backup doesn't exist."""
        result = await backup_manager.restore_backup("nonexistent_backup")
        assert result is False
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_restore_postgresql_from_backup(self, backup_manager):
        """Verify PostgreSQL restore executes psql command."""
        # The test DB URL is sqlite (conftest); force the Postgres branch.
        backup_manager.database_url = "postgresql://u:p@localhost:5432/db"
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_path = Path(tmpdir)
            backup_path.mkdir(exist_ok=True)

            # Create dummy dump file
            dump_file = backup_path / "database.sql"
            dump_file.write_text("SELECT 1;")
            
            # Mock subprocess for psql
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="")
                
                await backup_manager._restore_postgresql(backup_path)
                
                # Verify psql was called
                mock_run.assert_called_once()
                call_args = mock_run.call_args[0][0]
                assert "psql" in call_args
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_restore_qdrant_from_backup(self, backup_manager):
        """Verify Qdrant restore recreates collection and restores points."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_path = Path(tmpdir)
            backup_path.mkdir(exist_ok=True)
            
            # Create dummy Qdrant backup file
            points_data = [
                {"id": 1, "vector": [0.1] * 1024, "payload": {"photo_id": 1}},
                {"id": 2, "vector": [0.2] * 1024, "payload": {"photo_id": 2}}
            ]
            qdrant_file = backup_path / "qdrant_points.json"
            with open(qdrant_file, "w") as f:
                json.dump(points_data, f)
            
            # Mock Qdrant client
            mock_client = MagicMock()
            
            with patch('app.backup_manager.QdrantClient', return_value=mock_client):
                await backup_manager._restore_qdrant(backup_path)
                
                # Verify collection was deleted and recreated
                mock_client.delete_collection.assert_called_once()
                mock_client.create_collection.assert_called_once()
                # Verify upsert was called for points
                mock_client.upsert.assert_called()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_backup_status_lists_backups(self, backup_manager):
        """Verify backup status returns list of available backups."""
        # Create two backup dirs with DISTINCT ids directly — create_backup's id
        # is second-granularity, so two rapid calls collide into one dir and the
        # count is flaky. We're testing get_backup_status's listing here.
        for bid in ("20240101_000001", "20240101_000002"):
            d = backup_manager.backup_dir / bid
            d.mkdir(parents=True)
            (d / "metadata.json").write_text(json.dumps(
                {"backup_id": bid, "status": "completed", "timestamp": "2024-01-01T00:00:00"}
            ))

        status = await backup_manager.get_backup_status()

        assert status["total_backups"] == 2
        assert len(status["backups"]) == 2
        assert status["backup_interval_hours"] == 1
    
    @pytest.mark.unit
    def test_verify_backup_integrity_valid_backup(self, backup_manager):
        """Verify backup integrity check passes for valid backup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_path = Path(tmpdir) / "test_backup"
            backup_path.mkdir(parents=True)
            
            # Create valid metadata
            metadata = {
                "backup_id": "test_backup",
                "status": "completed",
                "timestamp": "2024-01-01T00:00:00"
            }
            metadata_file = backup_path / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)
            # A valid backup also needs its data files present (sqlite DB +
            # Qdrant points) — integrity is more than just metadata existing.
            (backup_path / "database.db").write_bytes(b"SQLite format 3\x00")
            (backup_path / "qdrant_points.json").write_text("[]")

            # Patch backup_dir to use test directory
            backup_manager.backup_dir = Path(tmpdir)

            result = backup_manager.verify_backup_integrity("test_backup")
            assert result is True
    
    @pytest.mark.unit
    def test_verify_backup_integrity_missing_backup(self, backup_manager):
        """Verify backup integrity check fails for missing backup."""
        result = backup_manager.verify_backup_integrity("nonexistent")
        assert result is False
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_schedule_automated_backups_creates_task(self, backup_manager):
        """Verify automated backup scheduling creates background task."""
        with patch.object(backup_manager, 'create_backup', new_callable=AsyncMock):
            await backup_manager.schedule_automated_backups()
            
            assert backup_manager.backup_task is not None
            assert not backup_manager.backup_task.done()
            
            # Clean up task
            backup_manager.backup_task.cancel()
            try:
                await backup_manager.backup_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_backup_recovery_cycle_preserves_data(self, backup_manager):
        """Verify complete backup and recovery cycle preserves data integrity.
        
        This test verifies:
        1. create_backup() successfully backs up PostgreSQL and Qdrant data
        2. Backup metadata includes collection info (vector size, distance metric)
        3. restore_backup() correctly restores data with original collection parameters
        """
        # A stateful fake that actually stores vectors, so we can assert they
        # survive the full cycle (the same instance backs both phases).
        fake = _StatefulFakeQdrant(vector_size=512)
        fake.seed([
            (1, [0.1] * 512, {"photo_id": 1}),
            (2, [0.2] * 512, {"photo_id": 2}),
        ])

        # Step 1: Create backup
        with patch('app.backup_manager.QdrantClient', return_value=fake):
            with patch.object(backup_manager, '_backup_postgresql', new_callable=AsyncMock):
                backup_id = await backup_manager.create_backup()

        backup_dir = backup_manager.backup_dir / backup_id
        assert (backup_dir / "metadata.json").exists()
        assert (backup_dir / "qdrant_points.json").exists()
        with open(backup_dir / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["status"] == "completed"
        assert metadata["qdrant_collection_metadata"]["vector_size"] == 512

        # The backup file must contain the ACTUAL vectors, not null — this is
        # the regression guard for the with_vectors=True fix (a null-vector
        # backup is silently unrestorable).
        with open(backup_dir / "qdrant_points.json") as f:
            saved = json.load(f)
        assert {p["id"] for p in saved} == {1, 2}
        assert all(p["vector"] is not None and len(p["vector"]) == 512 for p in saved)

        # Step 2: Restore — wipe the store first so we prove it's rebuilt from
        # the backup file, not left over.
        fake.points = {}
        with patch('app.backup_manager.QdrantClient', return_value=fake):
            with patch.object(backup_manager, '_restore_postgresql', new_callable=AsyncMock):
                result = await backup_manager.restore_backup(backup_id)

        assert result is True
        # Collection recreated at the backed-up dimension (from metadata).
        assert fake.created_with.size == 512
        # Vectors genuinely round-tripped: same ids, same values, same payload.
        assert set(fake.points.keys()) == {1, 2}
        assert fake.points[1][0] == [0.1] * 512
        assert fake.points[1][1] == {"photo_id": 1}
