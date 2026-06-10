"""Unit tests for async job queue with checkpoint persistence and state recovery.

The heavy dependencies (DINOv2 embedding model, Qdrant client, metadata
extractor) are mocked so the queue logic can be exercised hermetically — the
real EmbeddingGenerator loads a torch model on construction, which is both slow
and unavailable in CI. process_photo's per-photo work is stubbed to succeed so
the tests focus on counting, checkpointing, completion, and recovery.
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base, JobQueue, Photo, ProcessingState
from app.job_queue import JobQueueManager


class TestJobQueueManager:
    """Unit tests for JobQueueManager async job processing and checkpointing."""

    @pytest.fixture
    def db_url(self, tmp_path):
        """File-backed SQLite so a second manager (restart simulation) can
        reopen the same database. :memory: is per-connection and wouldn't
        survive recover_from_checkpoint's fresh manager."""
        return f"sqlite:///{tmp_path / 'jobqueue.db'}"

    @pytest.fixture
    def job_queue(self, db_url):
        """Create a JobQueueManager with all heavy deps mocked and per-photo
        processing stubbed to succeed."""
        with patch("app.job_queue.EmbeddingGenerator"), \
             patch("app.job_queue.MetadataExtractor"), \
             patch("app.job_queue.QdrantClient"):
            manager = JobQueueManager(database_url=db_url)
        Base.metadata.create_all(manager.engine)
        # Stub the per-photo pipeline: metadata + embedding + qdrant upsert.
        manager.metadata_extractor.extract = AsyncMock(return_value=MagicMock())
        manager.embedding_generator.generate = AsyncMock(return_value=[0.1] * 384)
        manager.qdrant_client.upsert = MagicMock()
        return manager

    @staticmethod
    def _seed_photos(manager, count, start=1):
        """Insert Photo + ProcessingState rows so process_photo finds a file
        path and a state row to mark completed."""
        session = manager.SessionLocal()
        for i in range(start, start + count):
            session.add(Photo(
                id=i, filename=f"p{i}.jpg", file_path=f"/photos/p{i}.jpg",
                file_size=1000, mime_type="image/jpeg",
            ))
            session.add(ProcessingState(photo_id=i, status="pending"))
        session.commit()
        session.close()

    @pytest.mark.unit
    def test_create_job(self, job_queue):
        """Verify job creation stores job in database with correct initial state."""
        job_id = "test_job_001"
        result = job_queue.create_job(job_id, 25)
        assert result is True

        session = job_queue.SessionLocal()
        job = session.query(JobQueue).filter(JobQueue.job_id == job_id).first()
        assert job is not None
        assert job.status == "pending"
        assert job.total_photos == 25
        assert job.processed_photos == 0
        assert job.checkpoint_count == 0
        # started_at must be stamped at creation so the progress ETA works on
        # the direct (non-orchestrator) path.
        assert job.started_at is not None
        session.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_photo_increments_counter(self, job_queue):
        """Verify processing a photo increments the processed_photos counter."""
        job_id = "test_job_002"
        job_queue.create_job(job_id, 10)
        self._seed_photos(job_queue, 3)

        for i in range(3):
            result = await job_queue.process_photo(job_id, photo_id=i + 1)
            assert result is True

        assert job_queue.active_jobs[job_id]["processed_photos"] == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_checkpoint_saved_after_5_photos(self, job_queue):
        """Verify checkpoint is saved to database after processing 5 photos."""
        job_id = "test_job_003"
        job_queue.create_job(job_id, 10)
        self._seed_photos(job_queue, 5)

        for i in range(5):
            await job_queue.process_photo(job_id, photo_id=i + 1)

        session = job_queue.SessionLocal()
        job = session.query(JobQueue).filter(JobQueue.job_id == job_id).first()
        assert job.processed_photos == 5
        assert job.checkpoint_count == 1
        assert job.last_checkpoint_at is not None
        session.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_checkpoints(self, job_queue):
        """Verify multiple checkpoints are saved correctly for 10+ photos."""
        job_id = "test_job_004"
        job_queue.create_job(job_id, 15)
        self._seed_photos(job_queue, 12)

        for i in range(12):
            await job_queue.process_photo(job_id, photo_id=i + 1)

        session = job_queue.SessionLocal()
        job = session.query(JobQueue).filter(JobQueue.job_id == job_id).first()
        assert job.processed_photos == 12
        assert job.checkpoint_count == 2  # 12 // 5 = 2
        session.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_job_auto_completes_when_all_photos_processed(self, job_queue):
        """Regression: the direct HTTP path fires process_photo tasks
        fire-and-forget with no explicit complete_job call. The queue must
        mark the job completed once the last photo is done — otherwise it
        hangs in 'processing' forever and the progress bar never finishes."""
        job_id = "test_job_complete"
        total = 7  # deliberately not a multiple of CHECKPOINT_INTERVAL
        job_queue.create_job(job_id, total)
        self._seed_photos(job_queue, total)

        for i in range(total):
            await job_queue.process_photo(job_id, photo_id=i + 1)

        # Removed from active tracking and marked completed in the DB.
        assert job_id not in job_queue.active_jobs
        session = job_queue.SessionLocal()
        job = session.query(JobQueue).filter(JobQueue.job_id == job_id).first()
        assert job.status == "completed"
        assert job.completed_at is not None
        # Final partial batch (7 is not a multiple of 5) is flushed exactly.
        assert job.processed_photos == total
        session.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_job_completes_even_with_failed_photos(self, job_queue):
        """A photo whose file/row is missing fails permanently. The job must
        still reach completion rather than hanging because one photo never
        'succeeds'."""
        job_id = "test_job_partial"
        job_queue.create_job(job_id, 3)
        self._seed_photos(job_queue, 2)  # only photos 1 and 2 exist; 3 is missing

        assert await job_queue.process_photo(job_id, 1) is True
        assert await job_queue.process_photo(job_id, 2) is True
        # Photo 3 has no DB row → permanent failure, but still counts as handled.
        assert await job_queue.process_photo(job_id, 3) is False

        assert job_id not in job_queue.active_jobs
        session = job_queue.SessionLocal()
        job = session.query(JobQueue).filter(JobQueue.job_id == job_id).first()
        assert job.status == "completed"
        session.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_complete_job_success(self, job_queue):
        """Verify job completion marks status as completed and clears active job."""
        job_id = "test_job_005"
        job_queue.create_job(job_id, 5)

        result = await job_queue.complete_job(job_id, success=True)
        assert result is True

        session = job_queue.SessionLocal()
        job = session.query(JobQueue).filter(JobQueue.job_id == job_id).first()
        assert job.status == "completed"
        assert job.completed_at is not None
        session.close()

        assert job_id not in job_queue.active_jobs

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_complete_job_failure(self, job_queue):
        """Verify job failure marks status as failed with error message."""
        job_id = "test_job_006"
        job_queue.create_job(job_id, 5)

        error_msg = "Processing failed: invalid image format"
        result = await job_queue.complete_job(job_id, success=False, error_message=error_msg)
        assert result is True

        session = job_queue.SessionLocal()
        job = session.query(JobQueue).filter(JobQueue.job_id == job_id).first()
        assert job.status == "failed"
        assert job.error_message == error_msg
        session.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_recover_from_checkpoint(self, job_queue):
        """Verify state recovery restores incomplete job from last checkpoint."""
        job_id = "test_job_007"
        job_queue.create_job(job_id, 15)
        self._seed_photos(job_queue, 7)

        for i in range(7):
            await job_queue.process_photo(job_id, photo_id=i + 1)

        # New manager instance (simulating restart) — reopen the same DB file.
        with patch("app.job_queue.EmbeddingGenerator"), \
             patch("app.job_queue.MetadataExtractor"), \
             patch("app.job_queue.QdrantClient"):
            new_manager = JobQueueManager(database_url=job_queue.database_url)

        recovered_job_id = await new_manager.recover_from_checkpoint()
        assert recovered_job_id == job_id
        assert job_id in new_manager.active_jobs
        assert new_manager.active_jobs[job_id]["processed_photos"] == 7
        assert new_manager.active_jobs[job_id]["checkpoint_count"] == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_status(self, job_queue):
        """Verify queue status returns correct job counts and statistics."""
        job_queue.create_job("job_001", 5)
        job_queue.create_job("job_002", 10)
        await job_queue.complete_job("job_001", success=True)

        status = await job_queue.get_status()
        assert status["total_jobs"] == 2
        assert status["completed_jobs"] == 1
        assert status["pending_jobs"] == 1
        assert status["checkpoint_interval"] == 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_incomplete_job_recovery(self, job_queue):
        """Verify recovery returns None when no incomplete jobs exist."""
        job_id = "test_job_008"
        job_queue.create_job(job_id, 5)
        await job_queue.complete_job(job_id, success=True)

        recovered_job_id = await job_queue.recover_from_checkpoint()
        assert recovered_job_id is None
