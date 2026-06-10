"""Async job queue manager for photo processing with checkpoint persistence and state recovery."""
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List, Optional, Dict
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session
from app.models import JobQueue, Base, Photo, ProcessingState, Embedding
from app.embedding_generator import EmbeddingGenerator
from app.metadata_extractor import MetadataExtractor
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid as _uuid

logger = logging.getLogger(__name__)


class JobQueueManager:
    """Manages async photo processing jobs with checkpoint persistence every 5 photos."""
    
    CHECKPOINT_INTERVAL = 5  # Save checkpoint after processing 5 photos
    
    def __init__(self, database_url: str = None):
        """Initialize job queue manager with database connection.
        
        Args:
            database_url: PostgreSQL connection URL (defaults to DATABASE_URL env var)
        """
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/app_db"
        )
        from app.database import make_engine
        self.engine = make_engine(self.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.active_jobs: Dict[str, Dict] = {}  # In-memory tracking of active jobs
        self.embedding_generator = EmbeddingGenerator()
        self.metadata_extractor = MetadataExtractor()
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333")
        )
        self.qdrant_collection = "embeddings"
        # Limit concurrent embedding generation so we don't exhaust CPU
        # threads or DB connections (each task holds a session open while
        # the model runs for tens of seconds on CPU).
        self._processing_semaphore = asyncio.Semaphore(2)
        self._cancelled_jobs: set = set()  # Job IDs that have been stopped
    
    def create_job(self, job_id: str, total_photos: int) -> bool:
        """Create a new processing job in the queue.
        
        Args:
            job_id: Unique job identifier
            total_photos: Total number of photos to process
        
        Returns:
            True if job created successfully, False otherwise
        """
        try:
            session = self.SessionLocal()
            job = JobQueue(
                job_id=job_id,
                status="pending",
                total_photos=total_photos,
                processed_photos=0,
                checkpoint_count=0,
                # Stamp started_at at creation so /ws/progress can compute an
                # ETA. The live HTTP path (rescan / process-pending) fires
                # process_photo tasks directly without the Orchestrator, so if
                # we don't set it here it stays NULL and the ETA is forever
                # None — see get_progress, which requires started_at.
                started_at=datetime.utcnow(),
            )
            session.add(job)
            session.commit()
            session.close()
            self.active_jobs[job_id] = {
                "status": "pending",
                "processed_photos": 0,
                "checkpoint_count": 0,
                # Total + finished let process_photo detect when the whole
                # batch is done and mark the job completed. "finished" counts
                # every terminal outcome (success, permanent failure, missing
                # file) so a batch with some failures still completes instead
                # of hanging in "processing" forever.
                "total_photos": total_photos,
                "finished": 0,
            }
            return True
        except Exception as e:
            logger.error("Error creating job %s: %s", job_id, e, exc_info=True)
            return False
    
    async def _record_finished(self, job_id: str) -> None:
        """Mark one photo as terminally handled (success OR permanent failure)
        and, once every photo in the batch has been handled, flush the final
        count and mark the job completed.

        Without this, the live HTTP path — which fires process_photo tasks
        fire-and-forget — would never call complete_job, leaving the job
        stuck in "processing" forever: the progress bar never reaches its
        terminal state and the WebSocket polls indefinitely. The Orchestrator
        path already completes jobs explicitly; this makes the direct path
        behave the same."""
        job = self.active_jobs.get(job_id)
        if job is None:
            return  # cancelled / already completed
        job["finished"] += 1
        if job["finished"] >= job.get("total_photos", 0):
            # Persist the true processed count (the last checkpoint may have
            # missed the final partial batch) before flipping to completed.
            await self.save_checkpoint(job_id)
            await self.complete_job(job_id, success=True)

    async def process_photo(self, job_id: str, photo_id: int) -> bool:
        """Process a single photo: extract metadata and generate embedding.
        
        Args:
            job_id: Job identifier
            photo_id: Photo ID to process
        
        Returns:
            True if photo processed successfully, False otherwise
        """
        # Check if this job has been cancelled before acquiring semaphore
        if job_id in self._cancelled_jobs:
            return False
        await self._processing_semaphore.acquire()
        try:
            # Re-check after acquiring semaphore (may have waited)
            if job_id in self._cancelled_jobs:
                return False
            # Open session lazily and read just the file path so we don't
            # hold a transaction open while the model is running.
            session = self.SessionLocal()
            photo = session.query(Photo).filter(Photo.id == photo_id).first()
            file_path = photo.file_path if photo else None
            session.close()
            if not file_path:
                # Photo row gone — nothing to retry. Count it as handled so
                # the batch can still reach completion.
                await self._record_finished(job_id)
                return False

            # Extract metadata from photo file
            metadata = await self.metadata_extractor.extract(file_path)
            
            # Generate embedding for photo
            embedding_vector = await self.embedding_generator.generate(file_path)

            # Persist embedding: upsert vector in Qdrant, store pointer row in Postgres
            point_id = str(_uuid.uuid4())
            await asyncio.to_thread(
                self.qdrant_client.upsert,
                collection_name=self.qdrant_collection,
                points=[PointStruct(id=point_id, vector=embedding_vector, payload={"photo_id": photo_id})],
            )
            embedding_row = Embedding(
                photo_id=photo_id,
                embedding_model="dinov2_vits14",
                vector_dimension=len(embedding_vector),
                qdrant_point_id=point_id,
            )
            # Open a fresh short-lived session to persist results
            session = self.SessionLocal()
            session.add(embedding_row)

            # Update processing state
            processing_state = session.query(ProcessingState).filter(
                ProcessingState.photo_id == photo_id
            ).first()
            if processing_state:
                processing_state.extraction_status = "completed"
                processing_state.embedding_status = "completed"
                processing_state.status = "completed"
                processing_state.completed_at = datetime.utcnow()
                processing_state.updated_at = datetime.utcnow()
            
            session.commit()
            session.close()

            # Signal that a new embedding was added (debounced index update).
            # Passing the photo_id lets the index fold it in incrementally
            # instead of recomputing the whole matrix — see
            # notify_embeddings_changed / _incremental_add_sync.
            from app.main import notify_embeddings_changed
            notify_embeddings_changed(photo_id)

            # Update in-memory tracking
            if job_id in self.active_jobs:
                self.active_jobs[job_id]["processed_photos"] += 1
                # Persist progress after every photo so the live progress bar
                # (get_progress reads job.processed_photos from the DB)
                # advances smoothly instead of jumping in steps of
                # CHECKPOINT_INTERVAL — and so the final, non-multiple-of-5
                # count is never lost. save_checkpoint still derives
                # checkpoint_count / last_checkpoint_at from the running total.
                await self.save_checkpoint(job_id)

            # Terminal success — may trigger job completion if it was the
            # last photo in the batch.
            await self._record_finished(job_id)
            return True
        except Exception as e:
            logger.error(
                "Processing failure for photo %d in job %s: %s",
                photo_id, job_id, e, exc_info=True,
            )
            # Mark the photo's processing state as failed so the UI can report it
            try:
                session = self.SessionLocal()
                processing_state = session.query(ProcessingState).filter(
                    ProcessingState.photo_id == photo_id
                ).first()
                if processing_state:
                    processing_state.status = "failed"
                    processing_state.error_message = str(e)[:500]
                    processing_state.updated_at = datetime.utcnow()
                session.commit()
                session.close()
            except Exception as inner_err:
                logger.error("Failed to record error state for photo %d: %s", photo_id, inner_err)
            # Permanent failure for this photo — still counts toward batch
            # completion so the job doesn't hang in "processing".
            await self._record_finished(job_id)
            return False
        finally:
            self._processing_semaphore.release()
    
    async def cancel_all_jobs(self) -> int:
        """Cancel all active jobs. Pending tasks will skip processing.

        Returns:
            Number of jobs cancelled.
        """
        cancelled = 0
        for job_id in list(self.active_jobs.keys()):
            self._cancelled_jobs.add(job_id)
            await self.complete_job(job_id, success=False, error_message="Cancelled by user")
            cancelled += 1
        return cancelled

    async def get_progress(self, job_id: str) -> dict:
        """Get current progress for a job including percentage and ETA.
        
        Args:
            job_id: Job identifier
        
        Returns:
            Dictionary with progress data: percentage, processed_photos, total_photos, eta_seconds
        """
        if job_id not in self.active_jobs:
            return {"status": "not_found"}
        
        try:
            session = self.SessionLocal()
            job = session.query(JobQueue).filter(JobQueue.job_id == job_id).first()
            session.close()
            
            if not job:
                return {"status": "not_found"}
            
            processed = job.processed_photos
            total = job.total_photos
            
            # Calculate percentage
            percentage = int((processed / total * 100)) if total > 0 else 0
            
            # Calculate ETA in seconds
            eta_seconds = None
            if job.started_at and processed > 0:
                elapsed = (datetime.utcnow() - job.started_at).total_seconds()
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = total - processed
                eta_seconds = int(remaining / rate) if rate > 0 else None
            
            return {
                "job_id": job_id,
                "status": job.status,
                "percentage": percentage,
                "processed_photos": processed,
                "total_photos": total,
                "eta_seconds": eta_seconds
            }
        except Exception as e:
            logger.error("Error getting progress for job %s: %s", job_id, e, exc_info=True)
            return {"error": str(e)}
    
    async def save_checkpoint(self, job_id: str) -> bool:
        """Save checkpoint after processing CHECKPOINT_INTERVAL photos.
        
        Args:
            job_id: Job identifier
        
        Returns:
            True if checkpoint saved successfully, False otherwise
        """
        try:
            session = self.SessionLocal()
            job = session.query(JobQueue).filter(JobQueue.job_id == job_id).first()
            if job:
                if job_id in self.active_jobs:
                    job.processed_photos = self.active_jobs[job_id]["processed_photos"]
                    job.checkpoint_count = job.processed_photos // self.CHECKPOINT_INTERVAL
                    job.last_checkpoint_at = datetime.utcnow()
                    job.updated_at = datetime.utcnow()
                    session.commit()
                    print(f"Checkpoint saved for job {job_id}: {job.processed_photos} photos processed")
            session.close()
            return True
        except Exception as e:
            print(f"Error saving checkpoint for job {job_id}: {e}")
            return False
    
    async def complete_job(self, job_id: str, success: bool = True, error_message: str = None) -> bool:
        """Mark job as completed or failed.
        
        Args:
            job_id: Job identifier
            success: True if job completed successfully, False if failed
            error_message: Error message if job failed
        
        Returns:
            True if job status updated successfully, False otherwise
        """
        try:
            session = self.SessionLocal()
            job = session.query(JobQueue).filter(JobQueue.job_id == job_id).first()
            if job:
                job.status = "completed" if success else "failed"
                job.completed_at = datetime.utcnow()
                job.updated_at = datetime.utcnow()
                if error_message:
                    job.error_message = error_message
                session.commit()
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
            session.close()
            return True
        except Exception as e:
            print(f"Error completing job {job_id}: {e}")
            return False
    
    async def recover_from_checkpoint(self) -> Optional[str]:
        """Recover incomplete job from last checkpoint on application restart.
        
        Returns:
            Job ID of recovered job, or None if no incomplete job found
        """
        try:
            session = self.SessionLocal()
            # Find most recent incomplete job
            incomplete_job = session.query(JobQueue).filter(
                JobQueue.status.in_(["pending", "processing"])
            ).order_by(JobQueue.created_at.desc()).first()
            
            if incomplete_job:
                # Read every attribute we still need into locals BEFORE the
                # commit. session.commit() expires the instance's attributes
                # (expire_on_commit defaults to True) and session.close()
                # detaches it, so touching incomplete_job.job_id afterwards
                # raises DetachedInstanceError — which the except below would
                # swallow, making recovery silently return None and the
                # "Resume processing" feature never actually resume.
                recovered_id = incomplete_job.job_id
                recovered_processed = incomplete_job.processed_photos
                recovered_checkpoint = incomplete_job.checkpoint_count
                recovered_total = incomplete_job.total_photos
                print(f"Recovering job {recovered_id}: ",
                      f"{recovered_processed}/{recovered_total} photos processed")
                # Restore job state to in-memory tracking
                self.active_jobs[recovered_id] = {
                    "status": "processing",
                    "processed_photos": recovered_processed,
                    "checkpoint_count": recovered_checkpoint,
                    "total_photos": recovered_total,
                    "finished": recovered_processed,
                }
                # Update job status to processing
                incomplete_job.status = "processing"
                incomplete_job.started_at = datetime.utcnow()
                incomplete_job.updated_at = datetime.utcnow()
                session.commit()
                session.close()
                return recovered_id
            session.close()
            return None
        except Exception as e:
            print(f"Error recovering from checkpoint: {e}")
            return None
    
    async def get_status(self) -> Dict:
        """Get current status of all jobs and queue statistics.
        
        Returns:
            Dictionary with queue status information
        """
        try:
            session = self.SessionLocal()
            total_jobs = session.query(JobQueue).count()
            pending_jobs = session.query(JobQueue).filter(JobQueue.status == "pending").count()
            processing_jobs = session.query(JobQueue).filter(JobQueue.status == "processing").count()
            completed_jobs = session.query(JobQueue).filter(JobQueue.status == "completed").count()
            failed_jobs = session.query(JobQueue).filter(JobQueue.status == "failed").count()
            
            session.close()
            return {
                "total_jobs": total_jobs,
                "pending_jobs": pending_jobs,
                "processing_jobs": processing_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "active_jobs": list(self.active_jobs.keys()),
                "checkpoint_interval": self.CHECKPOINT_INTERVAL
            }
        except Exception as e:
            print(f"Error getting queue status: {e}")
            return {"error": str(e)}
