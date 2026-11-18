import argparse
import logging
import os
from pathlib import Path
import signal
import sys
import time
from typing import Optional

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError

from backend.database.connection import get_session_factory, init_db
from backend.services.ingestion_service import ingestion_service
from config.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ekn.worker")

# Global graceful shutdown flag
_shutdown_requested = False


def _update_heartbeat(storage_root: str) -> None:
    """Record worker heartbeat timestamp to disk for operational health monitoring."""
    try:
        hb_path = Path(storage_root) / ".worker_heartbeat"
        hb_path.parent.mkdir(parents=True, exist_ok=True)
        temp_hb = hb_path.with_suffix(".tmp")
        temp_hb.write_text(f"{time.time()}:{os.getpid()}", encoding="utf-8")
        temp_hb.replace(hb_path)
    except Exception:
        pass


def signal_handler(signum, frame):
    global _shutdown_requested
    logger.info("Shutdown signal received (%s). Completing active operations and exiting...", signum)
    _shutdown_requested = True


def _recover_orphaned_jobs(session_factory) -> None:
    """Reset any jobs left in 'processing' status back to 'queued' so they are not abandoned."""
    try:
        with session_factory() as session:
            from backend.database.models import Document, IngestionJob
            orphaned = session.query(IngestionJob).filter(IngestionJob.status == "processing").all()
            if orphaned:
                logger.info("Found %d orphaned processing job(s). Resetting back to queued.", len(orphaned))
                for oj in orphaned:
                    oj.status = "queued"
                    oj.started_at = None
                    doc = session.query(Document).filter(Document.id == oj.document_id).first()
                    if doc and doc.processing_status != "completed":
                        doc.processing_status = "queued"
                session.commit()
    except Exception as rec_err:
        logger.warning("Error recovering orphaned processing jobs: %s", rec_err)


def run_worker(once: bool = False, poll_interval: Optional[float] = None) -> int:
    """Run the document ingestion background worker with resilient database recovery.

    Args:
        once: If True, processes available queued jobs until the queue is empty, then exits.
        poll_interval: Seconds to sleep between polls when no jobs are queued.

    Returns:
        Number of jobs processed.
    """
    settings = get_settings()
    interval = poll_interval if poll_interval is not None else settings.worker_poll_interval_seconds

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    # Initial operational heartbeat to confirm process startup
    _update_heartbeat(settings.storage_local_root)

    # 1. Startup phase: Wait for database readiness with capped exponential retry logic
    session_factory = None
    if settings.effective_database_url:
        max_retries = 20
        backoff = 1.0
        for attempt in range(1, max_retries + 1):
            if _shutdown_requested:
                return 0
            try:
                init_db()
                factory = get_session_factory()
                if factory is not None:
                    with factory() as test_session:
                        test_session.execute(text("SELECT 1"))
                    session_factory = factory
                    logger.info("Ingestion worker database connection established on attempt %d.", attempt)
                    break
            except Exception as exc:
                logger.warning(
                    "Ingestion worker waiting for database readiness (attempt %d/%d, retrying in %.1fs): %s",
                    attempt,
                    max_retries,
                    backoff,
                    exc,
                )
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 15.0)
    else:
        logger.error("Database is not configured. Ingestion worker cannot start.")
        return 0

    if session_factory is None:
        logger.error("Database is not accessible after retries. Ingestion worker cannot start.")
        return 0

    # 2. Recover orphaned processing jobs
    _recover_orphaned_jobs(session_factory)

    logger.info("EKN Ingestion Worker started (mode=%s, interval=%.1fs)", "once" if once else "continuous", interval)
    jobs_processed = 0

    # 3. Main processing loop with disconnection resilience
    while not _shutdown_requested:
        _update_heartbeat(settings.storage_local_root)

        try:
            with session_factory() as session:
                job = ingestion_service.claim_next_job(session)

                if job:
                    logger.info("Processing ingestion job %s (document=%s)", job.id, job.document_id)
                    try:
                        success = ingestion_service.execute_job(session, job)
                    except Exception as job_err:
                        logger.error("Ingestion job %s encountered unhandled error: %s", job.id, job_err, exc_info=True)
                        success = False
                    jobs_processed += 1
                    logger.info("Finished job %s with result: %s", job.id, "SUCCESS" if success else "FAILED")

                    if once:
                        continue
                else:
                    if once:
                        logger.info("No queued jobs remaining. Worker exiting (--once mode).")
                        break

            if not _shutdown_requested and not once:
                time.sleep(interval)

        except (SQLAlchemyError, OperationalError, DBAPIError) as db_err:
            logger.warning(
                "Database connectivity lost during worker loop: %s. Entering reconnection backoff...",
                db_err,
            )
            reconnect_backoff = 2.0
            reconnected = False
            while not _shutdown_requested and not reconnected:
                time.sleep(reconnect_backoff)
                try:
                    with session_factory() as test_session:
                        test_session.execute(text("SELECT 1"))
                    reconnected = True
                    logger.info("Database connectivity restored. Resuming ingestion processing.")
                    _recover_orphaned_jobs(session_factory)
                except Exception as retry_err:
                    reconnect_backoff = min(reconnect_backoff * 1.5, 30.0)
                    logger.debug("Database still unreachable (%s). Retrying in %.1fs...", retry_err, reconnect_backoff)
            if once and not reconnected:
                break

        except Exception as loop_err:
            logger.error("Unexpected error in worker loop: %s", loop_err, exc_info=True)
            if once:
                break
            time.sleep(interval)

    logger.info("Ingestion Worker terminated. Total jobs processed: %d", jobs_processed)
    return jobs_processed


def backfill_unindexed_documents() -> int:
    """Find all completed extraction documents that lack a completed vector index and index them."""
    session_factory = get_session_factory()
    if session_factory is None:
        logger.error("Database not configured for backfill.")
        return 0

    with session_factory() as session:
        from backend.database.models import Document, DocumentIndex

        docs = (
            session.query(Document)
            .filter(Document.processing_status == "completed")
            .all()
        )
        backfilled = 0
        for doc in docs:
            idx = session.query(DocumentIndex).filter(DocumentIndex.document_id == doc.id).first()
            if not idx or idx.status != "completed":
                logger.info("Backfilling vector index for document: %s (%s)", doc.original_filename, doc.id)
                try:
                    ingestion_service.index_document(session, doc, force=True)
                    backfilled += 1
                except Exception as exc:
                    logger.error("Failed to backfill document %s: %s", doc.id, exc)

        logger.info("Backfill complete. %d documents indexed.", backfilled)
        return backfilled


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise Knowledge Navigator — Document Ingestion Worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process all currently queued jobs and exit",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Polling interval in seconds between empty queue checks",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill vector indexes for existing extracted documents",
    )

    args = parser.parse_args()
    if args.backfill:
        backfill_unindexed_documents()
    else:
        run_worker(once=args.once, poll_interval=args.interval)


if __name__ == "__main__":
    main()
