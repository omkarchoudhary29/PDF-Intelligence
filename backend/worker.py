import argparse
import logging
import os
from pathlib import Path
import signal
import sys
import time
from typing import Optional

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
    logger.info("Shutdown signal received (%s). Completing active job and exiting...", signum)
    _shutdown_requested = True


def run_worker(once: bool = False, poll_interval: Optional[float] = None) -> int:
    """Run the document ingestion background worker.

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
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize database tables if needed
    try:
        init_db()
    except Exception as exc:
        logger.warning("Could not auto-initialize database on worker startup: %s", exc)

    session_factory = get_session_factory()
    if session_factory is None:
        logger.error("Database is not configured. Ingestion worker cannot start.")
        return 0

    # Recover any orphaned processing jobs left by previous unexpected interruptions
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

    logger.info("EKN Ingestion Worker started (mode=%s, interval=%.1fs)", "once" if once else "continuous", interval)
    jobs_processed = 0

    while not _shutdown_requested:
        _update_heartbeat(settings.storage_local_root)
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

                # If running in --once mode, continue draining queue without sleeping
                if once:
                    continue
            else:
                if once:
                    logger.info("No queued jobs remaining. Worker exiting (--once mode).")
                    break

        if not _shutdown_requested and not once:
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
