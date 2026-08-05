from datetime import datetime
import json
import logging
from pathlib import Path
from typing import List, Optional

from sqlalchemy.orm import Session

from backend.database.models import Document, DocumentIndex, IngestionJob
from backend.schemas.document import PageExtractionRecord
from backend.schemas.indexing import (
    DocumentIndexResponse,
    DocumentIndexStatusResponse,
)
from backend.services.bm25_service import bm25_service
from backend.services.chunking_service import chunking_service
from backend.services.embedding_service import embedding_service
from backend.services.pdf_extractor import pdf_extractor
from backend.services.storage_service import storage_service
from backend.services.vector_index_service import vector_index_service
from config.settings import get_settings

logger = logging.getLogger("ekn.ingestion")


class IngestionService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.storage_root = Path(self.settings.storage_local_root).resolve()

    def claim_next_job(self, db: Session) -> Optional[IngestionJob]:
        """Atomically claim the next queued ingestion job.

        Transitions both the IngestionJob and Document to 'processing' status.
        """
        job = (
            db.query(IngestionJob)
            .filter(IngestionJob.status == "queued")
            .order_by(IngestionJob.created_at.asc())
            .first()
        )

        if not job:
            return None

        job.status = "processing"
        job.started_at = datetime.now()

        doc = db.query(Document).filter(Document.id == job.document_id).first()
        if doc:
            doc.processing_status = "processing"
            doc.error_message = None

        db.commit()
        db.refresh(job)
        logger.info("Claimed ingestion job %s for document %s", job.id, job.document_id)
        return job

    def execute_job(self, db: Session, job: IngestionJob) -> bool:
        """Execute the full ingestion pipeline: PDF extraction -> chunking -> vector indexing.

        Handles extraction and indexing errors gracefully without terminating the worker daemon.
        """
        doc = db.query(Document).filter(Document.id == job.document_id).first()
        if not doc:
            error_msg = f"Document {job.document_id} no longer exists in database."
            logger.error(error_msg)
            job.status = "failed"
            job.error_message = error_msg
            job.completed_at = datetime.now()
            db.commit()
            return False

        try:
            logger.info("Beginning PDF extraction for: %s (id=%s)", doc.original_filename, doc.id)
            doc_file_path = storage_service.get_document_file_path(doc)

            # Step 1: Run PDF extraction engine
            result = pdf_extractor.extract_document(
                pdf_path=doc_file_path,
                document_id=doc.id,
                workspace_id=doc.workspace_id,
                original_filename=doc.original_filename,
            )

            doc.total_pages = result.total_pages
            doc.extracted_pages_path = result.relative_output_path
            doc.processing_status = "completed"
            doc.error_message = None
            db.commit()

            # Step 2: Load extracted pages and execute chunking + vector indexing
            pages = self.get_extracted_pages(db, doc.workspace_id, doc.id)
            logger.info("Proceeding to chunking & indexing for document %s (%d pages)", doc.original_filename, len(pages))
            self.index_document(db=db, doc=doc, pages=pages, force=False)

            job.status = "completed"
            job.completed_at = datetime.now()
            job.error_message = None
            db.commit()

            logger.info(
                "Ingestion & vector indexing completed for document %s: %d pages processed",
                doc.original_filename,
                result.total_pages,
            )
            return True

        except Exception as exc:
            db.rollback()
            err_msg = str(exc)
            logger.error(
                "Failed processing document %s (%s): %s",
                doc.original_filename,
                doc.id,
                err_msg,
                exc_info=True,
            )

            # Check if extraction succeeded but indexing failed
            if doc.extracted_pages_path:
                doc.processing_status = "completed"
            else:
                doc.processing_status = "failed"
            doc.error_message = err_msg

            job.status = "failed"
            job.completed_at = datetime.now()
            job.error_message = err_msg

            db.commit()
            return False
        finally:
            import gc
            gc.collect()

    def index_document(
        self,
        db: Session,
        doc: Document,
        pages: Optional[List[PageExtractionRecord]] = None,
        force: bool = False,
    ) -> DocumentIndex:
        """Perform hierarchical parent-child chunking, embedding generation, and ChromaDB vector indexing.

        Supports incremental indexing: if the file hash and embedding model have not changed,
        the existing completed index is reused unless force=True.
        """
        # Fetch or initialize DocumentIndex record
        doc_index = db.query(DocumentIndex).filter(DocumentIndex.document_id == doc.id).first()
        if not doc_index:
            doc_index = DocumentIndex(
                document_id=doc.id,
                status="pending",
                embedding_model=self.settings.embedding_model,
                embedding_dimension=self.settings.embedding_dimension,
            )
            db.add(doc_index)
            db.commit()
            db.refresh(doc_index)

        # Incremental check: reuse valid index if unchanged and not forced
        if (
            not force
            and doc_index.status == "completed"
            and doc_index.indexed_file_hash == doc.sha256_hash
            and doc_index.embedding_model == self.settings.embedding_model
        ):
            logger.info(
                "Document %s already indexed with identical hash (%s) and model (%s). Skipping re-embedding.",
                doc.id,
                doc.sha256_hash[:10],
                doc_index.embedding_model,
            )
            return doc_index

        # Mark indexing in progress
        doc_index.status = "indexing"
        doc_index.started_at = datetime.now()
        doc_index.error_message = None
        db.commit()

        try:
            # Load extracted pages if not provided
            if pages is None:
                pages = self.get_extracted_pages(db, doc.workspace_id, doc.id)

            if not pages:
                raise ValueError(
                    f"No extracted page records found for document '{doc.id}'. Ensure PDF extraction succeeded first."
                )

            # Step 1: Hierarchical chunking
            doc_version = getattr(doc, "version_number", 1) or 1
            parents, children = chunking_service.chunk_document(
                pages=pages,
                document_id=doc.id,
                workspace_id=doc.workspace_id,
                original_filename=doc.original_filename,
                file_hash=doc.sha256_hash,
                document_version=doc_version,
                visibility_scope=getattr(doc, "visibility_scope", "company") or "company",
                owner_user_id=getattr(doc, "owner_user_id", None),
            )

            # Step 2: Persist parent chunks to JSONL
            parent_storage_path = chunking_service.save_parent_chunks(
                parents=parents,
                workspace_id=doc.workspace_id,
                document_id=doc.id,
            )

            total_indexed_children = 0

            # Step 3: Embed and index child chunks into ChromaDB
            if children:
                child_texts = [c.text for c in children]
                embeddings = embedding_service.generate_embeddings(child_texts)

                total_indexed_children = vector_index_service.index_child_chunks(
                    chunks=children,
                    embeddings=embeddings,
                    document_id=doc.id,
                )

                # Step 3.5: Update workspace BM25 sparse keyword corpus
                try:
                    bm25_service.add_or_update_document_chunks(
                        workspace_id=doc.workspace_id,
                        document_id=doc.id,
                        chunks=children,
                    )
                except Exception as bm25_err:
                    logger.warning(
                        "BM25 index update encountered an issue for doc %s: %s",
                        doc.id,
                        bm25_err,
                    )

            # Step 4: Finalize DocumentIndex record
            doc_index.status = "completed"
            doc_index.total_parents = len(parents)
            doc_index.total_children = total_indexed_children
            doc_index.embedding_model = self.settings.embedding_model
            doc_index.embedding_dimension = embedding_service.dimension
            doc_index.parent_storage_path = parent_storage_path
            doc_index.indexed_file_hash = doc.sha256_hash
            doc_index.completed_at = datetime.now()
            doc_index.error_message = None
            db.commit()
            db.refresh(doc_index)

            logger.info(
                "DocumentIndex finalized for document %s: %d parents, %d child vectors in ChromaDB",
                doc.id,
                len(parents),
                total_indexed_children,
            )

            # Incremental Knowledge Graph Update for newly indexed document
            try:
                from backend.services.knowledge_graph_service import knowledge_graph_service
                knowledge_graph_service.build_workspace_graph(
                    db=db,
                    workspace_id=doc.workspace_id,
                    document_ids=[doc.id],
                )
            except Exception as graph_err:
                logger.warning("Incremental graph update for document %s encountered error: %s", doc.id, graph_err)

            return doc_index

        except Exception as exc:
            db.rollback()
            err_msg = str(exc)
            logger.error("Indexing failed for document %s: %s", doc.id, err_msg, exc_info=True)
            doc_index.status = "failed"
            doc_index.error_message = err_msg
            doc_index.completed_at = datetime.now()
            db.commit()
            raise

    def reindex_document(
        self,
        db: Session,
        workspace_id: str,
        document_id: str,
        force: bool = True,
    ) -> DocumentIndex:
        """Trigger re-indexing or backfilling for an existing extracted document."""
        doc = (
            db.query(Document)
            .filter(
                Document.id == document_id,
                Document.workspace_id == workspace_id,
            )
            .first()
        )
        if not doc:
            raise ValueError(f"Document '{document_id}' was not found in workspace '{workspace_id}'.")

        pages = self.get_extracted_pages(db, workspace_id, document_id)
        if not pages:
            raise ValueError(
                f"Document '{document_id}' has not been extracted yet. Please run extraction before indexing."
            )

        return self.index_document(db=db, doc=doc, pages=pages, force=force)

    def get_document_index_status(
        self,
        db: Session,
        workspace_id: str,
        document_id: str,
    ) -> DocumentIndexStatusResponse:
        """Retrieve comprehensive extraction and vector index status for a document."""
        doc = (
            db.query(Document)
            .filter(
                Document.id == document_id,
                Document.workspace_id == workspace_id,
            )
            .first()
        )
        if not doc:
            raise ValueError(f"Document '{document_id}' was not found in workspace '{workspace_id}'.")

        doc_index = db.query(DocumentIndex).filter(DocumentIndex.document_id == document_id).first()

        index_response = None
        if doc_index:
            index_response = DocumentIndexResponse.model_validate(doc_index)

        is_search_ready = bool(
            doc.processing_status == "completed"
            and doc_index
            and doc_index.status == "completed"
        )

        return DocumentIndexStatusResponse(
            document_id=doc.id,
            workspace_id=doc.workspace_id,
            original_filename=doc.original_filename,
            extraction_status=doc.processing_status,
            total_pages=doc.total_pages,
            index=index_response,
            is_search_ready=is_search_ready,
        )

    def retry_document_ingestion(self, db: Session, workspace_id: str, document_id: str) -> IngestionJob:
        """Reset failed document and queue an ingestion job for reprocessing."""
        doc = (
            db.query(Document)
            .filter(
                Document.id == document_id,
                Document.workspace_id == workspace_id,
            )
            .first()
        )
        if not doc:
            raise ValueError(f"Document '{document_id}' was not found in workspace '{workspace_id}'.")

        doc.processing_status = "queued"
        doc.error_message = None

        job = (
            db.query(IngestionJob)
            .filter(IngestionJob.document_id == document_id)
            .order_by(IngestionJob.created_at.desc())
            .first()
        )

        if job and job.status in ("failed", "completed"):
            job.status = "queued"
            job.error_message = None
            job.started_at = None
            job.completed_at = None
        else:
            job = IngestionJob(
                document_id=document_id,
                status="queued",
            )
            db.add(job)

        db.commit()
        db.refresh(job)
        logger.info("Queued document %s for re-ingestion (job=%s)", document_id, job.id)
        return job

    def get_extracted_pages(self, db: Session, workspace_id: str, document_id: str) -> List[PageExtractionRecord]:
        """Retrieve stored JSON Lines extraction records for a document."""
        doc = (
            db.query(Document)
            .filter(
                Document.id == document_id,
                Document.workspace_id == workspace_id,
            )
            .first()
        )
        if not doc:
            raise ValueError(f"Document '{document_id}' was not found in workspace '{workspace_id}'.")

        if not doc.extracted_pages_path:
            return []

        jsonl_path = (self.storage_root / doc.extracted_pages_path).resolve()
        if not jsonl_path.is_file():
            return []

        pages: List[PageExtractionRecord] = []
        with open(jsonl_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                line_str = line.strip()
                if line_str:
                    data = json.loads(line_str)
                    pages.append(PageExtractionRecord(**data))

        return pages


# Global singleton instance
ingestion_service = IngestionService()
