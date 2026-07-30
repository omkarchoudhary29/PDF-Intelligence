import hashlib
import logging
import os
from pathlib import Path
import re
import shutil
from typing import List, Optional, Tuple
import uuid

from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.database.models import Document, IngestionJob, Workspace
from backend.schemas.document import (
    DocumentBatchUploadResponse,
    DocumentResponse,
    DocumentUploadItemResult,
)
from config.settings import get_settings

logger = logging.getLogger("ekn.storage")


def sanitize_filename(filename: str) -> str:
    """Sanitize original filename to prevent path traversal and strip illegal characters."""
    if not filename:
        return f"document_{uuid.uuid4().hex[:8]}.pdf"

    # Take basename only to defeat any directory path prefix (e.g. ../ or /etc/)
    basename = Path(filename).name
    # Remove null bytes
    basename = basename.replace("\x00", "")
    # Remove dangerous characters but allow alphanumeric, spaces, dashes, dots, and underscores
    sanitized = re.sub(r'[^\w\s.-]', '_', basename).strip()
    if not sanitized or sanitized.startswith("."):
        sanitized = f"document_{uuid.uuid4().hex[:8]}.pdf"

    return sanitized


class StorageService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.storage_root = Path(self.settings.storage_local_root).resolve()
        self.upload_root = Path(self.settings.upload_dir).resolve()
        self.temp_root = Path(self.settings.temp_upload_dir).resolve()

        # Ensure base storage directories exist
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, relative_storage_key: str) -> Path:
        """Resolve a relative storage key and ensure it cannot escape storage_root."""
        resolved = (self.storage_root / relative_storage_key).resolve()
        try:
            resolved.relative_to(self.storage_root)
        except ValueError:
            raise PermissionError("Path traversal violation detected.")
        return resolved

    async def process_single_upload(
        self,
        db: Session,
        workspace: Workspace,
        upload_file: UploadFile,
        visibility_scope: str = "company",
        owner_user_id: Optional[str] = None,
        uploaded_by_user_id: Optional[str] = None,
    ) -> DocumentUploadItemResult:
        """Process an individual PDF upload with streaming, hash calculation, validation, and deduplication.

        Supports isolated company-wide vs. personal-employee document ownership scopes.
        Never reads the entire PDF into a single in-memory bytes object.
        """
        raw_filename = upload_file.filename or "unknown.pdf"
        clean_filename = sanitize_filename(raw_filename)

        scope = "personal" if visibility_scope == "personal" else "company"
        if scope == "personal":
            if not owner_user_id:
                raise ValueError("Personal document upload requires an authenticated owner_user_id.")
            owner_id = owner_user_id
            uploader_id = uploaded_by_user_id or owner_user_id
        else:
            owner_id = None
            uploader_id = uploaded_by_user_id

        # 1. Extension validation
        if not clean_filename.lower().endswith(".pdf"):
            return DocumentUploadItemResult(
                filename=clean_filename,
                status="error",
                message="Invalid file type. Only PDF documents (.pdf) are supported.",
            )

        # 2. Setup temporary file for streaming write
        temp_file_id = f"tmp_{uuid.uuid4().hex}.tmp"
        temp_file_path = self.temp_root / temp_file_id

        hasher = hashlib.sha256()
        total_bytes = 0
        is_first_chunk = True
        is_moved = False

        chunk_size = self.settings.upload_chunk_size_bytes
        max_size = self.settings.max_upload_size_bytes

        try:
            self.temp_root.mkdir(parents=True, exist_ok=True)
            with open(temp_file_path, "wb") as f_out:
                while True:
                    chunk = await upload_file.read(chunk_size)
                    if not chunk:
                        break

                    # 3. PDF Signature Validation on first chunk
                    if is_first_chunk:
                        if not chunk.startswith(b"%PDF-"):
                            return DocumentUploadItemResult(
                                filename=clean_filename,
                                status="error",
                                message="Invalid file format. File does not contain a valid PDF signature (%PDF-).",
                            )
                        is_first_chunk = False

                    total_bytes += len(chunk)

                    # 4. Size limit validation
                    if total_bytes > max_size:
                        max_mb = max_size // (1024 * 1024)
                        return DocumentUploadItemResult(
                            filename=clean_filename,
                            status="error",
                            message=f"File exceeds maximum allowed upload size of {max_mb} MB.",
                        )

                    hasher.update(chunk)
                    f_out.write(chunk)

            # Check for empty file
            if total_bytes == 0:
                return DocumentUploadItemResult(
                    filename=clean_filename,
                    status="error",
                    message="Uploaded file is empty (0 bytes).",
                )

            sha256_digest = hasher.hexdigest()

            # 5. Ownership-Aware Duplicate Detection within workspace
            dup_query = db.query(Document).filter(
                Document.workspace_id == workspace.id,
                Document.visibility_scope == scope,
            )
            if scope == "company":
                dup_query = dup_query.filter(
                    (Document.content_hash == sha256_digest) | (Document.sha256_hash == sha256_digest)
                )
                db_sha256 = sha256_digest
            else:
                dup_query = dup_query.filter(
                    Document.owner_user_id == owner_id,
                    (Document.content_hash == sha256_digest) | (Document.sha256_hash == sha256_digest),
                )
                # Compute scoped hash for SQLite unique constraint across distinct owners
                db_sha256 = hashlib.sha256(f"personal:{owner_id}:{sha256_digest}".encode()).hexdigest()

            existing_doc = dup_query.first()

            if existing_doc:
                logger.info(
                    "Duplicate content detected for workspace %s in scope %s. Matches existing document: %s (%s)",
                    workspace.id,
                    scope,
                    existing_doc.original_filename,
                    existing_doc.id,
                )
                return DocumentUploadItemResult(
                    filename=clean_filename,
                    status="duplicate",
                    document=DocumentResponse.model_validate(existing_doc),
                    message=f"Duplicate content detected in {scope} scope. Matches existing document '{existing_doc.original_filename}'.",
                )

            # 6. Finalize Storage with Scope Isolation
            if scope == "personal":
                target_dir = self.storage_root / "private" / workspace.id / owner_id
            else:
                target_dir = self.upload_root / workspace.id

            target_dir.mkdir(parents=True, exist_ok=True)

            unique_filename = f"{sha256_digest[:16]}_{uuid.uuid4().hex[:8]}.pdf"
            final_file_path = target_dir / unique_filename

            # Atomic move from temp to permanent storage
            shutil.move(str(temp_file_path), str(final_file_path))
            is_moved = True

            # Relative storage key from storage_local_root
            relative_key = str(final_file_path.relative_to(self.storage_root)).replace("\\", "/")

            # 7. Create Database Records
            doc_id = str(uuid.uuid4())
            new_document = Document(
                id=doc_id,
                workspace_id=workspace.id,
                original_filename=clean_filename,
                display_name=Path(clean_filename).stem,
                storage_key=relative_key,
                sha256_hash=db_sha256,
                content_hash=sha256_digest,
                visibility_scope=scope,
                owner_user_id=owner_id,
                uploaded_by_user_id=uploader_id,
                file_size_bytes=total_bytes,
                mime_type="application/pdf",
                processing_status="uploaded",
            )
            db.add(new_document)

            # Ingestion Job record in 'queued' status
            job = IngestionJob(
                document_id=doc_id,
                status="queued",
            )
            db.add(job)

            try:
                db.commit()
                db.refresh(new_document)
            except IntegrityError:
                # Concurrent upload duplicate conflict
                db.rollback()
                # Clean up stored file if DB insertion failed
                if final_file_path.exists():
                    final_file_path.unlink()

                existing = dup_query.first()
                if existing:
                    return DocumentUploadItemResult(
                        filename=clean_filename,
                        status="duplicate",
                        document=DocumentResponse.model_validate(existing),
                        message="Duplicate content detected during concurrent upload.",
                    )
                raise

            logger.info(
                "Document uploaded successfully: %s (id=%s, size=%d bytes, scope=%s, owner=%s)",
                clean_filename,
                new_document.id,
                total_bytes,
                scope,
                owner_id,
            )

            return DocumentUploadItemResult(
                filename=clean_filename,
                status="uploaded",
                document=DocumentResponse.model_validate(new_document),
                message="File uploaded successfully and queued for ingestion.",
            )

        except Exception as exc:
            db.rollback()
            logger.error("Failed to process upload for %s: %s", clean_filename, exc, exc_info=True)
            return DocumentUploadItemResult(
                filename=clean_filename,
                status="error",
                message=f"Upload processing failed: {str(exc)}",
            )
        finally:
            # 8. Temporary file cleanup
            if not is_moved and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except OSError as cleanup_err:
                    logger.warning("Failed to clean up temp upload file %s: %s", temp_file_path, cleanup_err)

    async def process_batch_upload(
        self,
        db: Session,
        workspace: Workspace,
        files: List[UploadFile],
        visibility_scope: str = "company",
        owner_user_id: Optional[str] = None,
        uploaded_by_user_id: Optional[str] = None,
    ) -> DocumentBatchUploadResponse:
        """Process a batch of uploaded PDF files with ownership isolation."""
        results: List[DocumentUploadItemResult] = []
        uploaded_count = 0
        duplicate_count = 0
        error_count = 0

        for upload_file in files:
            res = await self.process_single_upload(
                db=db,
                workspace=workspace,
                upload_file=upload_file,
                visibility_scope=visibility_scope,
                owner_user_id=owner_user_id,
                uploaded_by_user_id=uploaded_by_user_id,
            )
            results.append(res)
            if res.status == "uploaded":
                uploaded_count += 1
            elif res.status == "duplicate":
                duplicate_count += 1
            else:
                error_count += 1

        return DocumentBatchUploadResponse(
            total_submitted=len(files),
            total_uploaded=uploaded_count,
            total_duplicates=duplicate_count,
            total_errors=error_count,
            results=results,
        )

    def get_document_file_path(self, document: Document) -> Path:
        """Resolve physical file path for a document and verify existence."""
        file_path = self._resolve_safe_path(document.storage_key)
        if not file_path.is_file():
            raise FileNotFoundError(f"Underlying storage file for document {document.id} was not found on disk.")
        return file_path

    def delete_document(self, db: Session, workspace_id: str, document_id: str) -> bool:
        """Delete document record and remove its underlying physical file.

        Verifies that the document strictly belongs to the specified workspace.
        """
        doc = (
            db.query(Document)
            .filter(
                Document.id == document_id,
                Document.workspace_id == workspace_id,
            )
            .first()
        )
        if not doc:
            return False

        # Attempt to delete physical file
        try:
            file_path = self._resolve_safe_path(doc.storage_key)
            if file_path.exists():
                file_path.unlink()
                logger.info("Deleted physical file for document %s: %s", document_id, file_path)
        except Exception as exc:
            logger.warning("Could not delete physical storage file for document %s: %s", document_id, exc)

        # Clean up extracted pages and parent chunk files if present
        try:
            if doc.extracted_pages_path:
                pages_path = self._resolve_safe_path(doc.extracted_pages_path)
                if pages_path.exists():
                    pages_path.unlink()
            parents_file = self.storage_root / "processed" / workspace_id / f"{document_id}_parents.jsonl"
            if parents_file.exists():
                parents_file.unlink()
        except Exception as exc:
            logger.warning("Could not delete processed artifacts for document %s: %s", document_id, exc)

        # Prune knowledge graph associations and relationships
        try:
            from backend.services.knowledge_graph_service import knowledge_graph_service
            knowledge_graph_service.handle_document_deletion(db, workspace_id, document_id)
        except Exception as kg_exc:
            logger.warning("Could not prune knowledge graph for document %s: %s", document_id, kg_exc)

        # Delete database record (cascades to IngestionJob)
        db.delete(doc)
        db.commit()
        logger.info("Deleted document database record %s from workspace %s", document_id, workspace_id)
        return True


storage_service = StorageService()
