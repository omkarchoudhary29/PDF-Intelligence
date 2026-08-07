import hashlib
import logging
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database.models import (
    Document,
    DocumentFamily,
    IngestionJob,
    Workspace,
)
from backend.schemas.document import DocumentResponse, DocumentUploadItemResult
from backend.services.audit_service import audit_service
from backend.services.storage_service import sanitize_filename, storage_service
from config.settings import get_settings

logger = logging.getLogger("ekn.versioning")


def normalize_canonical_name(name: str) -> str:
    """Derive a clean, normalized canonical identifier from a title or filename."""
    normalized = re.sub(r"[^\w\s-]", "", name.lower()).strip()
    normalized = re.sub(r"[-\s]+", "_", normalized)
    return normalized[:255] if normalized else f"family_{uuid.uuid4().hex[:8]}"


class VersionService:
    """Manages document families, sequential versioning, approval workflows, and authoritative states."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.storage_service = storage_service

    def assign_families_to_existing_documents(self, db: Session, workspace_id: str) -> int:
        """Backfill existing documents in a workspace that do not yet have a document family.

        Safely assigns each independent document its own family with version_number=1,
        preserving all existing relational and vector bindings.
        """
        unassigned_docs = (
            db.query(Document)
            .filter(
                Document.workspace_id == workspace_id,
                Document.family_id.is_(None),
                Document.visibility_scope == "company",
            )
            .all()
        )

        if not unassigned_docs:
            return 0

        assigned_count = 0
        for doc in unassigned_docs:
            title = doc.display_name or Path(doc.original_filename).stem
            canonical = normalize_canonical_name(title)

            # Ensure canonical name is unique within workspace
            base_canonical = canonical
            counter = 1
            while True:
                existing_family = (
                    db.query(DocumentFamily)
                    .filter(
                        DocumentFamily.workspace_id == workspace_id,
                        DocumentFamily.canonical_name == canonical,
                    )
                    .first()
                )
                if not existing_family:
                    break
                canonical = f"{base_canonical}_{counter}"
                counter += 1

            family = DocumentFamily(
                id=str(uuid.uuid4()),
                workspace_id=workspace_id,
                title=title,
                canonical_name=canonical,
                current_document_id=doc.id,
            )
            db.add(family)
            db.flush()

            doc.family_id = family.id
            doc.version_number = 1
            doc.version_status = "current"
            if not doc.approval_status:
                doc.approval_status = "unverified"
            assigned_count += 1

        db.commit()
        logger.info(
            "Assigned %d legacy documents to new document families in workspace %s",
            assigned_count,
            workspace_id,
        )
        return assigned_count

    def get_or_create_family(
        self,
        db: Session,
        workspace_id: str,
        title: str,
        canonical_name: Optional[str] = None,
    ) -> DocumentFamily:
        """Retrieve existing family by canonical name or create a new family."""
        canonical = normalize_canonical_name(canonical_name or title)
        family = (
            db.query(DocumentFamily)
            .filter(
                DocumentFamily.workspace_id == workspace_id,
                DocumentFamily.canonical_name == canonical,
            )
            .first()
        )

        if family:
            return family

        family = DocumentFamily(
            id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            title=title.strip(),
            canonical_name=canonical,
        )
        db.add(family)
        db.commit()
        db.refresh(family)
        return family

    def list_families(self, db: Session, workspace_id: str) -> List[DocumentFamily]:
        """List all document families for a workspace, ensuring legacy documents are assigned."""
        self.assign_families_to_existing_documents(db, workspace_id)
        return (
            db.query(DocumentFamily)
            .filter(DocumentFamily.workspace_id == workspace_id)
            .order_by(DocumentFamily.title.asc())
            .all()
        )

    def get_family_by_id(
        self,
        db: Session,
        workspace_id: str,
        family_id: str,
    ) -> Optional[DocumentFamily]:
        """Retrieve document family by ID scoped to workspace."""
        self.assign_families_to_existing_documents(db, workspace_id)
        return (
            db.query(DocumentFamily)
            .filter(
                DocumentFamily.id == family_id,
                DocumentFamily.workspace_id == workspace_id,
            )
            .first()
        )

    async def upload_new_version(
        self,
        db: Session,
        workspace: Workspace,
        family: DocumentFamily,
        upload_file: UploadFile,
        uploaded_by_user_id: Optional[str] = None,
    ) -> DocumentUploadItemResult:
        """Upload a new PDF version to an existing document family.

        Preserves SHA-256 duplicate checking, assigns sequential version numbers,
        updates the family current version, and links previous version lineage.
        """
        raw_filename = upload_file.filename or "unknown.pdf"
        clean_filename = sanitize_filename(raw_filename)

        if not clean_filename.lower().endswith(".pdf"):
            return DocumentUploadItemResult(
                filename=clean_filename,
                status="error",
                message="Invalid file type. Only PDF documents (.pdf) are supported.",
            )

        # 1. Setup temporary file for streaming write
        temp_file_id = f"tmp_{uuid.uuid4().hex}.tmp"
        temp_file_path = self.storage_service.temp_root / temp_file_id

        hasher = hashlib.sha256()
        total_bytes = 0
        is_first_chunk = True

        chunk_size = self.settings.upload_chunk_size_bytes
        max_size = self.settings.max_upload_size_bytes

        try:
            self.storage_service.temp_root.mkdir(parents=True, exist_ok=True)
            with open(temp_file_path, "wb") as f_out:
                while True:
                    read_res = upload_file.read(chunk_size)
                    if hasattr(read_res, "__await__"):
                        chunk = await read_res
                    else:
                        chunk = read_res
                    if not chunk:
                        break

                    if is_first_chunk:
                        if not chunk.startswith(b"%PDF-"):
                            return DocumentUploadItemResult(
                                filename=clean_filename,
                                status="error",
                                message="Invalid file format. File does not contain a valid PDF signature (%PDF-).",
                            )
                        is_first_chunk = False

                    total_bytes += len(chunk)
                    if total_bytes > max_size:
                        max_mb = max_size // (1024 * 1024)
                        return DocumentUploadItemResult(
                            filename=clean_filename,
                            status="error",
                            message=f"File exceeds maximum allowed upload size of {max_mb} MB.",
                        )

                    hasher.update(chunk)
                    f_out.write(chunk)

            if total_bytes == 0:
                return DocumentUploadItemResult(
                    filename=clean_filename,
                    status="error",
                    message="Uploaded file is empty (0 bytes).",
                )

            sha256_digest = hasher.hexdigest()

            # 2. Duplicate Detection: Check if exact same file already exists in workspace
            existing_doc = (
                db.query(Document)
                .filter(
                    Document.workspace_id == workspace.id,
                    Document.sha256_hash == sha256_digest,
                )
                .first()
            )
            if existing_doc:
                logger.info(
                    "Duplicate content detected in workspace %s. Matches document: %s (%s)",
                    workspace.id,
                    existing_doc.original_filename,
                    existing_doc.id,
                )
                if temp_file_path.exists():
                    temp_file_path.unlink()
                return DocumentUploadItemResult(
                    filename=clean_filename,
                    status="duplicate",
                    document=DocumentResponse.model_validate(existing_doc),
                    message=f"Duplicate content detected. Matches existing document '{existing_doc.original_filename}'.",
                )

            # 3. Finalize Storage
            workspace_upload_dir = self.storage_service.upload_root / workspace.id
            workspace_upload_dir.mkdir(parents=True, exist_ok=True)

            unique_filename = f"{sha256_digest[:16]}_{uuid.uuid4().hex[:8]}.pdf"
            final_file_path = workspace_upload_dir / unique_filename

            shutil.move(str(temp_file_path), str(final_file_path))
            relative_key = str(final_file_path.relative_to(self.storage_service.storage_root)).replace("\\", "/")

            # 4. Determine next sequential version number
            max_version = (
                db.query(func.max(Document.version_number))
                .filter(Document.family_id == family.id)
                .scalar()
                or 0
            )
            next_version = max_version + 1
            previous_doc_id = family.current_document_id

            # 5. Create Document record
            doc_id = str(uuid.uuid4())
            new_document = Document(
                id=doc_id,
                workspace_id=workspace.id,
                family_id=family.id,
                version_number=next_version,
                previous_version_id=previous_doc_id,
                version_status="current",
                approval_status="unverified",
                original_filename=clean_filename,
                display_name=Path(clean_filename).stem,
                storage_key=relative_key,
                sha256_hash=sha256_digest,
                file_size_bytes=total_bytes,
                mime_type="application/pdf",
                processing_status="uploaded",
                uploaded_by_user_id=uploaded_by_user_id,
            )
            db.add(new_document)

            # 6. Mark previous current version as 'superseded'
            if previous_doc_id:
                prev_doc = db.query(Document).filter(Document.id == previous_doc_id).first()
                if prev_doc:
                    prev_doc.version_status = "superseded"

            # 7. Update family pointer
            family.current_document_id = doc_id
            db.add(family)

            # 8. Ingestion Job
            job = IngestionJob(
                document_id=doc_id,
                status="queued",
            )
            db.add(job)

            db.commit()
            db.refresh(new_document)

            audit_service.log_action(
                db=db,
                action="document_version_uploaded",
                outcome="success",
                actor_user_id=uploaded_by_user_id,
                workspace_id=workspace.id,
                resource_type="document",
                resource_id=doc_id,
                details_json={
                    "family_id": family.id,
                    "version_number": next_version,
                    "previous_version_id": previous_doc_id,
                    "original_filename": clean_filename,
                },
            )

            logger.info(
                "Uploaded version %d for family %s ('%s') as document %s",
                next_version,
                family.id,
                family.title,
                doc_id,
            )

            return DocumentUploadItemResult(
                filename=clean_filename,
                status="uploaded",
                document=DocumentResponse.model_validate(new_document),
                message=f"Successfully uploaded as Version {next_version} of '{family.title}'.",
            )

        except Exception as exc:
            db.rollback()
            if temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except OSError:
                    pass
            logger.error("Failed to upload version for family %s: %s", family.id, exc, exc_info=True)
            return DocumentUploadItemResult(
                filename=clean_filename,
                status="error",
                message=f"Upload failed: {str(exc)}",
            )

    def set_current_version(
        self,
        db: Session,
        workspace_id: str,
        family_id: str,
        document_id: str,
        actor_user_id: Optional[str] = None,
    ) -> DocumentFamily:
        """Designate a specific version as the active current version of the family."""
        family = self.get_family_by_id(db, workspace_id, family_id)
        if not family:
            raise ValueError(f"Document family '{family_id}' not found in workspace '{workspace_id}'.")

        target_doc = (
            db.query(Document)
            .filter(
                Document.id == document_id,
                Document.family_id == family_id,
                Document.workspace_id == workspace_id,
            )
            .first()
        )
        if not target_doc:
            raise ValueError(f"Document '{document_id}' does not belong to family '{family_id}'.")

        # In a transaction, update all versions in this family
        family_docs = db.query(Document).filter(Document.family_id == family_id).all()
        for doc in family_docs:
            if doc.id == document_id:
                doc.version_status = "current"
            else:
                if doc.version_status != "archived":
                    doc.version_status = "superseded"

        family.current_document_id = document_id
        db.commit()
        db.refresh(family)

        audit_service.log_action(
            db=db,
            action="document_current_version_changed",
            outcome="success",
            actor_user_id=actor_user_id,
            workspace_id=workspace_id,
            resource_type="document_family",
            resource_id=family_id,
            details_json={
                "current_document_id": document_id,
                "version_number": target_doc.version_number,
            },
        )
        logger.info(
            "Updated current version for family '%s' to document %s (v%d)",
            family.title,
            document_id,
            target_doc.version_number,
        )
        return family

    def update_approval_status(
        self,
        db: Session,
        workspace_id: str,
        document_id: str,
        approval_status: str,
        actor_user_id: Optional[str] = None,
    ) -> Document:
        """Update approval status of a document version."""
        allowed = {"unverified", "pending_review", "approved", "rejected"}
        if approval_status not in allowed:
            raise ValueError(f"Invalid approval status '{approval_status}'. Must be one of {allowed}.")

        doc = (
            db.query(Document)
            .filter(
                Document.id == document_id,
                Document.workspace_id == workspace_id,
            )
            .first()
        )
        if not doc:
            raise ValueError(f"Document '{document_id}' not found in workspace '{workspace_id}'.")

        doc.approval_status = approval_status
        db.commit()
        db.refresh(doc)

        audit_service.log_action(
            db=db,
            action="document_approval_status_updated",
            outcome="success",
            actor_user_id=actor_user_id,
            workspace_id=workspace_id,
            resource_type="document",
            resource_id=document_id,
            details_json={"approval_status": approval_status},
        )
        logger.info("Updated approval status for document %s to '%s'", document_id, approval_status)
        return doc

    def create_family(
        self,
        db: Session,
        workspace_id: str,
        title: str,
        canonical_name: Optional[str] = None,
        actor_user_id: Optional[str] = None,
    ) -> DocumentFamily:
        """Create a new explicit document family."""
        family = self.get_or_create_family(
            db=db,
            workspace_id=workspace_id,
            title=title,
            canonical_name=canonical_name,
        )
        audit_service.log_action(
            db=db,
            action="document_family_created",
            outcome="success",
            actor_user_id=actor_user_id,
            workspace_id=workspace_id,
            resource_type="document_family",
            resource_id=family.id,
            details_json={"title": family.title, "canonical_name": family.canonical_name},
        )
        return family

    def attach_document_to_family(
        self,
        db: Session,
        workspace_id: str,
        family_id: str,
        document_id: str,
        version_number: Optional[int] = None,
        actor_user_id: Optional[str] = None,
    ) -> Document:
        """Manually link an existing document to a family, updating its lineage."""
        family = self.get_family_by_id(db, workspace_id, family_id)
        if not family:
            raise ValueError(f"Document family '{family_id}' not found in workspace '{workspace_id}'.")

        doc = (
            db.query(Document)
            .filter(
                Document.id == document_id,
                Document.workspace_id == workspace_id,
            )
            .first()
        )
        if not doc:
            raise ValueError(f"Document '{document_id}' not found in workspace '{workspace_id}'.")

        # Determine version number
        if version_number is None:
            max_v = (
                db.query(func.max(Document.version_number))
                .filter(Document.family_id == family_id)
                .scalar()
                or 0
            )
            version_number = max_v + 1

        doc.family_id = family.id
        doc.version_number = version_number
        if not doc.version_status:
            doc.version_status = "superseded" if family.current_document_id and family.current_document_id != doc.id else "current"

        db.commit()
        db.refresh(doc)

        audit_service.log_action(
            db=db,
            action="document_version_attached",
            outcome="success",
            actor_user_id=actor_user_id,
            workspace_id=workspace_id,
            resource_type="document",
            resource_id=document_id,
            details_json={
                "family_id": family.id,
                "version_number": version_number,
            },
        )
        logger.info("Attached document %s to family %s as version %d", document_id, family.id, version_number)
        return doc

    def detach_document_from_family(
        self,
        db: Session,
        workspace_id: str,
        document_id: str,
        actor_user_id: Optional[str] = None,
    ) -> DocumentFamily:
        """Detach a document from its current family and establish it in an independent family."""
        doc = (
            db.query(Document)
            .filter(
                Document.id == document_id,
                Document.workspace_id == workspace_id,
            )
            .first()
        )
        if not doc:
            raise ValueError(f"Document '{document_id}' not found in workspace '{workspace_id}'.")

        old_family_id = doc.family_id
        title = doc.display_name or Path(doc.original_filename).stem
        canonical = normalize_canonical_name(title)

        new_family = DocumentFamily(
            id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            title=title,
            canonical_name=f"{canonical}_{uuid.uuid4().hex[:6]}",
            current_document_id=doc.id,
        )
        db.add(new_family)
        db.flush()

        doc.family_id = new_family.id
        doc.version_number = 1
        doc.version_status = "current"
        doc.previous_version_id = None

        # If old family had this document as current, reset old family current_document_id
        if old_family_id:
            old_fam = db.query(DocumentFamily).filter(DocumentFamily.id == old_family_id).first()
            if old_fam and old_fam.current_document_id == document_id:
                latest_remaining = (
                    db.query(Document)
                    .filter(Document.family_id == old_family_id, Document.id != document_id)
                    .order_by(Document.version_number.desc())
                    .first()
                )
                old_fam.current_document_id = latest_remaining.id if latest_remaining else None
                if latest_remaining:
                    latest_remaining.version_status = "current"

        db.commit()
        db.refresh(doc)
        db.refresh(new_family)

        audit_service.log_action(
            db=db,
            action="document_version_detached",
            outcome="success",
            actor_user_id=actor_user_id,
            workspace_id=workspace_id,
            resource_type="document",
            resource_id=document_id,
            details_json={
                "old_family_id": old_family_id,
                "new_family_id": new_family.id,
            },
        )
        logger.info("Detached document %s into new family %s", document_id, new_family.id)
        return new_family

    def detect_version_signals(
        self,
        db: Session,
        workspace_id: str,
        filename: str,
        file_hash: Optional[str] = None,
        text_content: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Multi-signal document version and duplicate detection.

        Evaluates:
        1. SHA-256 hash exact match.
        2. Extracted document reference codes (e.g. SOP-ENG-042, SPEC-789).
        3. Revision identifiers in filename and text (e.g. _v2, Rev B, 2025).
        4. Normalized title / canonical name string similarity.
        5. Semantic/text overlap.
        """
        import difflib
        clean_name = sanitize_filename(filename)
        stem = Path(clean_name).stem
        signals: List[str] = []

        # 1. Exact SHA-256 Match
        if file_hash:
            query = db.query(Document).filter(
                Document.workspace_id == workspace_id,
                Document.sha256_hash == file_hash,
                Document.visibility_scope == "company",
            )
            if doc_id:
                query = query.filter(Document.id != doc_id)
            exact_match = query.first()
            if exact_match:
                return {
                    "classification": "exact_duplicate",
                    "confidence": 1.0,
                    "matched_family_id": exact_match.family_id,
                    "matched_family_title": exact_match.family.title if exact_match.family else exact_match.display_name,
                    "matched_document_id": exact_match.id,
                    "detected_document_code": None,
                    "detected_revision": None,
                    "similarity_score": 1.0,
                    "explanation": f"Identical file content (SHA-256: {file_hash[:12]}...) matches existing document '{exact_match.original_filename}'.",
                    "signals": ["sha256_match"],
                }

        # 2. Extract Document Reference Numbers / Codes
        code_regex = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,6}(?:[-_][A-Z0-9]+){1,3}|[A-Z]{2,4}[0-9]{3,6})(?![A-Za-z0-9])")
        found_codes = code_regex.findall(stem)
        if text_content and not found_codes:
            found_codes = code_regex.findall(text_content[:2000])
        doc_code = found_codes[0] if found_codes else None
        if doc_code:
            signals.append("doc_code_detected")

        # 3. Extract Revision Identifiers
        rev_regex = re.compile(r"\b(?:rev(?:ision)?|ver(?:sion)?|v)[.\s_-]*([0-9]+(?:[.-][0-9]+)?|[A-Za-z])\b", re.IGNORECASE)
        found_revs = rev_regex.findall(stem)
        if text_content and not found_revs:
            found_revs = rev_regex.findall(text_content[:1500])
        revision_tag = found_revs[0] if found_revs else None
        if revision_tag:
            signals.append("revision_tag_detected")

        # Check for year-based revisions (e.g., 2024, 2025)
        year_regex = re.compile(r"\b(20[2-9][0-9])\b")
        years = year_regex.findall(stem)
        if years and not revision_tag:
            revision_tag = years[0]
            signals.append("year_revision_detected")

        # 4. Compare against existing company families and documents
        families = self.list_families(db, workspace_id)
        if not families:
            return {
                "classification": "new_document",
                "confidence": 0.95,
                "matched_family_id": None,
                "matched_family_title": None,
                "matched_document_id": None,
                "detected_document_code": doc_code,
                "detected_revision": revision_tag,
                "similarity_score": 0.0,
                "explanation": "No existing document families in workspace. Document is completely new.",
                "signals": signals,
            }

        # Normalize the candidate name by stripping revision patterns
        normalized_stem = re.sub(r"\b(?:rev(?:ision)?|ver(?:sion)?|v)[.\s_-]*[0-9A-Za-z]+\b", "", stem, flags=re.IGNORECASE)
        normalized_stem = re.sub(r"\b20[2-9][0-9]\b", "", normalized_stem)
        normalized_stem = re.sub(r"[-_\s]+", " ", normalized_stem).strip().lower()

        best_family = None
        best_doc = None
        best_score = 0.0
        code_matched_family = None

        tokens_stem = " ".join(sorted(normalized_stem.split()))

        for fam in families:
            fam_title_clean = re.sub(r"[-_\s]+", " ", fam.title).strip().lower()
            fam_canon_clean = re.sub(r"[-_\s]+", " ", fam.canonical_name).strip().lower()

            # Check if extracted code is part of family
            if doc_code and (doc_code.lower() in fam_title_clean or doc_code.lower() in fam_canon_clean):
                code_matched_family = fam

            # String similarity
            score1 = difflib.SequenceMatcher(None, normalized_stem, fam_title_clean).ratio()
            score2 = difflib.SequenceMatcher(None, normalized_stem, fam_canon_clean).ratio()
            tokens_fam_title = " ".join(sorted(fam_title_clean.split()))
            score_tokens = difflib.SequenceMatcher(None, tokens_stem, tokens_fam_title).ratio()
            max_score = max(score1, score2, score_tokens)

            # Check versions within family
            for v in fam.versions:
                if doc_id and v.id == doc_id:
                    continue
                v_stem = Path(v.original_filename).stem
                v_norm = re.sub(r"\b(?:rev(?:ision)?|ver(?:sion)?|v)[.\s_-]*[0-9A-Za-z]+\b", "", v_stem, flags=re.IGNORECASE)
                v_norm = re.sub(r"[-_\s]+", " ", v_norm).strip().lower()
                v_score = difflib.SequenceMatcher(None, normalized_stem, v_norm).ratio()
                v_tokens = " ".join(sorted(v_norm.split()))
                v_score_tokens = difflib.SequenceMatcher(None, tokens_stem, v_tokens).ratio()
                max_v = max(v_score, v_score_tokens)
                if max_v > max_score:
                    max_score = max_v
                    best_doc = v

            if max_score > best_score:
                best_score = max_score
                best_family = fam

        # 5. Determine Classification
        if code_matched_family:
            signals.append("code_family_match")
            classification = "possible_new_version"
            confidence = 0.92 if revision_tag else 0.85
            return {
                "classification": classification,
                "confidence": confidence,
                "matched_family_id": code_matched_family.id,
                "matched_family_title": code_matched_family.title,
                "matched_document_id": code_matched_family.current_document_id,
                "detected_document_code": doc_code,
                "detected_revision": revision_tag,
                "similarity_score": round(best_score, 3),
                "explanation": f"Document code '{doc_code}' matches family '{code_matched_family.title}' (Revision tag: {revision_tag or 'unspecified'}).",
                "signals": signals,
            }

        if best_score >= 0.82 and revision_tag:
            signals.append("high_title_similarity_with_revision")
            return {
                "classification": "possible_new_version",
                "confidence": 0.88,
                "matched_family_id": best_family.id if best_family else None,
                "matched_family_title": best_family.title if best_family else None,
                "matched_document_id": best_doc.id if best_doc else (best_family.current_document_id if best_family else None),
                "detected_document_code": doc_code,
                "detected_revision": revision_tag,
                "similarity_score": round(best_score, 3),
                "explanation": f"High filename similarity ({round(best_score*100)}%) and revision identifier '{revision_tag}' match family '{best_family.title}'.",
                "signals": signals,
            }

        if best_score >= 0.65:
            signals.append("moderate_title_similarity")
            confidence_val = round(min(max(best_score, 0.50), 0.85), 2)
            return {
                "classification": "uncertain_match",
                "confidence": confidence_val,
                "matched_family_id": best_family.id if best_family else None,
                "matched_family_title": best_family.title if best_family else None,
                "matched_document_id": best_doc.id if best_doc else (best_family.current_document_id if best_family else None),
                "detected_document_code": doc_code,
                "detected_revision": revision_tag,
                "similarity_score": round(best_score, 3),
                "explanation": f"Moderate similarity ({round(best_score*100)}%) with family '{best_family.title if best_family else 'unknown'}'. Administrator confirmation recommended.",
                "signals": signals,
            }

        # Otherwise completely new document
        return {
            "classification": "new_document",
            "confidence": 0.95,
            "matched_family_id": None,
            "matched_family_title": None,
            "matched_document_id": None,
            "detected_document_code": doc_code,
            "detected_revision": revision_tag,
            "similarity_score": round(best_score, 3),
            "explanation": f"Low similarity ({round(best_score*100)}%) across all existing families. Classified as a new independent document.",
            "signals": signals,
        }


version_service = VersionService()
