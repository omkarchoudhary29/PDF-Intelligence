"""Report Generation Service for Enterprise Knowledge Navigator.

Orchestrates the automated generation of Enterprise Knowledge Dossier PDFs,
managing the lifecycle (queued -> generating -> completed/partially_completed/failed),
document summary collection, graph embedding, and physical file persistence.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid
from sqlalchemy.orm import Session

from backend.database.models import Document, DocumentRelationship, Report, Workspace
from backend.schemas.report import (
    DocumentSummaryData,
    ExecutiveSummaryData,
    ReportCreateRequest,
    ReportResponse,
)
from backend.services.graph_visualization_service import graph_visualization_service
from backend.services.knowledge_graph_service import knowledge_graph_service
from backend.services.pdf_composer_service import pdf_composer_service
from backend.services.summarization_service import summarization_service
from config.settings import get_settings

logger = logging.getLogger("ekn.reports.generation")


class ReportGenerationService:
    """Orchestrates end-to-end report generation lifecycle and file persistence."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.reports_root = Path(self.settings.reports_dir).resolve()
        self.reports_root.mkdir(parents=True, exist_ok=True)

    def get_report_file_path(self, report: Report) -> Path:
        """Resolve physical PDF path on disk and verify existence."""
        if not report.storage_key:
            raise FileNotFoundError(f"Report {report.id} does not have a storage key.")
        file_path = (self.reports_root / report.storage_key).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"Generated PDF file for report {report.id} was not found on disk.")
        return file_path

    def create_report_job(
        self,
        db: Session,
        workspace_id: str,
        request: ReportCreateRequest,
    ) -> Report:
        """Create a new report record in 'queued' state."""
        name = (request.report_name or "").strip()
        if not name:
            name = f"Knowledge Dossier — {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        report = Report(
            workspace_id=workspace_id,
            report_name=name,
            report_type=request.report_type,
            status="queued",
            metadata_json={"include_graph": request.include_graph},
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        logger.info("Created report job %s in workspace %s", report.id, workspace_id)
        return report

    def execute_report_generation(
        self,
        db: Session,
        workspace_id: str,
        report_id: str,
    ) -> Report:
        """Execute complete generation workflow synchronously or worker-invoked."""
        report = (
            db.query(Report)
            .filter(Report.id == report_id, Report.workspace_id == workspace_id)
            .first()
        )
        if not report:
            raise ValueError(f"Report '{report_id}' was not found in workspace '{workspace_id}'.")

        ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if not ws:
            raise ValueError(f"Workspace '{workspace_id}' was not found.")

        # Update status to generating
        report.status = "generating"
        report.generation_started_at = datetime.now()
        report.error_message = None
        db.commit()

        try:
            # 1. Fetch eligible current indexed documents (avoiding superseded revisions in main index)
            docs = (
                db.query(Document)
                .filter(
                    Document.workspace_id == workspace_id,
                    (Document.version_status == "current") | (Document.version_status.is_(None)),
                )
                .order_by(Document.created_at.asc())
                .all()
            )
            if not docs:
                docs = (
                    db.query(Document)
                    .filter(Document.workspace_id == workspace_id)
                    .order_by(Document.created_at.asc())
                    .all()
                )

            if not docs:
                report.status = "failed"
                report.error_message = "No documents found in workspace. Upload documents before generating a dossier."
                report.generation_completed_at = datetime.now()
                db.commit()
                return report

            # 2. Collect or generate document summaries
            doc_summaries: List[DocumentSummaryData] = []
            has_partial = False

            for d in docs:
                try:
                    summary = summarization_service.summarize_document(
                        db=db,
                        workspace_id=workspace_id,
                        document=d,
                        force=False,
                    )
                    doc_summaries.append(summary)
                    if summary.summary_status != "completed":
                        has_partial = True
                except Exception as sum_err:
                    logger.warning("Summary generation failed for doc %s: %s", d.id, sum_err)
                    has_partial = True
                    doc_summaries.append(
                        DocumentSummaryData(
                            document_id=d.id,
                            original_filename=d.original_filename,
                            total_pages=d.total_pages or 1,
                            purpose=f"Summary generation error: {sum_err}",
                            summary_text="Partial summary due to processing limitation.",
                            summary_status="partial",
                        )
                    )

            # 3. Generate Executive Repository Overview
            exec_overview = summarization_service.generate_executive_overview(
                db=db,
                workspace_id=workspace_id,
                doc_summaries=doc_summaries,
            )

            # Attach contradiction findings and version review notices to executive overview
            from backend.database.models import ContradictionFinding
            findings = (
                db.query(ContradictionFinding)
                .filter(ContradictionFinding.workspace_id == workspace_id)
                .order_by(ContradictionFinding.created_at.desc())
                .all()
            )
            exec_overview.contradiction_notices = [
                {
                    "subject": f.subject,
                    "finding_type": f.finding_type,
                    "review_status": f.review_status,
                    "explanation": f.explanation,
                    "is_confirmed": f.review_status in {"confirmed", "resolved"},
                }
                for f in findings
            ]

            # 4. Fetch Cross-Document Relationships
            relationships = (
                db.query(DocumentRelationship)
                .filter(DocumentRelationship.workspace_id == workspace_id)
                .all()
            )
            rel_dicts: List[Dict[str, Any]] = []
            doc_name_map = {d.id: d.original_filename for d in docs}

            for r in relationships:
                rel_dicts.append({
                    "relationship_type": r.relationship_type,
                    "source_document_id": r.source_document_id,
                    "target_document_id": r.target_document_id,
                    "source_document_name": doc_name_map.get(r.source_document_id, "Unknown"),
                    "target_document_name": doc_name_map.get(r.target_document_id, "Unknown"),
                    "confidence_score": r.confidence_score,
                    "explanation": r.explanation,
                    "supporting_evidence": r.supporting_evidence,
                })

            # 5. Render Knowledge Graph Image if requested
            graph_png_bytes: Optional[bytes] = None
            include_graph = (report.metadata_json or {}).get("include_graph", True)
            if include_graph:
                try:
                    G = knowledge_graph_service.construct_networkx_graph(
                        db=db,
                        workspace_id=workspace_id,
                        limit=35,
                    )
                    graph_png_bytes = graph_visualization_service.render_graph_to_png(
                        G,
                        title=f"Knowledge Network — {ws.name}",
                    )
                except Exception as graph_err:
                    logger.warning("Graph rendering for report %s encountered error: %s", report.id, graph_err)

            # 6. Compose PDF File via ReportLab
            relative_storage_key = f"{workspace_id}/{report.id}.pdf"
            physical_pdf_path = self.reports_root / relative_storage_key

            total_report_pages = pdf_composer_service.compose_dossier_pdf(
                output_path=physical_pdf_path,
                workspace_name=ws.name,
                workspace_id=ws.id,
                executive_summary=exec_overview,
                document_summaries=doc_summaries,
                relationships=rel_dicts,
                graph_png_bytes=graph_png_bytes,
                report_title=report.report_name,
            )

            # 7. Update Report Record upon successful generation
            file_size = physical_pdf_path.stat().st_size
            report.status = "partially_completed" if has_partial else "completed"
            report.storage_key = relative_storage_key
            report.document_count = len(docs)
            report.total_pages_covered = sum(d.total_pages or 1 for d in docs)
            report.file_size_bytes = file_size
            report.generation_completed_at = datetime.now()
            report.metadata_json = {
                **(report.metadata_json or {}),
                "report_page_count": total_report_pages,
                "partially_completed": has_partial,
            }
            db.commit()
            db.refresh(report)

            logger.info("Report %s finalized: status=%s, size=%d bytes", report.id, report.status, file_size)
            return report

        except Exception as exc:
            db.rollback()
            logger.error("Report generation failed for %s: %s", report_id, exc, exc_info=True)
            report_failed = db.query(Report).filter(Report.id == report_id).first()
            if report_failed:
                report_failed.status = "failed"
                report_failed.error_message = str(exc)
                report_failed.generation_completed_at = datetime.now()
                db.commit()
                return report_failed
            return report

    def delete_report(self, db: Session, workspace_id: str, report_id: str) -> bool:
        """Safely delete report record and physical PDF file."""
        report = (
            db.query(Report)
            .filter(Report.id == report_id, Report.workspace_id == workspace_id)
            .first()
        )
        if not report:
            return False

        # Remove physical file if present
        if report.storage_key:
            try:
                physical_path = (self.reports_root / report.storage_key).resolve()
                if physical_path.is_file():
                    physical_path.unlink()
            except Exception as exc:
                logger.warning("Could not delete physical report file for %s: %s", report_id, exc)

        db.delete(report)
        db.commit()
        return True


report_generation_service = ReportGenerationService()
