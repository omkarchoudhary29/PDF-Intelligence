"""Document and Repository Summarization Service for Enterprise Knowledge Navigator.

Provides hierarchical document summarization (~100-200 words per document) and
executive repository-level synthesis with offline deterministic fallback.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from backend.database.models import Document, DocumentRelationship, Entity
from backend.schemas.report import DocumentSummaryData, ExecutiveSummaryData
from backend.services.ingestion_service import ingestion_service
from backend.services.llm_service import LLMError, LLMNotConfiguredError, llm_service

logger = logging.getLogger("ekn.reports.summarization")


class SummarizationService:
    """Generates, caches, and aggregates document-level and workspace-level summaries."""

    def summarize_document(
        self,
        db: Session,
        workspace_id: str,
        document: Document,
        force: bool = False,
    ) -> DocumentSummaryData:
        """Generate or retrieve a concise (~100-200 word) document summary preserving key page numbers."""
        # 1. Reuse existing stored summary if valid and not forced
        if not force and document.summary and document.summary_status == "completed":
            try:
                meta = json.loads(document.summary)
                return DocumentSummaryData(
                    document_id=document.id,
                    original_filename=document.original_filename,
                    total_pages=document.total_pages or 1,
                    purpose=meta.get("purpose", "Document overview."),
                    main_subjects=meta.get("main_subjects", []),
                    important_findings=meta.get("important_findings", []),
                    key_sections=meta.get("key_sections", []),
                    page_references=meta.get("page_references", [1]),
                    summary_text=meta.get("summary_text", ""),
                    related_documents=meta.get("related_documents", []),
                    summary_status="completed",
                    version_number=getattr(document, "version_number", 1) or 1,
                    version_status=getattr(document, "version_status", "current") or "current",
                    family_title=document.family.title if getattr(document, "family", None) else None,
                )
            except Exception:
                pass  # Fall through to re-generate if stored JSON was unparseable

        # 2. Load extracted pages
        pages = ingestion_service.get_extracted_pages(db, workspace_id, document.id)
        if not pages:
            return DocumentSummaryData(
                document_id=document.id,
                original_filename=document.original_filename,
                total_pages=document.total_pages or 1,
                purpose="Content has not been extracted yet.",
                summary_text="Extraction records unavailable.",
                summary_status="unavailable",
            )

        # Collect headings and key text excerpts
        sections: List[str] = []
        page_refs: List[int] = []
        text_snippets: List[str] = []

        for p in pages:
            if hasattr(p, "model_dump"):
                p_dict = p.model_dump()
            elif isinstance(p, dict):
                p_dict = p
            elif hasattr(p, "__dict__"):
                p_dict = p.__dict__
            else:
                p_dict = {}

            p_num = p_dict.get("page_number", 1)
            raw = (p_dict.get("text") or p_dict.get("raw_text") or "").strip()
            headings = p_dict.get("headings", [])
            for h in headings:
                h_text = h.get("text", "") if isinstance(h, dict) else (getattr(h, "text", str(h)))
                h_text = str(h_text).strip()
                if h_text and h_text not in sections:
                    sections.append(h_text)
                    page_refs.append(p_num)

            if raw:
                text_snippets.append(f"[Page {p_num}]: {raw[:400]}")

        # Find related documents from Phase 8 relationships
        rels = (
            db.query(DocumentRelationship)
            .filter(
                DocumentRelationship.workspace_id == workspace_id,
                (DocumentRelationship.source_document_id == document.id) |
                (DocumentRelationship.target_document_id == document.id),
            )
            .all()
        )
        related_doc_ids = set()
        for r in rels:
            other_id = r.target_document_id if r.source_document_id == document.id else r.source_document_id
            related_doc_ids.add(other_id)

        related_names = []
        if related_doc_ids:
            rel_docs = db.query(Document).filter(Document.id.in_(list(related_doc_ids))).all()
            related_names = [d.original_filename for d in rel_docs]

        # 3. LLM-based structured summary if available
        if llm_service.is_configured:
            context = "\n".join(text_snippets[:6])
            prompt = (
                "You are an enterprise technical writer. "
                f"Summarize the following PDF document: '{document.original_filename}'.\n\n"
                "Provide a concise summary (~100-180 words) in valid JSON only with keys:\n"
                '- "purpose": A 1-2 sentence statement of the document\'s purpose.\n'
                '- "main_subjects": List of 2-4 primary subjects or components.\n'
                '- "important_findings": List of 2-3 key technical findings, limits, or requirements.\n'
                '- "summary_text": Cohesive summary paragraph.\n\n'
                f"Document Excerpts:\n{context}\n"
            )
            try:
                model = llm_service.get_llm()
                resp = model.invoke(prompt)
                content = resp.content if hasattr(resp, "content") else str(resp)

                # Strip markdown fences
                cleaned = content.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                elif cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                parsed = json.loads(cleaned.strip())

                summary_obj = DocumentSummaryData(
                    document_id=document.id,
                    original_filename=document.original_filename,
                    total_pages=document.total_pages or len(pages),
                    purpose=parsed.get("purpose", f"Overview of {document.display_name}."),
                    main_subjects=parsed.get("main_subjects", sections[:3] or ["General Operations"]),
                    important_findings=parsed.get("important_findings", []),
                    key_sections=sections[:5],
                    page_references=list(set(page_refs[:5])) or [1],
                    summary_text=parsed.get("summary_text", ""),
                    related_documents=related_names,
                    summary_status="completed",
                    version_number=getattr(document, "version_number", 1) or 1,
                    version_status=getattr(document, "version_status", "current") or "current",
                    family_title=document.family.title if getattr(document, "family", None) else None,
                )

                # Persist to document record
                document.summary = json.dumps(summary_obj.model_dump())
                document.summary_status = "completed"
                db.commit()
                return summary_obj

            except Exception as exc:
                logger.warning("LLM summarization failed for doc %s (%s); using deterministic summary.", document.id, exc)

        # 4. Deterministic summary fallback
        first_page_text = ""
        if pages:
            p0 = pages[0]
            if hasattr(p0, "text") and p0.text:
                first_page_text = p0.text.strip()
            elif hasattr(p0, "raw_text") and p0.raw_text:
                first_page_text = p0.raw_text.strip()
            elif isinstance(p0, dict):
                first_page_text = (p0.get("text") or p0.get("raw_text") or "").strip()
        first_snippet = first_page_text[:350].replace("\n", " ").strip()
        purpose = f"Technical documentation covering {document.display_name}."
        if sections:
            purpose += f" Primary focus: {sections[0]}."

        fallback_summary = DocumentSummaryData(
            document_id=document.id,
            original_filename=document.original_filename,
            total_pages=document.total_pages or len(pages),
            purpose=purpose,
            main_subjects=sections[:4] or ["Technical Documentation"],
            important_findings=["Operational parameters detailed in document body."],
            key_sections=sections[:6],
            page_references=list(set(page_refs[:5])) or [1],
            summary_text=(
                f"{document.original_filename} ({document.total_pages or len(pages)} pages) "
                f"provides guidelines and technical specifications. "
                f"{first_snippet[:250]}..."
            ),
            related_documents=related_names,
            summary_status="completed",
            version_number=getattr(document, "version_number", 1) or 1,
            version_status=getattr(document, "version_status", "current") or "current",
            family_title=document.family.title if getattr(document, "family", None) else None,
        )

        document.summary = json.dumps(fallback_summary.model_dump())
        document.summary_status = "completed"
        db.commit()
        return fallback_summary

    def generate_executive_overview(
        self,
        db: Session,
        workspace_id: str,
        doc_summaries: List[DocumentSummaryData],
    ) -> ExecutiveSummaryData:
        """Generate high-level executive overview synthesizing the entire repository."""
        total_docs = len(doc_summaries)
        total_pages = sum(d.total_pages for d in doc_summaries)

        # Collect unique subjects
        all_subjects = []
        for d in doc_summaries:
            all_subjects.extend(d.main_subjects)
        unique_subjects = list(dict.fromkeys(all_subjects))[:8]

        # LLM Executive synthesis if available
        if llm_service.is_configured and doc_summaries:
            summary_digest = "\n".join(
                f"- {d.original_filename} ({d.total_pages}p): {d.purpose} Key subjects: {', '.join(d.main_subjects)}"
                for d in doc_summaries[:15]
            )
            prompt = (
                "You are an enterprise knowledge architect. "
                "Synthesize a concise executive repository overview (~150-250 words) from the following document summaries.\n\n"
                "Return valid JSON only with keys:\n"
                '- "overview_text": Synthesized overview paragraph.\n'
                '- "key_themes": List of 3-5 major overarching themes.\n'
                '- "documentation_gaps": List of 1-3 observed gaps or areas needing attention.\n\n'
                f"Document Digest ({total_docs} documents, {total_pages} pages total):\n"
                f"{summary_digest}\n"
            )
            try:
                model = llm_service.get_llm()
                resp = model.invoke(prompt)
                content = resp.content if hasattr(resp, "content") else str(resp)

                cleaned = content.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                elif cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                parsed = json.loads(cleaned.strip())

                return ExecutiveSummaryData(
                    overview_text=parsed.get("overview_text", ""),
                    key_themes=parsed.get("key_themes", unique_subjects[:4]),
                    documentation_gaps=parsed.get("documentation_gaps", []),
                    document_count=total_docs,
                    total_pages=total_pages,
                )
            except Exception as exc:
                logger.warning("Executive overview LLM synthesis failed: %s; using deterministic synthesis.", exc)

        # Deterministic Executive Overview
        overview = (
            f"This enterprise repository consolidates {total_docs} document(s) encompassing {total_pages} total pages. "
            f"Primary operational domains include {', '.join(unique_subjects[:4]) if unique_subjects else 'General Systems'}. "
            f"The repository establishes standardized guidelines, equipment operating manuals, and compliance protocols across enterprise workflows."
        )

        return ExecutiveSummaryData(
            overview_text=overview,
            key_themes=unique_subjects[:5] or ["Operations", "Safety", "Compliance"],
            documentation_gaps=["Periodic review recommended to maintain version synchronicity."],
            document_count=total_docs,
            total_pages=total_pages,
        )


summarization_service = SummarizationService()
