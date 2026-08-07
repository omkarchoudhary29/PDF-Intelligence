from datetime import datetime
import difflib
import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import uuid

from sqlalchemy.orm import Session

from backend.database.models import Document, DocumentFamily, VersionComparison
from backend.schemas.document import PageExtractionRecord
from backend.schemas.version import (
    NumericalChangeItem,
    PageAlignmentItem,
    TableDiffBlock,
    TextDiffBlock,
    VersionComparisonResponse,
)
from backend.services.ingestion_service import ingestion_service
from backend.services.llm_service import llm_service
from config.settings import get_settings

logger = logging.getLogger("ekn.comparison")


def normalize_page_text(text: str) -> str:
    """Normalize line endings and whitespace while strictly preserving numbers, codes, and symbols."""
    if not text:
        return ""
    # Normalize line endings
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # Normalize internal repeated whitespace on each line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    # Strip empty leading/trailing lines and collapse 3+ empty lines into 1
    result: List[str] = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run <= 1:
                result.append("")
        else:
            blank_run = 0
            result.append(line)
    return "\n".join(result).strip()


def compute_page_hash(record: PageExtractionRecord) -> str:
    """Compute deterministic SHA-256 hash of normalized text and structured tables on a page."""
    normalized_text = normalize_page_text(record.text)
    serialized_tables = json.dumps(record.tables or [], sort_keys=True)
    payload = f"{normalized_text}\n---TABLES---\n{serialized_tables}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def classify_change_category(old_text: str, new_text: str) -> str:
    """Classify text differences into enterprise change categories.

    Categories:
    - Administrative
    - Formatting-only
    - Procedural
    - Technical specification
    - Safety-related
    - Regulatory
    - Responsibility change
    - Other
    - Unclassified
    """
    if not old_text and not new_text:
        return "Unclassified"

    # Check for formatting-only changes (ignoring whitespace and capitalization)
    clean_old = re.sub(r"\s+", "", old_text).lower()
    clean_new = re.sub(r"\s+", "", new_text).lower()
    if clean_old == clean_new:
        return "Formatting-only"

    combined = f"{old_text} {new_text}".lower()

    # Safety-related
    safety_terms = [
        "danger", "hazard", "warning", "caution", "safety", "ppe",
        "protective", "emergency", "fire", "evacuat", "toxic", "lethal",
        "fatal", "injury", "fail-safe", "safeguard",
    ]
    if any(term in combined for term in safety_terms):
        return "Safety-related"

    # Regulatory
    reg_terms = [
        "regulation", "regulatory", "compliance", "standard", "iso",
        "osha", "faa", "audit", "certif", "legal", "statutory", "mandate",
        "requirement", "directive", "clause",
    ]
    if any(term in combined for term in reg_terms):
        return "Regulatory"

    # Procedural
    proc_terms = [
        "step", "procedure", "process", "workflow", "shall", "must",
        "instruct", "operation", "operate", "checklist", "protocol",
        "action", "sequence", "phase", "method", "execute", "task",
    ]
    if any(term in combined for term in proc_terms):
        return "Procedural"

    # Technical specification
    tech_terms = [
        "tolerance", "psi", "bar", "voltage", "amper", "hz", "rpm",
        "temperature", "celsius", "fahrenheit", "dimension", "diameter",
        "thickness", "weight", "frequency", "spec", "specification",
        "pressure", "velocity", "altitude", "knot", "knots", "calibration",
        "payload", "torque", "capacity", "maximum", "minimum", "limit",
    ]
    unit_regex = re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:kg|lbs?|mm|cm|m|in|ft|v|a|w|kw|kpa|mpa|psi|bar|hz|rpm|deg|°[cf]|%)\b",
        re.IGNORECASE,
    )
    if any(term in combined for term in tech_terms) or unit_regex.search(combined):
        return "Technical specification"

    # Administrative document control markers take precedence over incidental role mentions in approvals
    is_doc_control = (
        "document version" in combined
        or "revision history" in combined
        or ("version" in combined and any(t in combined for t in ["issued", "approved by", "effective", "author"]))
    )
    if is_doc_control:
        return "Administrative"

    # Responsibility change
    resp_terms = [
        "responsible", "accountable", "responsibility", "manager", "engineer",
        "director", "officer", "supervisor", "personnel", "team", "role",
        "approver", "sign-off", "assigned to", "operator", "crew",
    ]
    if any(term in combined for term in resp_terms):
        return "Responsibility change"

    # Administrative
    admin_terms = [
        "document", "title", "version", "revision", "author", "date",
        "approved by", "issued", "effective", "confidential", "classification",
        "page", "section", "reference", "index", "distribution",
    ]
    if any(term in combined for term in admin_terms):
        return "Administrative"

    return "Other"


class VersionComparisonService:
    """Performs deterministic page-level alignment, text/table diffing, and AI-assisted change explanation."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def align_pages(
        self,
        source_pages: List[PageExtractionRecord],
        target_pages: List[PageExtractionRecord],
    ) -> List[PageAlignmentItem]:
        """Align page sequences between source and target versions using difflib.SequenceMatcher on content hashes."""
        src_hashes = [compute_page_hash(p) for p in source_pages]
        tgt_hashes = [compute_page_hash(p) for p in target_pages]

        matcher = difflib.SequenceMatcher(None, src_hashes, tgt_hashes)
        alignments: List[PageAlignmentItem] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                # Both slices are identical
                for src_idx, tgt_idx in zip(range(i1, i2), range(j1, j2)):
                    alignments.append(
                        PageAlignmentItem(
                            source_page=source_pages[src_idx].page_number,
                            target_page=target_pages[tgt_idx].page_number,
                            status="unchanged",
                            similarity_score=1.0,
                        )
                    )
            elif tag == "replace":
                # Modified block
                src_slice = source_pages[i1:i2]
                tgt_slice = target_pages[j1:j2]
                min_len = min(len(src_slice), len(tgt_slice))

                for idx in range(min_len):
                    s_page = src_slice[idx]
                    t_page = tgt_slice[idx]
                    sim = difflib.SequenceMatcher(
                        None,
                        normalize_page_text(s_page.text),
                        normalize_page_text(t_page.text),
                    ).ratio()
                    alignments.append(
                        PageAlignmentItem(
                            source_page=s_page.page_number,
                            target_page=t_page.page_number,
                            status="modified",
                            similarity_score=round(sim, 4),
                        )
                    )

                # Excess source pages in replace block are removed
                if len(src_slice) > min_len:
                    for s_page in src_slice[min_len:]:
                        alignments.append(
                            PageAlignmentItem(
                                source_page=s_page.page_number,
                                target_page=None,
                                status="removed",
                                similarity_score=0.0,
                            )
                        )

                # Excess target pages in replace block are added
                if len(tgt_slice) > min_len:
                    for t_page in tgt_slice[min_len:]:
                        alignments.append(
                            PageAlignmentItem(
                                source_page=None,
                                target_page=t_page.page_number,
                                status="added",
                                similarity_score=0.0,
                            )
                        )

            elif tag == "delete":
                # Pages removed in target
                for src_idx in range(i1, i2):
                    alignments.append(
                        PageAlignmentItem(
                            source_page=source_pages[src_idx].page_number,
                            target_page=None,
                            status="removed",
                            similarity_score=0.0,
                        )
                    )

            elif tag == "insert":
                # Pages added in target
                for tgt_idx in range(j1, j2):
                    alignments.append(
                        PageAlignmentItem(
                            source_page=None,
                            target_page=target_pages[tgt_idx].page_number,
                            status="added",
                            similarity_score=0.0,
                        )
                    )

        return alignments

    def extract_numerical_changes(
        self,
        src_text: str,
        tgt_text: str,
        src_page: Optional[int],
        tgt_page: Optional[int],
    ) -> List[NumericalChangeItem]:
        """Detect modified numbers and parameters between source and target page excerpts."""
        changes: List[NumericalChangeItem] = []
        src_lines = [l.strip() for l in src_text.splitlines() if l.strip()]
        tgt_lines = [l.strip() for l in tgt_text.splitlines() if l.strip()]

        num_regex = re.compile(r"(\b\d+(?:[.,]\d+)?\b)\s*([a-zA-Z%°℃℉/]+)?")

        # Compare lines pairwise using difflib matcher
        line_matcher = difflib.SequenceMatcher(None, src_lines, tgt_lines)
        for tag, i1, i2, j1, j2 in line_matcher.get_opcodes():
            if tag == "replace":
                old_chunk = " ".join(src_lines[i1:i2])
                new_chunk = " ".join(tgt_lines[j1:j2])

                old_matches = num_regex.findall(old_chunk)
                new_matches = num_regex.findall(new_chunk)

                if old_matches and new_matches and old_matches != new_matches:
                    for (old_val, old_u), (new_val, new_u) in zip(old_matches, new_matches):
                        if old_val != new_val or old_u != new_u:
                            changes.append(
                                NumericalChangeItem(
                                    source_page=src_page,
                                    target_page=tgt_page,
                                    context=f"Old: '{old_chunk[:80]}' -> New: '{new_chunk[:80]}'",
                                    old_value=old_val,
                                    new_value=new_val,
                                    units=new_u or old_u or None,
                                )
                            )

        return changes

    def compare_tables(
        self,
        src_tables: List[Dict[str, Any]],
        tgt_tables: List[Dict[str, Any]],
        src_page: Optional[int],
        tgt_page: Optional[int],
    ) -> List[TableDiffBlock]:
        """Compare table headers, row counts, and cell data between aligned pages."""
        table_diffs: List[TableDiffBlock] = []
        max_tables = max(len(src_tables or []), len(tgt_tables or []))

        for idx in range(max_tables):
            s_tab = src_tables[idx] if idx < len(src_tables or []) else None
            t_tab = tgt_tables[idx] if idx < len(tgt_tables or []) else None

            if s_tab and not t_tab:
                table_diffs.append(
                    TableDiffBlock(
                        source_page=src_page,
                        target_page=tgt_page,
                        table_id=s_tab.get("table_id", f"tab_{idx}"),
                        change_type="rows_removed",
                        details={"message": f"Table removed in revised version ({s_tab.get('row_count', 0)} rows)."},
                    )
                )
            elif not s_tab and t_tab:
                table_diffs.append(
                    TableDiffBlock(
                        source_page=src_page,
                        target_page=tgt_page,
                        table_id=t_tab.get("table_id", f"tab_{idx}"),
                        change_type="rows_added",
                        details={"message": f"Table added in revised version ({t_tab.get('row_count', 0)} rows)."},
                    )
                )
            elif s_tab and t_tab:
                s_headers = s_tab.get("headers", [])
                t_headers = t_tab.get("headers", [])

                if s_headers != t_headers:
                    table_diffs.append(
                        TableDiffBlock(
                            source_page=src_page,
                            target_page=tgt_page,
                            table_id=t_tab.get("table_id", f"tab_{idx}"),
                            change_type="headers_changed",
                            details={"old_headers": s_headers, "new_headers": t_headers},
                        )
                    )

                s_rows = s_tab.get("rows", [])
                t_rows = t_tab.get("rows", [])
                if len(s_rows) != len(t_rows):
                    table_diffs.append(
                        TableDiffBlock(
                            source_page=src_page,
                            target_page=tgt_page,
                            table_id=t_tab.get("table_id", f"tab_{idx}"),
                            change_type="rows_added" if len(t_rows) > len(s_rows) else "rows_removed",
                            details={"old_row_count": len(s_rows), "new_row_count": len(t_rows)},
                        )
                    )
                elif s_rows != t_rows:
                    table_diffs.append(
                        TableDiffBlock(
                            source_page=src_page,
                            target_page=tgt_page,
                            table_id=t_tab.get("table_id", f"tab_{idx}"),
                            change_type="cells_modified",
                            details={"message": "Table cell values modified between versions."},
                        )
                    )

        return table_diffs

    def diff_aligned_pages(
        self,
        src_page_rec: Optional[PageExtractionRecord],
        tgt_page_rec: Optional[PageExtractionRecord],
    ) -> Tuple[List[TextDiffBlock], List[NumericalChangeItem], List[TableDiffBlock]]:
        """Compute fine-grained text diffs, numerical changes, and table diffs between two pages."""
        text_diffs: List[TextDiffBlock] = []
        num_changes: List[NumericalChangeItem] = []
        tab_diffs: List[TableDiffBlock] = []

        s_num = src_page_rec.page_number if src_page_rec else None
        t_num = tgt_page_rec.page_number if tgt_page_rec else None

        if src_page_rec and not tgt_page_rec:
            cat = classify_change_category(src_page_rec.text[:300], "")
            text_diffs.append(
                TextDiffBlock(
                    source_page=s_num,
                    target_page=None,
                    diff_type="removed",
                    category=cat,
                    source_excerpt=src_page_rec.text[:300],
                    summary=f"Page {s_num} removed in revised document.",
                )
            )
            return text_diffs, num_changes, tab_diffs

        if not src_page_rec and tgt_page_rec:
            cat = classify_change_category("", tgt_page_rec.text[:300])
            text_diffs.append(
                TextDiffBlock(
                    source_page=None,
                    target_page=t_num,
                    diff_type="added",
                    category=cat,
                    target_excerpt=tgt_page_rec.text[:300],
                    summary=f"Page {t_num} added in revised document.",
                )
            )
            return text_diffs, num_changes, tab_diffs

        if src_page_rec and tgt_page_rec:
            src_norm = normalize_page_text(src_page_rec.text)
            tgt_norm = normalize_page_text(tgt_page_rec.text)

            if src_norm != tgt_norm:
                header_footer_pattern = re.compile(
                    r"^(?:page\s+\d+(?:\s+of\s+\d+)?|\d+\s*/\s*\d+|confidential|internal use only)$",
                    re.IGNORECASE,
                )
                src_lines = [l for l in src_norm.splitlines() if not header_footer_pattern.match(l.strip())]
                tgt_lines = [l for l in tgt_norm.splitlines() if not header_footer_pattern.match(l.strip())]

                diff = list(difflib.ndiff(src_lines, tgt_lines))
                added = [line[2:] for line in diff if line.startswith("+ ")]
                removed = [line[2:] for line in diff if line.startswith("- ")]

                if added or removed:
                    src_sub = "\n".join(removed[:4])
                    tgt_sub = "\n".join(added[:4])
                    cat = classify_change_category(src_sub, tgt_sub)
                    text_diffs.append(
                        TextDiffBlock(
                            source_page=s_num,
                            target_page=t_num,
                            diff_type="modified",
                            category=cat,
                            source_excerpt=src_sub,
                            target_excerpt=tgt_sub,
                            summary=f"Page {s_num} modified ({len(added)} lines added, {len(removed)} lines removed). Category: {cat}.",
                        )
                    )

                # Numerical changes
                num_changes.extend(self.extract_numerical_changes(src_norm, tgt_norm, s_num, t_num))

            # Table comparison
            tab_diffs.extend(
                self.compare_tables(src_page_rec.tables or [], tgt_page_rec.tables or [], s_num, t_num)
            )

        return text_diffs, num_changes, tab_diffs

    def generate_ai_explanation(
        self,
        source_doc: Document,
        target_doc: Document,
        num_changes: List[NumericalChangeItem],
        text_diffs: List[TextDiffBlock],
        table_diffs: List[TableDiffBlock],
    ) -> Optional[str]:
        """Generate a concise, professional explanation of key changes using LLM, with deterministic fallback."""
        if not llm_service.is_configured:
            return None

        # Build concise evidence excerpts
        excerpts: List[str] = []
        if num_changes:
            for nc in num_changes[:5]:
                unit_str = f" {nc.units}" if nc.units else ""
                excerpts.append(
                    f"- Parameter change on Page {nc.source_page}->{nc.target_page}: from '{nc.old_value}{unit_str}' to '{nc.new_value}{unit_str}' ({nc.context})"
                )

        if text_diffs:
            for td in text_diffs[:5]:
                if td.source_excerpt or td.target_excerpt:
                    excerpts.append(
                        f"- Content change (Page {td.source_page}->{td.target_page}): Previous: '{td.source_excerpt[:100]}' | Revised: '{td.target_excerpt[:100]}'"
                    )

        if table_diffs:
            for tb in table_diffs[:3]:
                excerpts.append(
                    f"- Table change (Page {tb.source_page}->{tb.target_page}): {tb.change_type} - {tb.details}"
                )

        if not excerpts:
            return "No significant textual, numerical, or tabular differences detected between versions."

        evidence_block = "\n".join(excerpts)
        prompt = (
            f"You are an enterprise document intelligence assistant. Compare Version {source_doc.version_number} "
            f"('{source_doc.original_filename}') and Version {target_doc.version_number} ('{target_doc.original_filename}').\n"
            f"Based strictly on the following detected changes, provide a concise 2-4 sentence executive summary "
            f"explaining the key modifications, parameter updates, and policy/specification shifts:\n\n"
            f"{evidence_block}\n\n"
            f"Do not invent facts. Cite specific pages and values where relevant."
        )

        try:
            chat_model = llm_service.get_llm()
            from langchain_core.messages import HumanMessage, SystemMessage

            response = chat_model.invoke(
                [
                    SystemMessage(
                        content="You summarize document version differences accurately based solely on provided excerpts."
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            content = response.content if hasattr(response, "content") else str(response)
            return content.strip() if content else None
        except Exception as exc:
            logger.warning("LLM change summarization failed: %s. Using deterministic summary.", exc)
            return None

    def compare_versions(
        self,
        db: Session,
        workspace_id: str,
        family_id: str,
        source_document_id: str,
        target_document_id: str,
    ) -> VersionComparison:
        """Compare two document versions within a family, using caching when neither version has changed."""
        family = (
            db.query(DocumentFamily)
            .filter(
                DocumentFamily.id == family_id,
                DocumentFamily.workspace_id == workspace_id,
            )
            .first()
        )
        if not family:
            raise ValueError(f"Document family '{family_id}' not found in workspace '{workspace_id}'.")

        source_doc = (
            db.query(Document)
            .filter(
                Document.id == source_document_id,
                Document.family_id == family_id,
                Document.workspace_id == workspace_id,
            )
            .first()
        )
        target_doc = (
            db.query(Document)
            .filter(
                Document.id == target_document_id,
                Document.family_id == family_id,
                Document.workspace_id == workspace_id,
            )
            .first()
        )

        if not source_doc or not target_doc:
            raise ValueError("Both source and target documents must belong to the specified document family.")

        # Check for cached comparison
        cached_comp = (
            db.query(VersionComparison)
            .filter(
                VersionComparison.family_id == family_id,
                VersionComparison.source_document_id == source_document_id,
                VersionComparison.target_document_id == target_document_id,
            )
            .first()
        )

        # Invalidate cache if either source or target document was updated after the comparison was created
        if cached_comp:
            if cached_comp.created_at >= source_doc.updated_at and cached_comp.created_at >= target_doc.updated_at:
                logger.info(
                    "Returning cached version comparison %s for family %s (v%d -> v%d)",
                    cached_comp.id,
                    family_id,
                    cached_comp.source_version_number,
                    cached_comp.target_version_number,
                )
                return cached_comp
            else:
                db.delete(cached_comp)
                db.flush()

        # Retrieve extracted pages
        src_pages = ingestion_service.get_extracted_pages(db, workspace_id, source_document_id)
        tgt_pages = ingestion_service.get_extracted_pages(db, workspace_id, target_document_id)

        # Map page numbers to records
        src_map = {p.page_number: p for p in src_pages}
        tgt_map = {p.page_number: p for p in tgt_pages}

        # 1. Align pages
        alignments = self.align_pages(src_pages, tgt_pages)

        # 2. Compute fine-grained diffs for aligned pairs
        all_text_diffs: List[TextDiffBlock] = []
        all_num_changes: List[NumericalChangeItem] = []
        all_tab_diffs: List[TableDiffBlock] = []

        unchanged_count = 0
        modified_count = 0
        added_count = 0
        removed_count = 0

        for item in alignments:
            if item.status == "unchanged":
                unchanged_count += 1
            elif item.status == "modified":
                modified_count += 1
                sp = src_map.get(item.source_page) if item.source_page else None
                tp = tgt_map.get(item.target_page) if item.target_page else None
                td, nc, tab = self.diff_aligned_pages(sp, tp)
                all_text_diffs.extend(td)
                all_num_changes.extend(nc)
                all_tab_diffs.extend(tab)
            elif item.status == "added":
                added_count += 1
                tp = tgt_map.get(item.target_page) if item.target_page else None
                td, nc, tab = self.diff_aligned_pages(None, tp)
                all_text_diffs.extend(td)
            elif item.status == "removed":
                removed_count += 1
                sp = src_map.get(item.source_page) if item.source_page else None
                td, nc, tab = self.diff_aligned_pages(sp, None)
                all_text_diffs.extend(td)

        stats = {
            "total_pages_source": len(src_pages),
            "total_pages_target": len(tgt_pages),
            "unchanged_pages": unchanged_count,
            "modified_pages": modified_count,
            "added_pages": added_count,
            "removed_pages": removed_count,
            "text_changes_count": len(all_text_diffs),
            "numerical_changes_count": len(all_num_changes),
            "table_changes_count": len(all_tab_diffs),
        }

        # Deterministic change summary
        deterministic_summary = (
            f"Compared Version {source_doc.version_number} and Version {target_doc.version_number}: "
            f"{unchanged_count} unchanged, {modified_count} modified, {added_count} added, {removed_count} removed pages. "
            f"{len(all_num_changes)} numerical and {len(all_tab_diffs)} table changes detected."
        )

        # AI-Assisted explanation
        ai_exp = self.generate_ai_explanation(
            source_doc=source_doc,
            target_doc=target_doc,
            num_changes=all_num_changes,
            text_diffs=all_text_diffs,
            table_diffs=all_tab_diffs,
        )

        comparison_data = {
            "stats": stats,
            "page_alignments": [a.model_dump() for a in alignments],
            "text_diffs": [td.model_dump() for td in all_text_diffs],
            "table_diffs": [tb.model_dump() for tb in all_tab_diffs],
            "numerical_changes": [nc.model_dump() for nc in all_num_changes],
        }

        comparison = VersionComparison(
            id=str(uuid.uuid4()),
            family_id=family_id,
            source_document_id=source_document_id,
            target_document_id=target_document_id,
            source_version_number=source_doc.version_number,
            target_version_number=target_doc.version_number,
            comparison_data=comparison_data,
            change_summary=deterministic_summary,
            ai_explanation=ai_exp,
        )

        db.add(comparison)
        db.commit()
        db.refresh(comparison)

        logger.info(
            "Created version comparison %s for family %s (v%d -> v%d)",
            comparison.id,
            family_id,
            source_doc.version_number,
            target_doc.version_number,
        )
        return comparison

    def get_comparison_by_id(
        self,
        db: Session,
        workspace_id: str,
        family_id: str,
        comparison_id: str,
    ) -> Optional[VersionComparison]:
        """Retrieve stored version comparison verifying workspace ownership."""
        return (
            db.query(VersionComparison)
            .join(DocumentFamily, VersionComparison.family_id == DocumentFamily.id)
            .filter(
                VersionComparison.id == comparison_id,
                VersionComparison.family_id == family_id,
                DocumentFamily.workspace_id == workspace_id,
            )
            .first()
        )


version_comparison_service = VersionComparisonService()
