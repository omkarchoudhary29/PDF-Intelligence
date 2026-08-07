from datetime import datetime
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from sqlalchemy.orm import Session

from backend.database.models import (
    ContradictionFinding,
    Document,
    DocumentEntityAssociation,
    DocumentFamily,
    DocumentRelationship,
)
from backend.schemas.document import PageExtractionRecord
from backend.services.audit_service import audit_service
from backend.services.ingestion_service import ingestion_service
from backend.services.llm_service import llm_service
from config.settings import get_settings

logger = logging.getLogger("ekn.contradictions")

# Regex patterns for deterministic claim extraction
NUMERICAL_CLAIM_PATTERN = re.compile(
    r"\b([A-Za-z0-9_\-\.\s]{3,35})\b\s+(?:is|must be|shall be|should be|maximum|minimum|limit|set to|measured at|rated for|interval of|frequency of|tolerance is)\s+(?:strictly\s+|approximately\s+|exactly\s+)?([0-9]+(?:[\.,][0-9]+)?)\s*([A-Za-z%°℃℉/]+)?\b",
    re.IGNORECASE,
)

INSTRUCTION_MODAL_PATTERN = re.compile(
    r"\b([A-Za-z0-9_\-\.\s]{3,35})\b\s+(must|shall|is strictly required to|is prohibited from|shall not|must not|may optionally|is recommended to)\s+([^.,;\n]{5,60})",
    re.IGNORECASE,
)


class ContradictionService:
    """Evidence-backed candidate contradiction detection and human review orchestrator."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def find_candidate_document_pairs(
        self,
        db: Session,
        workspace_id: str,
    ) -> List[Tuple[Document, Document, str]]:
        """Identify promising document pairs sharing entities, relationships, or family revisions."""
        candidate_pairs: List[Tuple[Document, Document, str]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        def add_pair(d1: Document, d2: Document, reason: str) -> None:
            if d1.id == d2.id:
                return
            pair_key = (min(d1.id, d2.id), max(d1.id, d2.id))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                candidate_pairs.append((d1, d2, reason))

        # 1. Document Family Revisions (different versions of same family)
        families = db.query(DocumentFamily).filter(DocumentFamily.workspace_id == workspace_id).all()
        for fam in families:
            versions = sorted(fam.versions, key=lambda d: d.version_number)
            for i in range(len(versions)):
                for j in range(i + 1, len(versions)):
                    add_pair(versions[i], versions[j], f"Revisions of family '{fam.title}'")

        # 2. Knowledge Graph: Documents sharing confirmed entities
        shared_entity_query = (
            db.query(
                DocumentEntityAssociation.document_id,
                DocumentEntityAssociation.entity_id,
            )
            .join(Document, DocumentEntityAssociation.document_id == Document.id)
            .filter(Document.workspace_id == workspace_id)
            .all()
        )

        entity_to_docs: Dict[str, List[str]] = {}
        for doc_id, ent_id in shared_entity_query:
            entity_to_docs.setdefault(ent_id, []).append(doc_id)

        doc_cache: Dict[str, Document] = {}

        def get_doc(doc_id: str) -> Optional[Document]:
            if doc_id not in doc_cache:
                d = db.query(Document).filter(Document.id == doc_id).first()
                if d:
                    doc_cache[doc_id] = d
            return doc_cache.get(doc_id)

        for ent_id, doc_ids in entity_to_docs.items():
            unique_docs = list(dict.fromkeys(doc_ids))
            if len(unique_docs) > 1:
                for i in range(len(unique_docs)):
                    for j in range(i + 1, len(unique_docs)):
                        d1 = get_doc(unique_docs[i])
                        d2 = get_doc(unique_docs[j])
                        if d1 and d2:
                            add_pair(d1, d2, "Shared knowledge graph entity")

        # 3. Knowledge Graph: Direct Document Relationships
        rels = (
            db.query(DocumentRelationship)
            .filter(DocumentRelationship.workspace_id == workspace_id)
            .all()
        )
        for rel in rels:
            d1 = get_doc(rel.source_document_id)
            d2 = get_doc(rel.target_document_id)
            if d1 and d2:
                add_pair(d1, d2, f"Direct relationship: {rel.relationship_type}")

        # 4. Fallback for unlinked or small workspaces: compare completed workspace document pairs
        if not candidate_pairs:
            all_docs = (
                db.query(Document)
                .filter(
                    Document.workspace_id == workspace_id,
                    Document.processing_status == "completed",
                )
                .all()
            )
            for i in range(len(all_docs)):
                for j in range(i + 1, min(len(all_docs), i + 10)):
                    add_pair(all_docs[i], all_docs[j], "Workspace document pair comparison")

        logger.info(
            "Identified %d candidate document pairs for contradiction evaluation in workspace %s",
            len(candidate_pairs),
            workspace_id,
        )
        return candidate_pairs

    def extract_claims_from_pages(
        self,
        pages: List[PageExtractionRecord],
    ) -> List[Dict[str, Any]]:
        """Extract structured factual claims from extracted page text."""
        claims: List[Dict[str, Any]] = []

        for p in pages:
            page_text = getattr(p, "text", None) or getattr(p, "raw_text", "") or ""
            lines = page_text.splitlines()
            for line in lines:
                line_clean = line.strip()
                if not line_clean or len(line_clean) < 15:
                    continue

                # Numerical claims
                for m in NUMERICAL_CLAIM_PATTERN.finditer(line_clean):
                    subj = m.group(1).strip()
                    val = m.group(2).strip()
                    unit = (m.group(3) or "").strip()
                    claims.append(
                        {
                            "type": "numerical",
                            "subject": subj.lower(),
                            "predicate": "operating_value",
                            "value": val,
                            "unit": unit.lower(),
                            "page_number": p.page_number,
                            "excerpt": line_clean[:250],
                        }
                    )

                # Instruction / Modal claims
                for m in INSTRUCTION_MODAL_PATTERN.finditer(line_clean):
                    subj = m.group(1).strip()
                    modal = m.group(2).strip().lower()
                    action = m.group(3).strip()
                    polarity = "prohibited" if "not" in modal or "prohibited" in modal else "required"
                    claims.append(
                        {
                            "type": "instruction",
                            "subject": subj.lower(),
                            "predicate": action.lower()[:50],
                            "value": polarity,
                            "unit": "",
                            "page_number": p.page_number,
                            "excerpt": line_clean[:250],
                        }
                    )

        return claims

    def detect_candidate_contradictions(
        self,
        db: Session,
        workspace_id: str,
        doc1: Document,
        doc2: Document,
        pair_reason: str,
    ) -> List[Dict[str, Any]]:
        """Stage A & B: Compare extracted claims across two documents and classify contradictions."""
        pages1 = ingestion_service.get_extracted_pages(db, workspace_id, doc1.id)
        pages2 = ingestion_service.get_extracted_pages(db, workspace_id, doc2.id)

        if not pages1 or not pages2:
            return []

        claims1 = self.extract_claims_from_pages(pages1)
        claims2 = self.extract_claims_from_pages(pages2)

        raw_candidates: List[Dict[str, Any]] = []

        # Check for matching subjects with conflicting predicates/values
        for c1 in claims1:
            for c2 in claims2:
                # Subject similarity check
                s1, s2 = c1["subject"], c2["subject"]
                if s1 == s2 or (len(s1) > 4 and (s1 in s2 or s2 in s1)):
                    # Check for numerical discrepancy
                    if c1["type"] == "numerical" and c2["type"] == "numerical":
                        val1, val2 = c1["value"], c2["value"]
                        unit1, unit2 = c1["unit"], c2["unit"]

                        if val1 != val2:
                            # Contextual check: Are these different versions of the same family?
                            is_same_family = doc1.family_id and (doc1.family_id == doc2.family_id)
                            is_version_superseded = (
                                doc1.version_status == "superseded" or doc2.version_status == "superseded"
                            )

                            if is_same_family and is_version_superseded:
                                finding_type = "dated_revision"
                                explanation = (
                                    f"Version update: {doc1.original_filename} (v{doc1.version_number}) specifies "
                                    f"{val1} {unit1}, whereas revised {doc2.original_filename} (v{doc2.version_number}) "
                                    f"updates this requirement to {val2} {unit2}."
                                )
                            else:
                                finding_type = "numerical_discrepancy"
                                explanation = (
                                    f"Numerical discrepancy for '{s1}': {doc1.original_filename} specifies {val1} {unit1} "
                                    f"(Page {c1['page_number']}), whereas {doc2.original_filename} specifies {val2} {unit2} "
                                    f"(Page {c2['page_number']})."
                                )

                            raw_candidates.append(
                                {
                                    "subject": s1.title(),
                                    "predicate": c1["predicate"],
                                    "finding_type": finding_type,
                                    "explanation": explanation,
                                    "first_document_id": doc1.id,
                                    "first_page": c1["page_number"],
                                    "first_evidence_excerpt": c1["excerpt"],
                                    "second_document_id": doc2.id,
                                    "second_page": c2["page_number"],
                                    "second_evidence_excerpt": c2["excerpt"],
                                    "confidence_score": 0.90 if finding_type == "numerical_discrepancy" else 0.80,
                                }
                            )

                    # Check for instruction conflict
                    elif c1["type"] == "instruction" and c2["type"] == "instruction":
                        if c1["value"] != c2["value"] and c1["predicate"] == c2["predicate"]:
                            raw_candidates.append(
                                {
                                    "subject": s1.title(),
                                    "predicate": c1["predicate"],
                                    "finding_type": "instruction_conflict",
                                    "explanation": (
                                        f"Instruction conflict for '{s1}': {doc1.original_filename} states requirement is "
                                        f"'{c1['value']}', whereas {doc2.original_filename} states it is '{c2['value']}'."
                                    ),
                                    "first_document_id": doc1.id,
                                    "first_page": c1["page_number"],
                                    "first_evidence_excerpt": c1["excerpt"],
                                    "second_document_id": doc2.id,
                                    "second_page": c2["page_number"],
                                    "second_evidence_excerpt": c2["excerpt"],
                                    "confidence_score": 0.85,
                                }
                            )

        # Stage B: LLM Validation & Refinement if available
        validated_candidates: List[Dict[str, Any]] = []
        for cand in raw_candidates[:6]:  # Bounded set per pair
            cand = self.validate_candidate_with_llm(cand, doc1, doc2)
            if cand and cand.get("finding_type") != "not_a_contradiction":
                validated_candidates.append(cand)

        return validated_candidates

    def validate_candidate_with_llm(
        self,
        candidate: Dict[str, Any],
        doc1: Document,
        doc2: Document,
    ) -> Dict[str, Any]:
        """Optionally validate and refine candidate conflict using application LLM."""
        if not llm_service.is_configured:
            return candidate

        prompt = (
            f"You are a regulatory and compliance auditor. Evaluate the following potential factual conflict:\n"
            f"Subject: {candidate['subject']}\n"
            f"Statement 1 (Doc: {doc1.original_filename}, Page {candidate['first_page']}): '{candidate['first_evidence_excerpt']}'\n"
            f"Statement 2 (Doc: {doc2.original_filename}, Page {candidate['second_page']}): '{candidate['second_evidence_excerpt']}'\n\n"
            f"Determine the true relationship. Choose exactly one category:\n"
            f"- 'numerical_discrepancy' (unresolved conflicting numbers for the exact same condition)\n"
            f"- 'instruction_conflict' (direct policy or procedure contradiction)\n"
            f"- 'dated_revision' (one document is a newer official update/revision of the other)\n"
            f"- 'context_dependent_difference' (difference is due to different models, ambient conditions, or units)\n"
            f"- 'not_a_contradiction' (statements are compatible or unrelated)\n\n"
            f"Respond with JSON: {{\"category\": \"...\", \"explanation\": \"...\"}}"
        )

        try:
            chat_model = llm_service.get_llm()
            from langchain_core.messages import HumanMessage, SystemMessage

            response = chat_model.invoke(
                [
                    SystemMessage(content="You evaluate factual consistency between enterprise documents."),
                    HumanMessage(content=prompt),
                ]
            )
            content = response.content if hasattr(response, "content") else str(response)
            # Parse JSON
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                category = data.get("category")
                explanation = data.get("explanation")
                valid_cats = {
                    "numerical_discrepancy",
                    "instruction_conflict",
                    "dated_revision",
                    "context_dependent_difference",
                    "not_a_contradiction",
                }
                if category in valid_cats:
                    candidate["finding_type"] = category
                if explanation:
                    candidate["explanation"] = explanation
        except Exception as exc:
            logger.debug("LLM contextual validation skipped or failed: %s", exc)

        return candidate

    def scan_workspace_contradictions(
        self,
        db: Session,
        workspace_id: str,
    ) -> List[ContradictionFinding]:
        """Scan promising document pairs in a workspace and persist unique candidate findings."""
        pairs = self.find_candidate_document_pairs(db, workspace_id)
        created_findings: List[ContradictionFinding] = []

        for doc1, doc2, reason in pairs:
            candidates = self.detect_candidate_contradictions(db, workspace_id, doc1, doc2, reason)
            for c in candidates:
                # Deduplication check: Do not insert duplicate candidate finding
                existing = (
                    db.query(ContradictionFinding)
                    .filter(
                        ContradictionFinding.workspace_id == workspace_id,
                        (
                            (
                                (ContradictionFinding.first_document_id == c["first_document_id"])
                                & (ContradictionFinding.second_document_id == c["second_document_id"])
                            )
                            | (
                                (ContradictionFinding.first_document_id == c["second_document_id"])
                                & (ContradictionFinding.second_document_id == c["first_document_id"])
                            )
                        ),
                        ContradictionFinding.subject == c["subject"],
                    )
                    .first()
                )

                if existing:
                    continue

                finding = ContradictionFinding(
                    id=str(uuid.uuid4()),
                    workspace_id=workspace_id,
                    subject=c["subject"],
                    predicate=c["predicate"],
                    finding_type=c["finding_type"],
                    explanation=c["explanation"],
                    first_document_id=c["first_document_id"],
                    first_page=c["first_page"],
                    first_evidence_excerpt=c["first_evidence_excerpt"],
                    second_document_id=c["second_document_id"],
                    second_page=c["second_page"],
                    second_evidence_excerpt=c["second_evidence_excerpt"],
                    confidence_score=c.get("confidence_score", 0.85),
                    review_status="candidate",
                )
                db.add(finding)
                created_findings.append(finding)

        if created_findings:
            db.commit()
            logger.info(
                "Created %d new contradiction candidates in workspace %s",
                len(created_findings),
                workspace_id,
            )

        # Return all findings in workspace
        return (
            db.query(ContradictionFinding)
            .filter(ContradictionFinding.workspace_id == workspace_id)
            .order_by(ContradictionFinding.created_at.desc())
            .all()
        )

    def list_findings(
        self,
        db: Session,
        workspace_id: str,
        review_status: Optional[str] = None,
        finding_type: Optional[str] = None,
    ) -> List[ContradictionFinding]:
        """List findings in workspace with optional filters."""
        query = db.query(ContradictionFinding).filter(ContradictionFinding.workspace_id == workspace_id)
        if review_status:
            query = query.filter(ContradictionFinding.review_status == review_status)
        if finding_type:
            query = query.filter(ContradictionFinding.finding_type == finding_type)
        return query.order_by(ContradictionFinding.created_at.desc()).all()

    def get_finding_by_id(
        self,
        db: Session,
        workspace_id: str,
        finding_id: str,
    ) -> Optional[ContradictionFinding]:
        """Retrieve finding by ID scoped to workspace."""
        return (
            db.query(ContradictionFinding)
            .filter(
                ContradictionFinding.id == finding_id,
                ContradictionFinding.workspace_id == workspace_id,
            )
            .first()
        )

    def review_finding(
        self,
        db: Session,
        workspace_id: str,
        finding_id: str,
        review_status: str,
        review_reason: Optional[str] = None,
        authoritative_document_id: Optional[str] = None,
        reviewer_user_id: Optional[str] = None,
    ) -> ContradictionFinding:
        """Authorized review of a contradiction finding: confirm, dismiss, or resolve."""
        status_norm = review_status.strip().lower().replace(" ", "_").replace("-", "_")
        status_aliases = {
            "confirmed_issue": "confirmed",
            "pending_review": "candidate",
            "not_contradiction": "not_a_contradiction",
        }
        status_norm = status_aliases.get(status_norm, status_norm)

        allowed_statuses = {
            "candidate",
            "under_review",
            "confirmed",
            "dismissed",
            "not_a_contradiction",
            "resolved",
        }
        if status_norm not in allowed_statuses:
            raise ValueError(f"Invalid review status '{review_status}'. Allowed: {allowed_statuses}")

        finding = self.get_finding_by_id(db, workspace_id, finding_id)
        if not finding:
            raise ValueError(f"Contradiction finding '{finding_id}' not found in workspace '{workspace_id}'.")

        # Validate authoritative document if supplied
        if authoritative_document_id:
            auth_doc = (
                db.query(Document)
                .filter(
                    Document.id == authoritative_document_id,
                    Document.workspace_id == workspace_id,
                )
                .first()
            )
            if not auth_doc:
                raise ValueError(
                    f"Authoritative document '{authoritative_document_id}' not found in workspace '{workspace_id}'."
                )
            finding.authoritative_document_id = authoritative_document_id

        finding.review_status = status_norm
        finding.review_reason = review_reason
        finding.reviewer_user_id = reviewer_user_id
        db.commit()
        db.refresh(finding)

        audit_service.log_action(
            db=db,
            action="contradiction_finding_reviewed",
            outcome="success",
            actor_user_id=reviewer_user_id,
            workspace_id=workspace_id,
            resource_type="contradiction_finding",
            resource_id=finding_id,
            details_json={
                "review_status": status_norm,
                "review_reason": review_reason,
                "authoritative_document_id": authoritative_document_id,
            },
        )
        logger.info(
            "Reviewed contradiction finding %s in workspace %s: status set to '%s'",
            finding_id,
            workspace_id,
            status_norm,
        )
        return finding


contradiction_service = ContradictionService()
