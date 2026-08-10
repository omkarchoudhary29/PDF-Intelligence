"""Relationship Discovery Engine for Enterprise Knowledge Navigator.

Discovers explainable, evidence-backed cross-document relationships:
1. Shared Entity relationships (common confirmed entities)
2. Direct Reference relationships (explicit document filename or citation mentions)
3. Related Topic relationships (calibrated semantic cosine similarity)
4. Document Connectivity Score computation
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from sqlalchemy.orm import Session

from backend.database.models import (
    Document,
    DocumentEntityAssociation,
    DocumentRelationship,
    Entity,
    EntityRelationship,
)
from backend.services.embedding_service import embedding_service
from backend.services.llm_service import LLMError, LLMNotConfiguredError, llm_service

logger = logging.getLogger("ekn.graph.relationship_discovery")

# Calibrated cosine similarity threshold for semantic topic overlap
SEMANTIC_TOPIC_THRESHOLD = 0.72

# Deterministic patterns for entity-to-entity relationship extraction
ENTITY_RELATION_PATTERNS: List[Tuple[str, re.Pattern, float]] = [
    ("PART_OF", re.compile(r"\b(?:is\s+a\s+)?(?:part|component|subsystem|module|element)\s+of\b|\b(?:fitted|installed|integrated)\s+(?:in|into|on)\b", re.IGNORECASE), 0.95),
    ("HAS_COMPONENT", re.compile(r"\b(?:includes|comprises|contains|houses|incorporates|consists\s+of)\b", re.IGNORECASE), 0.90),
    ("CONNECTED_TO", re.compile(r"\b(?:connected\s+to|couples?\s+with|interfaces?\s+with|linked\s+to|attached\s+to|wires?\s+into)\b", re.IGNORECASE), 0.95),
    ("OPERATES", re.compile(r"\b(?:operates|drives|actuates|powers|controls|regulates|governs)\b", re.IGNORECASE), 0.90),
    ("USED_FOR", re.compile(r"\b(?:used\s+for|designed\s+for|utilised\s+for|utilized\s+for|employed\s+for|serves\s+to)\b", re.IGNORECASE), 0.85),
    ("LOCATED_IN", re.compile(r"\b(?:located\s+in|situated\s+in|positioned\s+in|mounted\s+on|housed\s+in)\b", re.IGNORECASE), 0.90),
    ("MAINTAINED_BY", re.compile(r"\b(?:maintained\s+by|serviced\s+by|overhauled\s+by|inspected\s+by)\b", re.IGNORECASE), 0.90),
    ("REQUIRES", re.compile(r"\b(?:requires|mandates|necessitates|calls\s+for)\b", re.IGNORECASE), 0.90),
    ("DEPENDS_ON", re.compile(r"\b(?:depends\s+on|dependent\s+on|reliant\s+upon|contingent\s+on)\b", re.IGNORECASE), 0.90),
    ("REGULATED_BY", re.compile(r"\b(?:regulated\s+by|complies\s+with|in\s+accordance\s+with|per\s+standard|governed\s+by)\b", re.IGNORECASE), 0.95),
    ("CAUSES", re.compile(r"\b(?:causes|leads\s+to|results\s+in|triggers|induces)\b", re.IGNORECASE), 0.85),
    ("PREVENTS", re.compile(r"\b(?:prevents|mitigates|inhibits|protects\s+against|avoids)\b", re.IGNORECASE), 0.90),
    ("MEASURES", re.compile(r"\b(?:measures|monitors|detects|senses|gauges)\b", re.IGNORECASE), 0.90),
    ("REPLACES", re.compile(r"\b(?:replaces|supersedes|substitutes\s+for)\b", re.IGNORECASE), 0.90),
]


class RelationshipDiscoveryService:
    """Discovers, explains, and persists cross-document relationships within a workspace."""

    def discover_shared_entity_relationships(
        self,
        db: Session,
        workspace_id: str,
        target_document_id: Optional[str] = None,
    ) -> List[DocumentRelationship]:
        """Discover documents sharing identical confirmed entities."""
        # Query associations for target document or entire workspace
        query = (
            db.query(DocumentEntityAssociation, Entity, Document)
            .join(Entity, DocumentEntityAssociation.entity_id == Entity.id)
            .join(Document, DocumentEntityAssociation.document_id == Document.id)
            .filter(DocumentEntityAssociation.workspace_id == workspace_id)
        )

        all_records = query.all()

        # Group associations by entity_id: entity_id -> list of (doc_id, doc_name, page, excerpt, entity_name, entity_type)
        entity_map: Dict[str, List[Dict[str, Any]]] = {}
        for assoc, ent, doc in all_records:
            if assoc.entity_id not in entity_map:
                entity_map[assoc.entity_id] = []
            entity_map[assoc.entity_id].append({
                "document_id": doc.id,
                "document_name": doc.original_filename,
                "page_start": assoc.page_start,
                "page_end": assoc.page_end,
                "excerpt": assoc.excerpt,
                "entity_name": ent.canonical_name,
                "entity_type": ent.entity_type,
            })

        relationships: List[DocumentRelationship] = []

        # Track relationships in this batch to prevent duplicate inserts for multiple shared entities
        existing_rels_list = (
            db.query(DocumentRelationship)
            .filter(
                DocumentRelationship.workspace_id == workspace_id,
                DocumentRelationship.relationship_type == "shared_entity",
            )
            .all()
        )
        tracked_rels: Dict[Tuple[str, str], DocumentRelationship] = {
            (r.source_document_id, r.target_document_id): r for r in existing_rels_list
        }

        # Find pairs of documents sharing each entity
        for entity_id, mentions in entity_map.items():
            doc_ids = list({m["document_id"] for m in mentions})
            if len(doc_ids) < 2:
                continue

            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    d1, d2 = doc_ids[i], doc_ids[j]

                    # Filter by target document if specified
                    if target_document_id and target_document_id not in (d1, d2):
                        continue

                    # Canonical ordering for undirected shared_entity
                    src_id, tgt_id = (d1, d2) if d1 < d2 else (d2, d1)

                    m1 = next(m for m in mentions if m["document_id"] == src_id)
                    m2 = next(m for m in mentions if m["document_id"] == tgt_id)

                    entity_name = m1["entity_name"]
                    entity_type = m1["entity_type"]

                    explanation = (
                        f"Both documents discuss the {entity_type} '{entity_name}'. "
                        f"Found in '{m1['document_name']}' (Page {m1['page_start'] or 1}) "
                        f"and '{m2['document_name']}' (Page {m2['page_start'] or 1})."
                    )

                    supporting_evidence = [
                        {
                            "document_id": src_id,
                            "filename": m1["document_name"],
                            "page": m1["page_start"],
                            "excerpt": m1["excerpt"],
                            "entity": entity_name,
                        },
                        {
                            "document_id": tgt_id,
                            "filename": m2["document_name"],
                            "page": m2["page_start"],
                            "excerpt": m2["excerpt"],
                            "entity": entity_name,
                        },
                    ]

                    # Check tracked relationship
                    rel_pair = (src_id, tgt_id)
                    existing = tracked_rels.get(rel_pair)

                    if existing:
                        existing.explanation = explanation
                        cur_evidence = list(existing.supporting_evidence or [])
                        cur_evidence.extend(supporting_evidence)
                        existing.supporting_evidence = cur_evidence[:10]
                        if existing not in relationships:
                            relationships.append(existing)
                    else:
                        rel = DocumentRelationship(
                            workspace_id=workspace_id,
                            source_document_id=src_id,
                            target_document_id=tgt_id,
                            relationship_type="shared_entity",
                            confidence_score=0.90,
                            explanation=explanation,
                            supporting_evidence=supporting_evidence,
                            source_page_start=m1["page_start"],
                            target_page_start=m2["page_start"],
                        )
                        db.add(rel)
                        tracked_rels[rel_pair] = rel
                        relationships.append(rel)

        db.commit()
        return relationships

    def discover_direct_reference_relationships(
        self,
        db: Session,
        workspace_id: str,
        source_doc: Document,
        pages_records: List[Dict[str, Any]],
        all_workspace_docs: List[Document],
    ) -> List[DocumentRelationship]:
        """Detect explicit citations or mentions of other workspace documents in source document text."""
        relationships: List[DocumentRelationship] = []

        # Build target lookup dictionary
        target_candidates: Dict[str, Document] = {}
        for other in all_workspace_docs:
            if other.id == source_doc.id:
                continue
            # Match by original filename or display name
            target_candidates[other.original_filename.lower()] = other
            if other.display_name:
                target_candidates[other.display_name.lower()] = other

        for page in pages_records:
            if isinstance(page, dict):
                raw_text = page.get("text") or page.get("raw_text", "")
                page_num = page.get("page_number", 1)
            else:
                raw_text = getattr(page, "text", "") or getattr(page, "raw_text", "") or ""
                page_num = getattr(page, "page_number", 1)
            raw_text_lower = raw_text.lower()

            for target_name, target_doc in target_candidates.items():
                if len(target_name) > 4 and target_name in raw_text_lower:
                    # Found explicit direct citation
                    idx = raw_text_lower.find(target_name)
                    start = max(0, idx - 40)
                    end = min(len(raw_text), idx + len(target_name) + 60)
                    excerpt = raw_text[start:end].strip()

                    explanation = (
                        f"Document '{source_doc.original_filename}' (Page {page_num}) "
                        f"explicitly references '{target_doc.original_filename}'."
                    )
                    evidence = [{
                        "source_document_id": source_doc.id,
                        "source_filename": source_doc.original_filename,
                        "page": page_num,
                        "excerpt": excerpt,
                    }]

                    # Check existing
                    existing = (
                        db.query(DocumentRelationship)
                        .filter(
                            DocumentRelationship.workspace_id == workspace_id,
                            DocumentRelationship.source_document_id == source_doc.id,
                            DocumentRelationship.target_document_id == target_doc.id,
                            DocumentRelationship.relationship_type == "direct_reference",
                        )
                        .first()
                    )

                    if existing:
                        existing.explanation = explanation
                        existing.supporting_evidence = evidence
                        relationships.append(existing)
                    else:
                        rel = DocumentRelationship(
                            workspace_id=workspace_id,
                            source_document_id=source_doc.id,
                            target_document_id=target_doc.id,
                            relationship_type="direct_reference",
                            confidence_score=1.0,
                            explanation=explanation,
                            supporting_evidence=evidence,
                            source_page_start=page_num,
                            target_page_start=1,
                        )
                        db.add(rel)
                        relationships.append(rel)

        db.commit()
        return relationships

    def discover_related_topics_via_embeddings(
        self,
        db: Session,
        workspace_id: str,
        source_doc: Document,
        source_text_summary: str,
        other_docs: List[Document],
        other_texts: Dict[str, str],
        threshold: float = SEMANTIC_TOPIC_THRESHOLD,
    ) -> List[DocumentRelationship]:
        """Discover semantic topic overlap using Sentence Transformer embeddings."""
        relationships: List[DocumentRelationship] = []
        if not other_docs or not source_text_summary.strip():
            return relationships

        # Compute source embedding
        source_emb = embedding_service.embed_query(source_text_summary[:2000])

        for other in other_docs:
            if other.id == source_doc.id:
                continue
            other_text = other_texts.get(other.id, "")
            if not other_text.strip():
                continue

            other_emb = embedding_service.embed_query(other_text[:2000])

            # Cosine similarity between L2-normalized vectors is dot product
            sim = float(np.dot(source_emb, other_emb))

            if sim >= threshold:
                src_id, tgt_id = (source_doc.id, other.id) if source_doc.id < other.id else (other.id, source_doc.id)
                explanation = (
                    f"High semantic topical overlap (cosine similarity: {sim:.2f}) "
                    f"between '{source_doc.original_filename}' and '{other.original_filename}'. "
                    f"Note: This is a similarity-based connection, not a verified factual dependency."
                )

                evidence = [{
                    "similarity": round(sim, 3),
                    "model": "all-MiniLM-L6-v2",
                    "note": "Semantic topical correspondence based on dense text embeddings.",
                }]

                existing = (
                    db.query(DocumentRelationship)
                    .filter(
                        DocumentRelationship.workspace_id == workspace_id,
                        DocumentRelationship.source_document_id == src_id,
                        DocumentRelationship.target_document_id == tgt_id,
                        DocumentRelationship.relationship_type == "related_topic",
                    )
                    .first()
                )

                if existing:
                    existing.confidence_score = round(sim, 3)
                    existing.explanation = explanation
                    relationships.append(existing)
                else:
                    rel = DocumentRelationship(
                        workspace_id=workspace_id,
                        source_document_id=src_id,
                        target_document_id=tgt_id,
                        relationship_type="related_topic",
                        confidence_score=round(sim, 3),
                        explanation=explanation,
                        supporting_evidence=evidence,
                    )
                    db.add(rel)
                    relationships.append(rel)

        db.commit()
        return relationships

    def calculate_document_connectivity_score(
        self,
        db: Session,
        workspace_id: str,
        document_id: str,
    ) -> float:
        """Calculate explainable document connectivity score based on supported relationships:

        Score Formula:
        Connectivity = 3.0 * (direct_references) + 2.0 * (shared_entities) + 1.0 * (related_topics)
        Distinguishes structural/factual references from semantic topical similarity.
        """
        rels = (
            db.query(DocumentRelationship)
            .filter(
                DocumentRelationship.workspace_id == workspace_id,
                (DocumentRelationship.source_document_id == document_id) |
                (DocumentRelationship.target_document_id == document_id),
            )
            .all()
        )

        direct_refs = sum(1 for r in rels if r.relationship_type == "direct_reference")
        shared_ents = sum(1 for r in rels if r.relationship_type == "shared_entity")
        topics = sum(1 for r in rels if r.relationship_type == "related_topic")

        return float(3.0 * direct_refs + 2.0 * shared_ents + 1.0 * topics)

    def discover_entity_relationships(
        self,
        db: Session,
        workspace_id: str,
        document: Document,
        pages_records: List[Any],
        use_llm: bool = True,
    ) -> List[EntityRelationship]:
        """Discover and persist direct entity-to-entity relationships established in this document.

        Every EntityRelationship record stores:
          - workspace_id
          - source_entity_id
          - target_entity_id
          - source_document_id
          - relationship_type (PART_OF, CONNECTED_TO, OPERATES, USED_FOR, LOCATED_IN, MAINTAINED_BY,
                               REQUIRES, DEPENDS_ON, HAS_COMPONENT, REGULATED_BY, ASSOCIATED_WITH,
                               CAUSES, PREVENTS, MEASURES, REPLACES)
          - confidence_score
          - explanation
          - evidence_excerpt
          - page_number

        Supports multiple independent evidence records across distinct company documents.
        """
        created_or_updated_rels: List[EntityRelationship] = []

        # 1. Fetch all entity associations for this document
        associations = (
            db.query(DocumentEntityAssociation, Entity)
            .join(Entity, DocumentEntityAssociation.entity_id == Entity.id)
            .filter(
                DocumentEntityAssociation.workspace_id == workspace_id,
                DocumentEntityAssociation.document_id == document.id,
            )
            .all()
        )

        if not associations:
            logger.info("No entity associations found for document %s to build entity relationships", document.id)
            return []

        # Map entities by page
        page_to_entities: Dict[int, List[Entity]] = {}
        all_doc_entities: Dict[str, Entity] = {}
        for assoc, ent in associations:
            p_num = assoc.page_start or 1
            if p_num not in page_to_entities:
                page_to_entities[p_num] = []
            if ent not in page_to_entities[p_num]:
                page_to_entities[p_num].append(ent)
            all_doc_entities[ent.id] = ent

        # 2. Iterate through each page record
        for page in pages_records:
            if isinstance(page, dict):
                raw_text = page.get("text") or page.get("raw_text") or ""
                page_num = page.get("page_number", 1)
            else:
                raw_text = getattr(page, "text", "") or getattr(page, "raw_text", "") or ""
                page_num = getattr(page, "page_number", 1)

            if not raw_text.strip():
                continue

            entities_on_page = list(page_to_entities.get(page_num, []))
            raw_text_lower = raw_text.lower()
            for ent in all_doc_entities.values():
                if ent not in entities_on_page:
                    if ent.canonical_name.lower() in raw_text_lower:
                        entities_on_page.append(ent)
                    elif ent.aliases_json and isinstance(ent.aliases_json, list):
                        if any(str(al).lower() in raw_text_lower for al in ent.aliases_json if al):
                            entities_on_page.append(ent)

            if len(entities_on_page) < 2:
                continue

            # Split text into sentences for co-occurrence inspection
            sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", raw_text) if len(s.strip()) > 15]

            # Candidate pairs to discover
            for i in range(len(entities_on_page)):
                for j in range(i + 1, len(entities_on_page)):
                    ent_a = entities_on_page[i]
                    ent_b = entities_on_page[j]

                    names_a = [ent_a.canonical_name.lower()] + [str(al).lower() for al in (ent_a.aliases_json or []) if al]
                    names_b = [ent_b.canonical_name.lower()] + [str(al).lower() for al in (ent_b.aliases_json or []) if al]

                    matched_sentence = None
                    for sent in sentences:
                        sent_lower = sent.lower()
                        has_a = any(na in sent_lower for na in names_a)
                        has_b = any(nb in sent_lower for nb in names_b)
                        if has_a and has_b:
                            matched_sentence = sent
                            break

                    if not matched_sentence:
                        continue

                    # Determine relationship type and direction from sentence
                    sent_lower = matched_sentence.lower()
                    pos_a = min((sent_lower.find(na) for na in names_a if na in sent_lower), default=0)
                    pos_b = min((sent_lower.find(nb) for nb in names_b if nb in sent_lower), default=0)

                    src_ent, tgt_ent = (ent_a, ent_b) if pos_a <= pos_b else (ent_b, ent_a)

                    rel_type = "ASSOCIATED_WITH"
                    confidence = 0.70

                    for p_type, p_regex, p_conf in ENTITY_RELATION_PATTERNS:
                        if p_regex.search(matched_sentence):
                            rel_type = p_type
                            confidence = p_conf
                            break

                    # Check for Regulation / Standard types
                    if src_ent.entity_type in ("Regulation", "Standard") and tgt_ent.entity_type not in ("Regulation", "Standard"):
                        src_ent, tgt_ent = tgt_ent, src_ent
                        rel_type = "REGULATED_BY"
                        confidence = 0.95
                    elif tgt_ent.entity_type in ("Regulation", "Standard") and src_ent.entity_type not in ("Regulation", "Standard"):
                        rel_type = "REGULATED_BY"
                        confidence = 0.95

                    explanation = (
                        f"'{src_ent.canonical_name}' has relationship '{rel_type}' with '{tgt_ent.canonical_name}' "
                        f"as documented in '{document.original_filename}' (Page {page_num})."
                    )
                    evidence_excerpt = matched_sentence[:450]

                    # Check if exact evidence record already exists for this document and page
                    existing_rel = (
                        db.query(EntityRelationship)
                        .filter(
                            EntityRelationship.workspace_id == workspace_id,
                            EntityRelationship.source_entity_id == src_ent.id,
                            EntityRelationship.target_entity_id == tgt_ent.id,
                            EntityRelationship.relationship_type == rel_type,
                            EntityRelationship.source_document_id == document.id,
                            EntityRelationship.page_number == page_num,
                        )
                        .first()
                    )

                    if existing_rel:
                        existing_rel.confidence_score = max(existing_rel.confidence_score, confidence)
                        existing_rel.explanation = explanation
                        existing_rel.evidence_excerpt = evidence_excerpt
                        created_or_updated_rels.append(existing_rel)
                    else:
                        new_rel = EntityRelationship(
                            workspace_id=workspace_id,
                            source_entity_id=src_ent.id,
                            target_entity_id=tgt_ent.id,
                            source_document_id=document.id,
                            relationship_type=rel_type,
                            confidence_score=confidence,
                            explanation=explanation,
                            evidence_excerpt=evidence_excerpt,
                            page_number=page_num,
                        )
                        db.add(new_rel)
                        created_or_updated_rels.append(new_rel)

        # 3. LLM-assisted relationship extraction if enabled
        if use_llm and llm_service.is_configured and len(all_doc_entities) >= 2:
            try:
                for page in pages_records[:5]:  # Bounded
                    p_num = page.get("page_number", 1) if isinstance(page, dict) else getattr(page, "page_number", 1)
                    p_text = page.get("text", "") if isinstance(page, dict) else getattr(page, "text", "")
                    if len(p_text.strip()) < 100:
                        continue

                    ents_on_p = page_to_entities.get(p_num, [])
                    if len(ents_on_p) < 2:
                        continue

                    ent_names_list = [e.canonical_name for e in ents_on_p]
                    prompt = (
                        "Analyze the following document passage and extract factual relationships between these specific entities: "
                        f"{', '.join(ent_names_list)}.\n\n"
                        "Allowed relationship types: PART_OF, CONNECTED_TO, OPERATES, USED_FOR, LOCATED_IN, MAINTAINED_BY, "
                        "REQUIRES, DEPENDS_ON, HAS_COMPONENT, REGULATED_BY, ASSOCIATED_WITH, CAUSES, PREVENTS, MEASURES, REPLACES.\n\n"
                        "Return a valid JSON object only:\n"
                        "{\n"
                        '  "relationships": [\n'
                        '    {"source": "...", "target": "...", "relationship_type": "...", "explanation": "...", "evidence": "exact quote"}\n'
                        "  ]\n"
                        "}\n\n"
                        f"Passage (Page {p_num}):\n{p_text[:2500]}"
                    )

                    response = llm_service.generate_json(prompt=prompt, system_instruction="Extract factual relationships between entities in valid JSON.")
                    items = response.get("relationships", []) if isinstance(response, dict) else []

                    name_to_ent = {e.canonical_name.lower(): e for e in ents_on_p}
                    for item in items:
                        src_name = str(item.get("source", "")).strip().lower()
                        tgt_name = str(item.get("target", "")).strip().lower()
                        r_type = str(item.get("relationship_type", "ASSOCIATED_WITH")).strip().upper()
                        ev = str(item.get("evidence", "")).strip()[:450]
                        expl = str(item.get("explanation", "")).strip()

                        s_ent = name_to_ent.get(src_name)
                        t_ent = name_to_ent.get(tgt_name)

                        if s_ent and t_ent and s_ent.id != t_ent.id and ev:
                            existing = (
                                db.query(EntityRelationship)
                                .filter(
                                    EntityRelationship.workspace_id == workspace_id,
                                    EntityRelationship.source_entity_id == s_ent.id,
                                    EntityRelationship.target_entity_id == t_ent.id,
                                    EntityRelationship.relationship_type == r_type,
                                    EntityRelationship.source_document_id == document.id,
                                    EntityRelationship.page_number == p_num,
                                )
                                .first()
                            )
                            if not existing:
                                llm_rel = EntityRelationship(
                                    workspace_id=workspace_id,
                                    source_entity_id=s_ent.id,
                                    target_entity_id=t_ent.id,
                                    source_document_id=document.id,
                                    relationship_type=r_type,
                                    confidence_score=0.92,
                                    explanation=expl or f"'{s_ent.canonical_name}' {r_type} '{t_ent.canonical_name}'.",
                                    evidence_excerpt=ev,
                                    page_number=p_num,
                                )
                                db.add(llm_rel)
                                created_or_updated_rels.append(llm_rel)
            except Exception as llm_err:
                logger.warning("LLM entity relationship extraction note for doc %s: %s", document.id, llm_err)

        db.commit()
        return created_or_updated_rels


relationship_discovery_service = RelationshipDiscoveryService()
