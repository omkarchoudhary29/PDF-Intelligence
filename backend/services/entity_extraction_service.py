"""Entity Extraction Service for Enterprise Knowledge Navigator.

Extracts named entities (Machines, Equipment, Policies, Processes, Standards, etc.)
from document pages using a combination of deterministic technical regex matchers
and LLM-based structured extraction via the configured LLM service.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

from backend.database.models import Document, DocumentEntityAssociation, Entity
from backend.services.entity_normalization_service import entity_normalization_service
from backend.services.llm_service import LLMError, LLMNotConfiguredError, llm_service

logger = logging.getLogger("ekn.graph.entity_extraction")

# Deterministic patterns for technical, systems, components, standards, and engineering entities
PATTERNS = [
    # Technical Equipment & Model codes: AX-450, GT-200, RTU-12A, X-500
    (re.compile(r"\b([A-Z]{1,5}-\d{2,5}[A-Za-z]?)\b"), "Equipment"),
    # Industrial & Aerospace standards: ISO-9001, AS9100, IEEE-802, OSHA-1910, FAA Part 25, MIL-STD-810
    (re.compile(r"\b(ISO[-\s]?\d{3,5}(?:-\d+)?|AS9100[A-Z]?|IEEE[-\s]?\d+|OSHA[-\s]?\d+|FAA\s+Part\s+\d+|MIL-STD-\d+[A-Z]?)\b", re.IGNORECASE), "Standard"),
    # Industrial Machines / Units: Turbine Unit 4, Turbine Model X-500, Boiler 2, Compressor C-1
    (re.compile(r"\b((?:Gas\s+)?Turbine\s+(?:(?:Unit|Model)\s+)?[A-Z0-9-]+|Boiler\s+[A-Z0-9-]+|Compressor\s+[A-Z0-9-]+)\b", re.IGNORECASE), "Machine"),
    # Engineering Systems
    (re.compile(r"\b((?:Hydraulic|Propulsion|Avionics|Electrical|Fuel\s+Management|Cooling|Lubrication|Pneumatic|Braking|Telemetry)\s+System)\b", re.IGNORECASE), "System"),
    # Critical Components
    (re.compile(r"\b((?:Hydraulic|Fuel|Oil)\s+Pump|Pressure\s+Relief\s+Valve|Fuel\s+Injector|Turbine\s+Rotor|Combustion\s+Chamber|Actuator|Heat\s+Exchanger)\b", re.IGNORECASE), "Component"),
    # Engineering Materials
    (re.compile(r"\b(Inconel\s+\d+|Titanium\s+Alloy|Carbon\s+Composite|Stainless\s+Steel\s+\d+[A-Z]?|Silicon\s+Carbide)\b", re.IGNORECASE), "Material"),
    # Maintenance Procedures & Inspections
    (re.compile(r"\b((?:Borescope|Pre-Flight|Post-Flight|Scheduled|NDT|Ultrasonic)\s+Inspection|(?:Turbine|Engine)\s+Overhaul\s+Procedure)\b", re.IGNORECASE), "Maintenance procedure"),
    # AI / LLM Models & Architectures
    (re.compile(r"\b(Transformer\s+Architecture|Large\s+Language\s+Models?|Vector\s+Databases?|ChromaDB|PyTorch|TensorFlow|Hugging\s+Face|BERT|GPT-\d+[A-Za-z]?|Gemini|Claude)\b", re.IGNORECASE), "Technical concept"),
    # RAG & Information Retrieval Concepts
    (re.compile(r"\b(Retrieval-Augmented\s+Generation|RAG|Dense\s+Embeddings?|Cross-Encoder|Bi-Encoder|Reciprocal\s+Rank\s+Fusion|RRF|BM25|HNSW|Cosine\s+Similarity)\b", re.IGNORECASE), "Technical concept"),
    # Prompting & Model Techniques
    (re.compile(r"\b(Chain-of-Thought|Few-Shot\s+Prompting|Zero-Shot\s+Prompting|Fine-Tuning|Quantization|LoRA|RLHF|Tokenization|Byte-Pair\s+Encoding|BPE|WordPiece)\b", re.IGNORECASE), "Technical concept"),
    # Software Engineering & Version Control
    (re.compile(r"\b(Git\s+Version\s+Control|GitHub\s+Repository|Pull\s+Requests?|GitLab|Merge\s+Conflicts?)\b", re.IGNORECASE), "Technical concept"),
    # Document filenames cited in text: Manual_AX450.pdf
    (re.compile(r"\b([A-Za-z0-9_-]+\.pdf)\b", re.IGNORECASE), "Other"),
]


class RawExtractedEntity:
    def __init__(
        self,
        name: str,
        entity_type: str,
        page_number: int,
        excerpt: str,
        section_title: Optional[str] = None,
        description: Optional[str] = None,
        extraction_method: str = "regex",
        aliases: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
    ):
        self.name = name
        self.entity_type = entity_type
        self.page_number = page_number
        self.excerpt = excerpt
        self.section_title = section_title
        self.description = description
        self.extraction_method = extraction_method
        self.aliases = aliases or []
        self.attributes = attributes or {}
        self.confidence = confidence


class EntityExtractionService:
    """Extracts, grounds, and persists document entities and page-level associations."""

    def extract_deterministic_entities(self, page_record: Any) -> List[RawExtractedEntity]:
        """Extract entities using fast, deterministic regex patterns across page text."""
        if isinstance(page_record, str):
            raw_text = page_record
            page_num = 1
            headings = []
        elif isinstance(page_record, dict):
            raw_text = page_record.get("text") or page_record.get("raw_text", "")
            page_num = page_record.get("page_number", 1)
            headings = page_record.get("headings", [])
        else:
            raw_text = getattr(page_record, "text", "") or getattr(page_record, "raw_text", "") or ""
            page_num = getattr(page_record, "page_number", 1)
            headings = getattr(page_record, "headings", [])

        primary_section = None
        if headings:
            h0 = headings[0]
            if hasattr(h0, "text"):
                primary_section = h0.text
            elif isinstance(h0, dict) and "text" in h0:
                primary_section = h0["text"]
            elif isinstance(h0, str):
                primary_section = h0

        results: List[RawExtractedEntity] = []
        seen_names = set()

        for pattern, default_type in PATTERNS:
            for match in pattern.finditer(raw_text):
                matched_str = match.group(1).strip()
                norm = entity_normalization_service.normalize_name(matched_str)
                if not norm or norm in seen_names or len(norm) < 3:
                    continue
                seen_names.add(norm)

                # Extract surrounding sentence or 120-char window as excerpt
                start_idx = max(0, match.start() - 40)
                end_idx = min(len(raw_text), match.end() + 60)
                excerpt = raw_text[start_idx:end_idx].strip()

                results.append(
                    RawExtractedEntity(
                        name=matched_str,
                        entity_type=default_type,
                        page_number=page_num,
                        excerpt=excerpt,
                        section_title=primary_section,
                        description=f"{default_type} identified in document text.",
                        extraction_method="regex",
                    )
                )

        return results

    def extract_llm_entities(
        self,
        text_batch: str,
        page_number: int,
        section_title: Optional[str] = None,
    ) -> List[RawExtractedEntity]:
        """Extract entities using structured LLM inference when configured."""
        if not llm_service.is_configured:
            return []

        prompt = (
            "You are a technical document knowledge graph engineer. "
            "Extract distinct enterprise entities from the following document passage. "
            "Return valid JSON only, without any markdown formatting or commentary.\n\n"
            "Allowed entity types:\n"
            "- Equipment\n- Machine\n- Component\n- System\n- Process\n- Material\n"
            "- Product\n- Project\n- Organization\n- Department\n- Policy\n"
            "- Regulation\n- Standard\n- Person\n- Location\n- Technical concept\n"
            "- Safety hazard\n- Maintenance procedure\n- Measurement\n- Other\n\n"
            "Format:\n"
            "{\n"
            '  "entities": [\n'
            '    {\n'
            '      "name": "...",\n'
            '      "type": "...",\n'
            '      "description": "...",\n'
            '      "aliases": ["..."],\n'
            '      "excerpt": "exact text passage supporting entity"\n'
            '    }\n'
            '  ]\n'
            "}\n\n"
            f"Passage (Page {page_number}):\n"
            f"{text_batch[:3000]}\n"
        )

        try:
            model = llm_service.get_llm()
            resp = model.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)

            # Strip possible markdown code fences
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            entities_list = parsed.get("entities", [])

            results: List[RawExtractedEntity] = []
            for item in entities_list:
                name = item.get("name", "").strip()
                if not name or len(name) < 2:
                    continue
                raw_type = item.get("type", "Other")
                validated_type = entity_normalization_service.validate_and_normalize_type(raw_type)
                excerpt = item.get("excerpt") or text_batch[:150]
                desc = item.get("description")
                aliases = [a.strip() for a in item.get("aliases", []) if isinstance(a, str) and a.strip()]

                results.append(
                    RawExtractedEntity(
                        name=name,
                        entity_type=validated_type,
                        page_number=page_number,
                        excerpt=excerpt[:400],
                        section_title=section_title,
                        description=desc[:300] if desc else None,
                        extraction_method="llm",
                        aliases=aliases,
                    )
                )

            return results

        except (LLMNotConfiguredError, LLMError) as err:
            logger.info("LLM extraction unavailable or failed (%s); falling back to regex.", err)
            return []
        except Exception as exc:
            logger.warning("LLM entity extraction output parsing failed: %s", exc)
            return []

    def extract_and_store_document_entities(
        self,
        db: Session,
        workspace_id: str,
        document: Document,
        pages_records: List[Dict[str, Any]],
        use_llm: bool = True,
    ) -> List[DocumentEntityAssociation]:
        """Extract entities across all document pages and persist normalized entities and associations."""
        if not pages_records:
            return []

        all_extracted: List[RawExtractedEntity] = []

        # 1. Deterministic Extraction per page
        for page in pages_records:
            det_entities = self.extract_deterministic_entities(page)
            all_extracted.extend(det_entities)

        # 2. LLM Extraction (batching pages if configured and requested)
        if use_llm and llm_service.is_configured:
            for page in pages_records[:10]:  # Cap to first 10 pages per doc for efficiency
                if isinstance(page, dict):
                    text = (page.get("text") or page.get("raw_text") or "").strip()
                    headings = page.get("headings", [])
                    p_num = page.get("page_number", 1)
                else:
                    text = (getattr(page, "text", "") or getattr(page, "raw_text", "") or "").strip()
                    headings = getattr(page, "headings", [])
                    p_num = getattr(page, "page_number", 1)

                if len(text) > 80:
                    sec = None
                    if headings:
                        h0 = headings[0]
                        sec = h0.text if hasattr(h0, "text") else (h0["text"] if isinstance(h0, dict) and "text" in h0 else (h0 if isinstance(h0, str) else None))
                    llm_ents = self.extract_llm_entities(text, p_num, sec)
                    all_extracted.extend(llm_ents)

        # 3. Deduplicate and Persist to Database
        associations: List[DocumentEntityAssociation] = []
        created_associations = 0

        for item in all_extracted:
            norm_name = entity_normalization_service.normalize_name(item.name)
            canon_name = entity_normalization_service.clean_canonical_name(item.name)
            entity_type = entity_normalization_service.validate_and_normalize_type(item.entity_type)

            if not norm_name:
                continue

            # Look up or create canonical Entity record
            entity = (
                db.query(Entity)
                .filter(
                    Entity.workspace_id == workspace_id,
                    Entity.normalized_name == norm_name,
                    Entity.entity_type == entity_type,
                )
                .first()
            )

            if not entity:
                entity = Entity(
                    workspace_id=workspace_id,
                    canonical_name=canon_name,
                    normalized_name=norm_name,
                    entity_type=entity_type,
                    description=item.description,
                    aliases_json=item.aliases if hasattr(item, "aliases") else [],
                    attributes_json=item.attributes if hasattr(item, "attributes") else {},
                )
                db.add(entity)
                db.flush()
            else:
                # Merge aliases if any
                if hasattr(item, "aliases") and item.aliases:
                    cur_aliases = list(entity.aliases_json or [])
                    for al in item.aliases:
                        if al and al not in cur_aliases:
                            cur_aliases.append(al)
                    entity.aliases_json = cur_aliases
                if not entity.description and item.description:
                    entity.description = item.description

            # Check if association already exists on this page
            existing_assoc = (
                db.query(DocumentEntityAssociation)
                .filter(
                    DocumentEntityAssociation.workspace_id == workspace_id,
                    DocumentEntityAssociation.document_id == document.id,
                    DocumentEntityAssociation.entity_id == entity.id,
                    DocumentEntityAssociation.page_start == item.page_number,
                )
                .first()
            )

            if existing_assoc:
                existing_assoc.mention_count += 1
                associations.append(existing_assoc)
            else:
                assoc = DocumentEntityAssociation(
                    workspace_id=workspace_id,
                    document_id=document.id,
                    entity_id=entity.id,
                    section_title=item.section_title,
                    page_start=item.page_number,
                    page_end=item.page_number,
                    excerpt=item.excerpt,
                    mention_count=1,
                    extraction_method=item.extraction_method,
                )
                db.add(assoc)
                associations.append(assoc)
                created_associations += 1

        db.commit()
        logger.info(
            "Extracted and stored %d associations for document %s in workspace %s",
            len(associations),
            document.id,
            workspace_id,
        )
        return associations


entity_extraction_service = EntityExtractionService()
