import json
import logging
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from backend.schemas.document import PageExtractionRecord
from backend.schemas.indexing import ChildChunkRecord, ParentChunkRecord
from backend.services.embedding_service import embedding_service
from config.settings import get_settings

logger = logging.getLogger("ekn.chunking")


class ChunkingService:
    """Implements hierarchical parent-child, heading-aware, and table-aware document chunking.

    Transforms extracted page-level JSONL records into token-bounded child chunks
    for vector indexing and larger contextual parent chunks for downstream evidence retrieval.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._storage_root: Optional[Path] = None
        self._processed_root: Optional[Path] = None
        self.parent_target_tokens = self.settings.chunking_parent_target_tokens
        self.child_target_tokens = self.settings.chunking_child_target_tokens
        self.child_overlap_tokens = self.settings.chunking_child_overlap_tokens
        self._parent_cache: Dict[Tuple[str, str], List[ParentChunkRecord]] = {}
        self._cache_lock = threading.Lock()

    @property
    def storage_root(self) -> Path:
        if self._storage_root is not None:
            return self._storage_root
        return Path(self.settings.storage_local_root).resolve()

    @storage_root.setter
    def storage_root(self, value: Any) -> None:
        self._storage_root = Path(value) if value is not None else None

    @property
    def processed_root(self) -> Path:
        if self._processed_root is not None:
            return self._processed_root
        return self.storage_root / "processed"

    @processed_root.setter
    def processed_root(self, value: Any) -> None:
        self._processed_root = Path(value) if value is not None else None

    def chunk_document(
        self,
        pages: List[PageExtractionRecord],
        document_id: str,
        workspace_id: str,
        original_filename: str,
        file_hash: str,
        document_version: int = 1,
        visibility_scope: str = "company",
        owner_user_id: Optional[str] = None,
    ) -> Tuple[List[ParentChunkRecord], List[ChildChunkRecord]]:
        """Perform complete hierarchical chunking on extracted document pages.

        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        if not pages:
            logger.warning("No pages provided for chunking document %s", document_id)
            return [], []

        # 1. Segment pages into structural sections based on headings
        sections = self._segment_into_sections(pages)

        parent_chunks: List[ParentChunkRecord] = []
        child_chunks: List[ChildChunkRecord] = []

        parent_counter = 1
        child_counter = 1

        for section in sections:
            section_title = section["title"]
            section_page_start = section["page_start"]
            section_page_end = section["page_end"]
            section_items = section["items"]  # list of dict(type='text'|'table', content=..., page=...)

            section_children: List[ChildChunkRecord] = []

            for item in section_items:
                if item["type"] == "text":
                    item_text = item["content"].strip()
                    if not item_text:
                        continue
                    text_children = self._chunk_text(
                        text=item_text,
                        page_number=item["page"],
                        section_title=section_title,
                        document_id=document_id,
                        workspace_id=workspace_id,
                        original_filename=original_filename,
                        file_hash=file_hash,
                        document_version=document_version,
                        start_counter=child_counter,
                    )
                    child_counter += len(text_children)
                    section_children.extend(text_children)

                elif item["type"] == "table":
                    table_dict = item["content"]
                    table_children = self._chunk_table(
                        table=table_dict,
                        page_number=item["page"],
                        section_title=section_title,
                        document_id=document_id,
                        workspace_id=workspace_id,
                        original_filename=original_filename,
                        file_hash=file_hash,
                        document_version=document_version,
                        start_counter=child_counter,
                    )
                    child_counter += len(table_children)
                    section_children.extend(table_children)

            if not section_children:
                continue

            # 2. Group section child chunks into parent chunks (~800 tokens)
            parents, updated_children = self._group_into_parents(
                children=section_children,
                document_id=document_id,
                workspace_id=workspace_id,
                section_title=section_title,
                default_page_start=section_page_start,
                default_page_end=section_page_end,
                start_counter=parent_counter,
            )
            parent_counter += len(parents)
            parent_chunks.extend(parents)
            child_chunks.extend(updated_children)

        # Stamp document ownership scope on all generated parent and child chunks
        for p in parent_chunks:
            p.visibility_scope = visibility_scope
            p.owner_user_id = owner_user_id
        for c in child_chunks:
            c.visibility_scope = visibility_scope
            c.owner_user_id = owner_user_id

        logger.info(
            "Chunked document %s (%s): %d parent chunks, %d child chunks generated",
            original_filename,
            document_id,
            len(parent_chunks),
            len(child_chunks),
        )
        return parent_chunks, child_chunks

    def _segment_into_sections(self, pages: List[PageExtractionRecord]) -> List[Dict[str, Any]]:
        """Segment extracted pages into heading-delimited sections across multiple pages."""
        sections: List[Dict[str, Any]] = []
        current_title = "Document Overview"
        current_page_start = pages[0].page_number if pages else 1
        current_page_end = current_page_start
        current_items: List[Dict[str, Any]] = []

        for page in pages:
            page_num = page.page_number
            text = (getattr(page, "text", None) or getattr(page, "raw_text", "") or "").strip()
            headings = page.headings or []
            tables = page.tables or []

            # If page contains headings, split text by heading locations
            if headings and text:
                # Track heading text and positions
                heading_splits = self._split_page_by_headings(text, headings)
                for part in heading_splits:
                    part_title = part["heading"]
                    part_body = part["body"].strip()

                    if part_title:
                        # Heading boundary detected: finalize current section if it contains items
                        if current_items:
                            sections.append({
                                "title": current_title,
                                "page_start": current_page_start,
                                "page_end": current_page_end,
                                "items": current_items,
                            })
                            current_items = []
                        current_title = part_title
                        current_page_start = page_num

                    if part_body:
                        current_items.append({
                            "type": "text",
                            "content": part_body,
                            "page": page_num,
                        })
                        current_page_end = page_num
            else:
                # No headings on this page: append text to current continuous section
                if text:
                    current_items.append({
                        "type": "text",
                        "content": text,
                        "page": page_num,
                    })
                    current_page_end = page_num

            # Append structured tables on this page
            for tbl in tables:
                tbl_dict = tbl.model_dump() if hasattr(tbl, "model_dump") else dict(tbl)
                current_items.append({
                    "type": "table",
                    "content": tbl_dict,
                    "page": page_num,
                })
                current_page_end = page_num

        # Finalize trailing section
        if current_items:
            sections.append({
                "title": current_title,
                "page_start": current_page_start,
                "page_end": current_page_end,
                "items": current_items,
            })

        return sections

    def _split_page_by_headings(
        self,
        text: str,
        headings: List[Any],
    ) -> List[Dict[str, str]]:
        """Split page text along heading positions while preserving heading titles."""
        parts: List[Dict[str, str]] = []
        if not headings:
            return [{"heading": None, "body": text}]

        # Sort headings by their appearance in the text
        heading_positions = []
        for h in headings:
            if isinstance(h, str):
                h_text = h.strip()
            elif hasattr(h, "text"):
                h_text = str(h.text).strip()
            elif isinstance(h, dict):
                h_text = str(h.get("text", "")).strip()
            else:
                h_text = str(h).strip()

            if not h_text:
                continue
            idx = text.find(h_text)
            if idx != -1:
                heading_positions.append((idx, h_text))

        heading_positions.sort(key=lambda x: x[0])

        if not heading_positions:
            return [{"heading": None, "body": text}]

        # Text before first heading
        first_idx, first_h = heading_positions[0]
        if first_idx > 0:
            pre_text = text[:first_idx].strip()
            if pre_text:
                parts.append({"heading": None, "body": pre_text})

        # Process each heading and its subsequent body
        for i, (idx, h_text) in enumerate(heading_positions):
            start = idx + len(h_text)
            end = heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(text)
            body = text[start:end].strip()
            parts.append({"heading": h_text, "body": body})

        return parts

    def _chunk_text(
        self,
        text: str,
        page_number: int,
        section_title: str,
        document_id: str,
        workspace_id: str,
        original_filename: str,
        file_hash: str,
        document_version: int,
        start_counter: int,
    ) -> List[ChildChunkRecord]:
        """Recursively split text into token-bounded child chunks respecting boundaries."""
        chunks: List[ChildChunkRecord] = []
        counter = start_counter

        # Step 1: Split into paragraphs
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not raw_paragraphs:
            raw_paragraphs = [text.strip()]

        # Step 2: Split oversized paragraphs into sentences
        atomic_units: List[str] = []
        for para in raw_paragraphs:
            para_tokens = embedding_service.count_tokens(para)
            if para_tokens <= self.child_target_tokens:
                atomic_units.append(para)
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    sent_tokens = embedding_service.count_tokens(sent)
                    if sent_tokens <= self.child_target_tokens:
                        atomic_units.append(sent)
                    else:
                        # Step 3: Sub-sentence token fallback
                        sub_units = self._split_by_token_window(sent, self.child_target_tokens, self.child_overlap_tokens)
                        atomic_units.extend(sub_units)

        # Step 3: Accumulate atomic units into chunks up to target size with overlap
        current_tokens = 0
        current_pieces: List[str] = []

        for unit in atomic_units:
            unit_tokens = embedding_service.count_tokens(unit)
            if current_pieces and (current_tokens + unit_tokens > self.child_target_tokens):
                # Emit child chunk
                chunk_text = " ".join(current_pieces).strip()
                chunk_id = f"{document_id}_c{counter:04d}"
                chunks.append(
                    ChildChunkRecord(
                        chunk_id=chunk_id,
                        parent_id="",  # Assigned in _group_into_parents
                        document_id=document_id,
                        workspace_id=workspace_id,
                        original_filename=original_filename,
                        page_start=page_number,
                        page_end=page_number,
                        section_title=section_title,
                        text=chunk_text,
                        chunk_type="text",
                        token_count=current_tokens,
                        file_hash=file_hash,
                        embedding_model=self.settings.embedding_model,
                        document_version=document_version,
                    )
                )
                counter += 1

                # Carry forward overlap pieces from the end of current_pieces
                overlap_pieces = self._extract_overlap(current_pieces, self.child_overlap_tokens)
                current_pieces = list(overlap_pieces)
                current_tokens = sum(embedding_service.count_tokens(p) for p in current_pieces)

            current_pieces.append(unit)
            current_tokens += unit_tokens

        # Emit remaining pieces
        if current_pieces:
            chunk_text = " ".join(current_pieces).strip()
            if chunk_text:
                chunk_id = f"{document_id}_c{counter:04d}"
                chunks.append(
                    ChildChunkRecord(
                        chunk_id=chunk_id,
                        parent_id="",
                        document_id=document_id,
                        workspace_id=workspace_id,
                        original_filename=original_filename,
                        page_start=page_number,
                        page_end=page_number,
                        section_title=section_title,
                        text=chunk_text,
                        chunk_type="text",
                        token_count=embedding_service.count_tokens(chunk_text),
                        file_hash=file_hash,
                        embedding_model=self.settings.embedding_model,
                        document_version=document_version,
                    )
                )

        return chunks

    def _split_by_token_window(self, text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        """Split a long string by exact token window fallback."""
        token_ids = embedding_service.encode_tokens(text)
        if len(token_ids) <= max_tokens:
            return [text]

        pieces: List[str] = []
        step = max(1, max_tokens - overlap_tokens)
        for i in range(0, len(token_ids), step):
            slice_ids = token_ids[i : i + max_tokens]
            decoded = embedding_service.decode_tokens(slice_ids)
            if decoded:
                pieces.append(decoded)
            if i + max_tokens >= len(token_ids):
                break
        return pieces

    def _extract_overlap(self, pieces: List[str], target_overlap_tokens: int) -> List[str]:
        """Select pieces from the end of a list that fit within target overlap tokens."""
        overlap: List[str] = []
        accumulated = 0
        for p in reversed(pieces):
            cnt = embedding_service.count_tokens(p)
            if accumulated + cnt > target_overlap_tokens and overlap:
                break
            overlap.insert(0, p)
            accumulated += cnt
        return overlap

    def _chunk_table(
        self,
        table: Dict[str, Any],
        page_number: int,
        section_title: str,
        document_id: str,
        workspace_id: str,
        original_filename: str,
        file_hash: str,
        document_version: int,
        start_counter: int,
    ) -> List[ChildChunkRecord]:
        """Generate structured, table-aware child chunks preserving headers and row values."""
        headers: List[str] = table.get("headers") or []
        rows: List[List[str]] = table.get("rows") or []

        if not rows and not headers:
            return []

        chunks: List[ChildChunkRecord] = []
        counter = start_counter

        # Format full table
        full_table_text = self._format_markdown_table(headers, rows)
        full_tokens = embedding_service.count_tokens(full_table_text)

        if full_tokens <= self.child_target_tokens:
            # Table fits comfortably in a single child chunk
            chunk_id = f"{document_id}_c{counter:04d}"
            chunks.append(
                ChildChunkRecord(
                    chunk_id=chunk_id,
                    parent_id="",
                    document_id=document_id,
                    workspace_id=workspace_id,
                    original_filename=original_filename,
                    page_start=page_number,
                    page_end=page_number,
                    section_title=section_title,
                    text=full_table_text,
                    chunk_type="table",
                    token_count=full_tokens,
                    file_hash=file_hash,
                    embedding_model=self.settings.embedding_model,
                    document_version=document_version,
                )
            )
            return chunks

        # Large table: split rows into manageable subsets while repeating headers
        row_batch: List[List[str]] = []
        for row in rows:
            candidate_rows = row_batch + [row]
            candidate_text = self._format_markdown_table(headers, candidate_rows)
            candidate_tokens = embedding_service.count_tokens(candidate_text)

            if row_batch and (candidate_tokens > self.child_target_tokens):
                # Emit sub-table chunk
                sub_text = self._format_markdown_table(headers, row_batch)
                chunk_id = f"{document_id}_c{counter:04d}"
                chunks.append(
                    ChildChunkRecord(
                        chunk_id=chunk_id,
                        parent_id="",
                        document_id=document_id,
                        workspace_id=workspace_id,
                        original_filename=original_filename,
                        page_start=page_number,
                        page_end=page_number,
                        section_title=section_title,
                        text=sub_text,
                        chunk_type="table",
                        token_count=embedding_service.count_tokens(sub_text),
                        file_hash=file_hash,
                        embedding_model=self.settings.embedding_model,
                        document_version=document_version,
                    )
                )
                counter += 1
                row_batch = [row]
            else:
                row_batch.append(row)

        if row_batch:
            sub_text = self._format_markdown_table(headers, row_batch)
            chunk_id = f"{document_id}_c{counter:04d}"
            chunks.append(
                ChildChunkRecord(
                    chunk_id=chunk_id,
                    parent_id="",
                    document_id=document_id,
                    workspace_id=workspace_id,
                    original_filename=original_filename,
                    page_start=page_number,
                    page_end=page_number,
                    section_title=section_title,
                    text=sub_text,
                    chunk_type="table",
                    token_count=embedding_service.count_tokens(sub_text),
                    file_hash=file_hash,
                    embedding_model=self.settings.embedding_model,
                    document_version=document_version,
                )
            )

        return chunks

    def _format_markdown_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Render headers and rows into clean Markdown table format."""
        clean_headers = [str(h).strip().replace("\n", " ") if h is not None else "" for h in headers]
        if not clean_headers and rows:
            # If headers are absent, synthesize column count from first row
            col_count = len(rows[0])
            clean_headers = [f"Col_{i + 1}" for i in range(col_count)]

        lines: List[str] = []
        if clean_headers:
            lines.append("| " + " | ".join(clean_headers) + " |")
            lines.append("| " + " | ".join("---" for _ in clean_headers) + " |")

        for r in rows:
            clean_cells = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in r]
            lines.append("| " + " | ".join(clean_cells) + " |")

        return "\n".join(lines)

    def _group_into_parents(
        self,
        children: List[ChildChunkRecord],
        document_id: str,
        workspace_id: str,
        section_title: str,
        default_page_start: int,
        default_page_end: int,
        start_counter: int,
    ) -> Tuple[List[ParentChunkRecord], List[ChildChunkRecord]]:
        """Group child chunks into larger parent contextual units (~800 tokens)."""
        parents: List[ParentChunkRecord] = []
        updated_children: List[ChildChunkRecord] = []
        parent_counter = start_counter

        current_child_group: List[ChildChunkRecord] = []
        accumulated_tokens = 0

        for child in children:
            if current_child_group and (accumulated_tokens + child.token_count > self.parent_target_tokens):
                parent, bound_children = self._create_parent_record(
                    children=current_child_group,
                    parent_counter=parent_counter,
                    document_id=document_id,
                    workspace_id=workspace_id,
                    section_title=section_title,
                )
                parents.append(parent)
                updated_children.extend(bound_children)
                parent_counter += 1

                current_child_group = [child]
                accumulated_tokens = child.token_count
            else:
                current_child_group.append(child)
                accumulated_tokens += child.token_count

        if current_child_group:
            parent, bound_children = self._create_parent_record(
                children=current_child_group,
                parent_counter=parent_counter,
                document_id=document_id,
                workspace_id=workspace_id,
                section_title=section_title,
            )
            parents.append(parent)
            updated_children.extend(bound_children)

        return parents, updated_children

    def _create_parent_record(
        self,
        children: List[ChildChunkRecord],
        parent_counter: int,
        document_id: str,
        workspace_id: str,
        section_title: str,
    ) -> Tuple[ParentChunkRecord, List[ChildChunkRecord]]:
        """Construct a ParentChunkRecord and link its ID to all constituent child chunks."""
        parent_id = f"{document_id}_p{parent_counter:04d}"
        page_start = min(c.page_start for c in children)
        page_end = max(c.page_end for c in children)

        # Determine content type
        types = {c.chunk_type for c in children}
        if len(types) == 1:
            content_type = next(iter(types))
        else:
            content_type = "mixed"

        # Combine text cleanly
        parent_text = "\n\n".join(c.text for c in children)
        child_ids = [c.chunk_id for c in children]

        # Link parent ID into children
        bound_children: List[ChildChunkRecord] = []
        for c in children:
            updated_child = c.model_copy(update={"parent_id": parent_id})
            bound_children.append(updated_child)

        parent = ParentChunkRecord(
            parent_id=parent_id,
            document_id=document_id,
            workspace_id=workspace_id,
            section_title=section_title,
            page_start=page_start,
            page_end=page_end,
            text=parent_text,
            content_type=content_type,
            child_ids=child_ids,
        )

        return parent, bound_children

    def save_parent_chunks(
        self,
        parents: List[ParentChunkRecord],
        workspace_id: str,
        document_id: str,
    ) -> str:
        """Incrementally and atomically persist parent chunks to JSONL.

        Returns:
            Relative path to parent_chunks.jsonl from storage root.
        """
        target_dir = self.processed_root / workspace_id / document_id
        target_dir.mkdir(parents=True, exist_ok=True)

        final_path = target_dir / "parent_chunks.jsonl"
        temp_path = target_dir / "parent_chunks.jsonl.tmp"

        with open(temp_path, "w", encoding="utf-8") as f:
            for p in parents:
                f.write(json.dumps(p.model_dump(), ensure_ascii=False) + "\n")

        # Atomic rename to prevent partial state on failure
        if final_path.exists():
            final_path.unlink()
        temp_path.rename(final_path)

        try:
            relative_path = str(final_path.relative_to(self.storage_root)).replace("\\", "/")
        except ValueError:
            relative_path = f"processed/{workspace_id}/{document_id}/parent_chunks.jsonl"
        # Update in-memory parent cache
        with self._cache_lock:
            self._parent_cache[(workspace_id, document_id)] = list(parents)

        logger.info("Persisted %d parent chunks to: %s", len(parents), relative_path)
        return relative_path

    def invalidate_parent_cache(self, workspace_id: Optional[str] = None, document_id: Optional[str] = None) -> None:
        """Invalidate in-memory parent chunks cache."""
        with self._cache_lock:
            if workspace_id and document_id:
                self._parent_cache.pop((workspace_id, document_id), None)
            elif workspace_id:
                keys_to_remove = [k for k in self._parent_cache if k[0] == workspace_id]
                for k in keys_to_remove:
                    self._parent_cache.pop(k, None)
            else:
                self._parent_cache.clear()

    def load_parent_chunks(
        self,
        workspace_id: str,
        document_id: str,
    ) -> List[ParentChunkRecord]:
        """Read and deserialize parent chunks from cache or JSONL storage."""
        cache_key = (workspace_id, document_id)
        with self._cache_lock:
            if cache_key in self._parent_cache:
                return self._parent_cache[cache_key]

        file_path = self.processed_root / workspace_id / document_id / "parent_chunks.jsonl"

        if not file_path.exists():
            return []

        parents: List[ParentChunkRecord] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parents.append(ParentChunkRecord.model_validate(json.loads(line)))

        with self._cache_lock:
            # Keep bounded cache (max 100 documents)
            if len(self._parent_cache) > 100:
                self._parent_cache.pop(next(iter(self._parent_cache)))
            self._parent_cache[cache_key] = parents

        return parents


# Global singleton instance
chunking_service = ChunkingService()
