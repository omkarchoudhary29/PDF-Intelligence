import json
import logging
import math
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from backend.schemas.indexing import ChildChunkRecord
from config.settings import get_settings

logger = logging.getLogger("ekn.bm25")


class LuceneBM25Okapi(BM25Okapi):
    """BM25Okapi variant with non-negative Lucene IDF: ln(1.0 + (N - n + 0.5) / (n + 0.5)).

    Guarantees strictly positive term weights regardless of corpus size, eliminating zero-IDF
    anomalies on small or early-stage workspace document collections.
    """

    def _calc_idf(self, nd: Dict[str, int]) -> None:
        for word, freq in nd.items():
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


class BM25Service:
    """Workspace-scoped BM25Okapi keyword retrieval service.

    Maintains a persistent JSONL corpus per workspace (no unsafe pickling),
    tokenizes text using domain-aware preservation of technical identifiers,
    and caches in-memory BM25 index models for sub-millisecond keyword retrieval.
    """

    # Domain-aware regex tokenizer: preserves hyphenated identifiers (AX-450),
    # underscored names (PART_33), dotted numbers (v1.2.3), and technical terms.
    TOKEN_REGEX = re.compile(r"[a-zA-Z0-9]+(?:[-_/.][a-zA-Z0-9]+)*")

    def __init__(self) -> None:
        self.settings = get_settings()
        self._storage_root: Optional[Path] = None
        self._processed_root: Optional[Path] = None
        self._cache: Dict[str, Tuple[BM25Okapi, List[Dict[str, Any]]]] = {}
        self._lock = threading.Lock()

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

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Domain-aware tokenizer preserving technical codes, model numbers, and identifiers."""
        if not text:
            return []
        return cls.TOKEN_REGEX.findall(text.lower())

    def _get_corpus_path(self, workspace_id: str) -> Path:
        """Resolve workspace-isolated BM25 corpus file path."""
        workspace_dir = self.processed_root / workspace_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return workspace_dir / "bm25_corpus.jsonl"

    def invalidate_cache(self, workspace_id: Optional[str] = None) -> None:
        """Clear cached BM25 index for a specific workspace or all workspaces."""
        with self._lock:
            if workspace_id:
                self._cache.pop(workspace_id, None)
                logger.debug("Cleared BM25 cache for workspace %s", workspace_id)
            else:
                self._cache.clear()
                logger.debug("Cleared all BM25 workspace caches")

    def load_or_build_index(
        self,
        workspace_id: str,
    ) -> Optional[Tuple[BM25Okapi, List[Dict[str, Any]]]]:
        """Load corpus and build BM25Okapi model for workspace, cached in memory."""
        with self._lock:
            if workspace_id in self._cache:
                return self._cache[workspace_id]

            corpus_file = self._get_corpus_path(workspace_id)
            if not corpus_file.exists():
                logger.debug("No BM25 corpus found for workspace %s at %s", workspace_id, corpus_file)
                return None

            records: List[Dict[str, Any]] = []
            tokenized_corpus: List[List[str]] = []

            try:
                with open(corpus_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        text = record.get("text", "")
                        tokens = self.tokenize(text)
                        records.append(record)
                        tokenized_corpus.append(tokens)
            except Exception as exc:
                logger.error("Failed to read BM25 corpus from %s: %s", corpus_file, exc)
                return None

            if not records:
                logger.debug("BM25 corpus at %s is empty", corpus_file)
                return None

            bm25 = LuceneBM25Okapi(tokenized_corpus)
            self._cache[workspace_id] = (bm25, records)
            logger.info(
                "Built and cached in-memory BM25 index for workspace %s (%d chunks)",
                workspace_id,
                len(records),
            )
            return self._cache[workspace_id]

    def search(
        self,
        workspace_id: str,
        query: str,
        top_k: int = 30,
        document_ids: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None,
        scope: Optional[str] = None,
        owner_user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform sparse keyword retrieval using BM25Okapi within a workspace.

        Args:
            workspace_id: Workspace UUID.
            query: User search query or keywords.
            top_k: Maximum number of sparse results to return.
            document_ids: Optional document ID filter.
            chunk_types: Optional chunk type filter ('text', 'table').

        Returns:
            Ranked list of chunk dictionaries with 'bm25_score' and 'bm25_rank' (1-based).
        """
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        index_entry = self.load_or_build_index(workspace_id)
        if not index_entry:
            return []

        bm25, records = index_entry
        scores = bm25.get_scores(query_tokens)

        doc_filter = set(document_ids) if document_ids else None
        type_filter = set(chunk_types) if chunk_types else None

        candidates = []
        for idx, score in enumerate(scores):
            # Only consider positive scores (BM25 returns > 0 when at least one token matches)
            if score <= 1e-6:
                continue

            rec = records[idx]

            # Apply document_ids filter
            if doc_filter is not None and rec.get("document_id") not in doc_filter:
                continue

            # Apply chunk_types filter
            if type_filter is not None and rec.get("chunk_type") not in type_filter:
                continue

            # Apply scope filtering
            rec_scope = rec.get("visibility_scope", "company")
            rec_owner = rec.get("owner_user_id")

            if scope == "company":
                if rec_scope != "company":
                    continue
            elif scope == "personal":
                if rec_scope != "personal" or rec_owner != owner_user_id:
                    continue
            elif scope == "combined":
                if rec_scope != "company" and not (rec_scope == "personal" and rec_owner == owner_user_id):
                    continue

            candidates.append({
                "chunk_id": rec["chunk_id"],
                "parent_id": rec.get("parent_id", ""),
                "document_id": rec.get("document_id", ""),
                "workspace_id": rec.get("workspace_id", workspace_id),
                "original_filename": rec.get("original_filename", ""),
                "page_start": rec.get("page_start", 1),
                "page_end": rec.get("page_end", 1),
                "section_title": rec.get("section_title", ""),
                "chunk_type": rec.get("chunk_type", "text"),
                "text": rec.get("text", ""),
                "visibility_scope": rec_scope,
                "owner_user_id": rec_owner,
                "bm25_score": float(score),
            })

        # Sort descending by BM25 score
        candidates.sort(key=lambda x: x["bm25_score"], reverse=True)
        top_candidates = candidates[:top_k]

        # Assign 1-based ranks
        for rank, item in enumerate(top_candidates, start=1):
            item["bm25_rank"] = rank

        return top_candidates

    def add_or_update_document_chunks(
        self,
        workspace_id: str,
        document_id: str,
        chunks: List[ChildChunkRecord],
    ) -> int:
        """Incrementally update BM25 JSONL corpus for a workspace when a document is indexed.

        Replaces any existing entries for the specified document_id to guarantee idempotence,
        then writes atomically via a temporary file and invalidates the workspace cache.
        """
        corpus_file = self._get_corpus_path(workspace_id)
        existing_records: List[Dict[str, Any]] = []

        if corpus_file.exists():
            try:
                with open(corpus_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            # Keep records belonging to other documents
                            if record.get("document_id") != document_id:
                                existing_records.append(record)
            except Exception as exc:
                logger.warning("Error reading existing corpus for workspace %s: %s", workspace_id, exc)

        # Append new chunk records
        new_records = [
            {
                "chunk_id": c.chunk_id,
                "parent_id": c.parent_id,
                "document_id": c.document_id,
                "workspace_id": c.workspace_id,
                "original_filename": c.original_filename,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "section_title": c.section_title,
                "chunk_type": c.chunk_type,
                "text": c.text,
                "visibility_scope": getattr(c, "visibility_scope", "company") or "company",
                "owner_user_id": getattr(c, "owner_user_id", None),
            }
            for c in chunks
        ]

        all_records = existing_records + new_records

        # Atomic write via temp file
        temp_file = corpus_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                for r in all_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(temp_file, corpus_file)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

        self.invalidate_cache(workspace_id)
        logger.info(
            "Updated BM25 corpus for workspace %s (document %s: added %d chunks, total %d chunks)",
            workspace_id,
            document_id,
            len(new_records),
            len(all_records),
        )
        return len(new_records)

    def remove_document_chunks(
        self,
        workspace_id: str,
        document_id: str,
    ) -> int:
        """Purge all chunks belonging to a deleted document from the workspace BM25 corpus."""
        corpus_file = self._get_corpus_path(workspace_id)
        if not corpus_file.exists():
            return 0

        kept_records: List[Dict[str, Any]] = []
        removed_count = 0

        try:
            with open(corpus_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        if record.get("document_id") == document_id:
                            removed_count += 1
                        else:
                            kept_records.append(record)
        except Exception as exc:
            logger.error("Error reading BM25 corpus during deletion: %s", exc)
            return 0

        if removed_count == 0:
            return 0

        temp_file = corpus_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                for r in kept_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(temp_file, corpus_file)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

        self.invalidate_cache(workspace_id)
        logger.info(
            "Removed %d chunks from BM25 corpus for document %s in workspace %s",
            removed_count,
            document_id,
            workspace_id,
        )
        return removed_count


bm25_service = BM25Service()
