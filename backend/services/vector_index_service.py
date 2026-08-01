import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection

from backend.database.connection import get_chroma_client
from backend.schemas.indexing import ChildChunkRecord
from config.settings import get_settings

logger = logging.getLogger("ekn.vector_index")


class VectorIndexService:
    """Manages persistent ChromaDB vector collection and batch upserts for document child chunks.

    Configures HNSW indexing with cosine space and ensures idempotent, batch-level upserts.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.collection_name = self.settings.chroma_collection_name
        self.upsert_batch_size = self.settings.chroma_upsert_batch_size
        self._collection: Optional[Collection] = None

    def get_collection(self) -> Collection:
        """Retrieve or create the document chunks collection with cosine distance."""
        if self._collection is not None:
            return self._collection

        client = get_chroma_client()
        try:
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB collection '%s' ready (space=cosine)", self.collection_name)
            return self._collection
        except Exception as exc:
            logger.error("Failed to get/create ChromaDB collection '%s': %s", self.collection_name, exc)
            raise RuntimeError(f"Vector index initialization failed: {exc}") from exc

    def index_child_chunks(
        self,
        chunks: List[ChildChunkRecord],
        embeddings: List[List[float]],
        document_id: str,
    ) -> int:
        """Upsert child chunks and their embeddings into ChromaDB in batches.

        Cleans up obsolete vector IDs for this document if previous indexed versions existed.

        Args:
            chunks: List of child chunk records.
            embeddings: Corresponding normalized embedding vectors.
            document_id: Document UUID.

        Returns:
            Total number of chunks indexed.
        """
        if not chunks:
            logger.info("No child chunks provided for indexing document %s", document_id)
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})"
            )

        collection = self.get_collection()

        # Step 1: Query existing chunk IDs for this document to detect obsolete vectors
        existing_result = collection.get(where={"document_id": document_id})
        existing_ids = set(existing_result["ids"]) if existing_result and "ids" in existing_result else set()

        new_ids = {c.chunk_id for c in chunks}
        obsolete_ids = list(existing_ids - new_ids)

        # Step 2: Batch upsert new/updated vectors
        batch_size = self.upsert_batch_size
        total_upserted = 0

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            batch_ids = [c.chunk_id for c in batch_chunks]
            batch_docs = [c.text for c in batch_chunks]
            batch_metadatas = [
                {
                    "document_id": c.document_id,
                    "workspace_id": c.workspace_id,
                    "parent_id": c.parent_id,
                    "original_filename": c.original_filename,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section_title": c.section_title,
                    "chunk_type": c.chunk_type,
                    "file_hash": c.file_hash,
                    "embedding_model": c.embedding_model,
                    "document_version": c.document_version,
                    "visibility_scope": getattr(c, "visibility_scope", "company") or "company",
                    "owner_user_id": getattr(c, "owner_user_id", "") or "",
                }
                for c in batch_chunks
            ]

            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_docs,
                metadatas=batch_metadatas,
            )
            total_upserted += len(batch_ids)

        # Step 3: Clean up obsolete IDs from prior versions if any exist
        if obsolete_ids:
            collection.delete(ids=obsolete_ids)
            logger.info(
                "Cleaned up %d obsolete chunk vectors for document %s",
                len(obsolete_ids),
                document_id,
            )

        logger.info(
            "Successfully indexed %d child chunks in ChromaDB for document %s",
            total_upserted,
            document_id,
        )
        return total_upserted

    def delete_document_vectors(self, document_id: str) -> int:
        """Remove all indexed chunk vectors for a document."""
        collection = self.get_collection()
        existing = collection.get(where={"document_id": document_id})
        ids = existing.get("ids", []) if existing else []
        if ids:
            collection.delete(ids=ids)
            logger.info("Deleted %d vectors from ChromaDB for document %s", len(ids), document_id)
        return len(ids)

    def get_document_vector_count(self, document_id: str) -> int:
        """Return the number of vectors stored for a given document."""
        collection = self.get_collection()
        existing = collection.get(where={"document_id": document_id})
        return len(existing["ids"]) if existing and "ids" in existing else 0

    def get_total_vector_count(self) -> int:
        """Return the total number of vectors in the collection."""
        collection = self.get_collection()
        return collection.count()

    def backfill_legacy_metadata(self) -> int:
        """Safely backfill visibility_scope and owner_user_id on legacy ChromaDB vectors.

        Preserves vector IDs, embeddings, and existing metadata in-place.
        Returns the number of vectors updated.
        """
        collection = self.get_collection()
        total_vectors = collection.count()
        if total_vectors == 0:
            return 0

        # Retrieve all items in chunks
        all_data = collection.get(include=["metadatas"])
        if not all_data or not all_data.get("ids"):
            return 0

        ids = all_data["ids"]
        metadatas = all_data["metadatas"] or []

        batch_ids = []
        batch_metas = []
        total_updated = 0

        for vid, vmeta in zip(ids, metadatas):
            meta = dict(vmeta) if vmeta else {}
            needs_update = False
            if "visibility_scope" not in meta or not meta["visibility_scope"]:
                meta["visibility_scope"] = "company"
                needs_update = True
            if "owner_user_id" not in meta:
                meta["owner_user_id"] = ""
                needs_update = True

            if needs_update:
                batch_ids.append(vid)
                batch_metas.append(meta)

            if len(batch_ids) >= 100:
                collection.update(ids=batch_ids, metadatas=batch_metas)
                total_updated += len(batch_ids)
                batch_ids = []
                batch_metas = []

        if batch_ids:
            collection.update(ids=batch_ids, metadatas=batch_metas)
            total_updated += len(batch_ids)

        logger.info("ChromaDB metadata backfill complete: updated %d / %d vectors", total_updated, total_vectors)
        return total_updated

    def query_dense(
        self,
        query_embedding: List[float],
        top_k: int = 30,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform dense vector search against the ChromaDB collection using cosine similarity.

        Args:
            query_embedding: Dense embedding vector for the query.
            top_k: Maximum number of nearest neighbours to retrieve.
            where_filter: Optional ChromaDB metadata filter dictionary.

        Returns:
            List of candidate dictionaries ordered by cosine distance with 1-based 'dense_rank'.
        """
        collection = self.get_collection()
        total_vectors = collection.count()
        if total_vectors == 0:
            return []

        # ChromaDB n_results cannot exceed total count
        n_results = min(top_k, total_vectors)
        if n_results <= 0:
            return []

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where_filter:
            kwargs["where"] = where_filter

        try:
            results = collection.query(**kwargs)
        except Exception as exc:
            logger.error("ChromaDB query failed with filter %s: %s", where_filter, exc)
            return []

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else [""] * len(ids)
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        candidates = []
        for rank, (chunk_id, doc_text, meta, dist) in enumerate(zip(ids, docs, metas, distances), start=1):
            candidates.append({
                "chunk_id": chunk_id,
                "parent_id": meta.get("parent_id", ""),
                "document_id": meta.get("document_id", ""),
                "workspace_id": meta.get("workspace_id", ""),
                "original_filename": meta.get("original_filename", ""),
                "page_start": meta.get("page_start", 1),
                "page_end": meta.get("page_end", 1),
                "section_title": meta.get("section_title", ""),
                "chunk_type": meta.get("chunk_type", "text"),
                "text": doc_text,
                "visibility_scope": meta.get("visibility_scope", "company"),
                "owner_user_id": meta.get("owner_user_id") or None,
                "dense_distance": float(dist),
                "dense_rank": rank,
            })

        return candidates


# Global singleton instance
vector_index_service = VectorIndexService()
