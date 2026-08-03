import logging
from typing import Any, Dict, List, Optional
from sqlalchemy import and_, or_

from backend.schemas.search import SearchRequest, SearchResponse, SearchResultItem
from backend.services.bm25_service import bm25_service
from backend.services.chunking_service import chunking_service
from backend.services.embedding_service import embedding_service
from backend.services.reranking_service import reranking_service
from backend.services.scope_resolver import scope_resolver
from backend.services.vector_index_service import vector_index_service
from config.settings import get_settings

logger = logging.getLogger("ekn.retrieval")


class RetrievalService:
    """Master hybrid document retrieval orchestrator.

    Integrates:
    1. Dense semantic search via ChromaDB HNSW cosine index.
    2. Sparse keyword search via domain-aware BM25Okapi.
    3. Metadata filtering (workspace isolation, document_ids, chunk_types).
    4. Reciprocal Rank Fusion (RRF) combining dense and sparse results.
    5. Cross-Encoder reranking using ms-marco-MiniLM-L-6-v2.
    6. Contextual parent recovery with exact page-level provenance.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_service = embedding_service
        self.vector_index_service = vector_index_service
        self.bm25_service = bm25_service
        self.reranking_service = reranking_service
        self.chunking_service = chunking_service

    def _build_dense_filter(
        self,
        workspace_id: str,
        document_ids: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None,
        scope: str = "company",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Construct ChromaDB metadata filter dictionary supporting multi-attribute constraints."""
        return scope_resolver.build_chroma_filter(
            workspace_id=workspace_id,
            user_id=user_id or "",
            scope=scope,
            document_ids=document_ids,
            chunk_types=chunk_types,
        )

    def resolve_version_scoped_document_ids(
        self,
        workspace_id: str,
        requested_document_ids: Optional[List[str]] = None,
        include_historical: bool = False,
        family_id: Optional[str] = None,
        version_status: Optional[str] = None,
        scope: str = "company",
        user_id: Optional[str] = None,
    ) -> Optional[List[str]]:
        """Resolve document IDs to search based on ownership and version scoping parameters."""
        try:
            from backend.database.connection import get_db_session
            from backend.database.models import Document

            with get_db_session() as db:
                base_query = db.query(Document.id).filter(Document.workspace_id == workspace_id)
                total_docs = base_query.count()
                if total_docs == 0:
                    return requested_document_ids

                # Apply ownership scope
                if scope == "company":
                    query = base_query.filter(Document.visibility_scope == "company")
                elif scope == "personal":
                    query = base_query.filter(
                        Document.visibility_scope == "personal",
                        Document.owner_user_id == user_id,
                    )
                elif scope == "combined":
                    query = base_query.filter(
                        or_(
                            Document.visibility_scope == "company",
                            and_(
                                Document.visibility_scope == "personal",
                                Document.owner_user_id == user_id,
                            ),
                        )
                    )
                else:
                    query = base_query.filter(Document.visibility_scope == "company")

                # If specific document IDs requested, validate against authorized scope
                if requested_document_ids is not None:
                    query = query.filter(Document.id.in_(requested_document_ids))
                    return [row[0] for row in query.all()]

                if family_id:
                    query = query.filter(Document.family_id == family_id)

                if version_status:
                    query = query.filter(Document.version_status == version_status)
                elif not include_historical:
                    query = query.filter(
                        (Document.version_status == "current") | (Document.version_status.is_(None))
                    )
                elif family_id is None:
                    return None

                doc_ids = [row[0] for row in query.all()]
                return doc_ids
        except Exception as exc:
            logger.warning("Failed to resolve version-scoped document IDs: %s. Using default scope.", exc)
            return requested_document_ids

    def search(
        self,
        workspace_id: str,
        request: SearchRequest,
    ) -> SearchResponse:
        """Execute hybrid search pipeline: Dense + BM25 -> RRF -> Cross-Encoder -> Parent Recovery."""
        query = request.query.strip()
        top_k = request.top_k
        norm_scope = scope_resolver.normalize_scope(request.scope)
        document_ids = self.resolve_version_scoped_document_ids(
            workspace_id=workspace_id,
            requested_document_ids=request.document_ids,
            include_historical=request.include_historical,
            family_id=request.family_id,
            version_status=request.version_status,
            scope=norm_scope,
            user_id=request.user_id,
        )
        chunk_types = request.chunk_types

        if document_ids is not None and len(document_ids) == 0:
            logger.info("Scope/version filter yielded 0 matching document IDs in workspace %s", workspace_id)
            return SearchResponse(
                query=query,
                workspace_id=workspace_id,
                total_results=0,
                results=[],
            )

        logger.info(
            "Executing hybrid search in workspace '%s' (query='%s', scope=%s, user=%s, top_k=%d, doc_filter=%s, type_filter=%s)",
            workspace_id,
            query,
            norm_scope,
            request.user_id,
            top_k,
            document_ids,
            chunk_types,
        )

        # -------------------------------------------------------------
        # 1. Dense Semantic Retrieval (ChromaDB)
        # -------------------------------------------------------------
        dense_candidates: List[Dict[str, Any]] = []
        try:
            query_vector = self.embedding_service.embed_query(query)
            where_filter = self._build_dense_filter(
                workspace_id=workspace_id,
                document_ids=document_ids,
                chunk_types=chunk_types,
                scope=norm_scope,
                user_id=request.user_id,
            )
            dense_candidates = self.vector_index_service.query_dense(
                query_embedding=query_vector,
                top_k=self.settings.retrieval_dense_top_k,
                where_filter=where_filter,
            )
            logger.debug(
                "Dense retrieval returned %d candidates for workspace %s",
                len(dense_candidates),
                workspace_id,
            )
        except Exception as exc:
            logger.error("Dense retrieval failed: %s", exc, exc_info=True)

        # -------------------------------------------------------------
        # 2. Sparse Keyword Retrieval (BM25)
        # -------------------------------------------------------------
        bm25_candidates: List[Dict[str, Any]] = []
        try:
            bm25_candidates = self.bm25_service.search(
                workspace_id=workspace_id,
                query=query,
                top_k=self.settings.retrieval_bm25_top_k,
                document_ids=document_ids,
                chunk_types=chunk_types,
                scope=norm_scope,
                owner_user_id=request.user_id,
            )
            logger.debug(
                "BM25 retrieval returned %d candidates for workspace %s",
                len(bm25_candidates),
                workspace_id,
            )
        except Exception as exc:
            logger.error("BM25 retrieval failed: %s", exc, exc_info=True)

        # -------------------------------------------------------------
        # 3. Reciprocal Rank Fusion (RRF)
        # -------------------------------------------------------------
        k = self.settings.retrieval_rrf_k
        merged: Dict[str, Dict[str, Any]] = {}

        # Add dense candidates to fusion map
        for item in dense_candidates:
            cid = item["chunk_id"]
            d_rank = item["dense_rank"]
            rrf_score = 1.0 / (k + d_rank)
            merged[cid] = {
                **item,
                "rrf_score": rrf_score,
                "dense_rank": d_rank,
                "dense_distance": item.get("dense_distance"),
                "bm25_rank": None,
                "bm25_score": None,
            }

        # Merge BM25 candidates
        for item in bm25_candidates:
            cid = item["chunk_id"]
            b_rank = item["bm25_rank"]
            rrf_score = 1.0 / (k + b_rank)
            if cid in merged:
                merged[cid]["rrf_score"] += rrf_score
                merged[cid]["bm25_rank"] = b_rank
                merged[cid]["bm25_score"] = item.get("bm25_score")
            else:
                merged[cid] = {
                    **item,
                    "rrf_score": rrf_score,
                    "dense_rank": None,
                    "dense_distance": None,
                    "bm25_rank": b_rank,
                    "bm25_score": item.get("bm25_score"),
                }

        if not merged:
            logger.info("Hybrid retrieval found 0 matching candidates in workspace %s", workspace_id)
            return SearchResponse(
                query=query,
                workspace_id=workspace_id,
                total_results=0,
                results=[],
            )

        # Rank all merged candidates by RRF score descending
        fused_candidates = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)
        for rrf_rank, item in enumerate(fused_candidates, start=1):
            item["rrf_rank"] = rrf_rank

        # Limit to configured RRF rerank pool size
        candidates_to_rerank = fused_candidates[: self.settings.retrieval_rerank_candidates]

        # -------------------------------------------------------------
        # 4. Cross-Encoder Reranking
        # -------------------------------------------------------------
        try:
            reranked_candidates = self.reranking_service.rerank(
                query=query,
                candidates=candidates_to_rerank,
                top_k=top_k,
            )
        except Exception as exc:
            logger.error("Cross-Encoder reranking failed: %s. Falling back to RRF ranking.", exc)
            reranked_candidates = candidates_to_rerank[:top_k]
            for idx, c in enumerate(reranked_candidates, start=1):
                c["final_rank"] = idx
                c["cross_encoder_score"] = None

        # -------------------------------------------------------------
        # 5. Parent Context Recovery
        # -------------------------------------------------------------
        if request.include_parent:
            doc_parent_cache: Dict[str, Dict[str, str]] = {}
            for item in reranked_candidates:
                doc_id = item.get("document_id")
                parent_id = item.get("parent_id")
                if doc_id and parent_id:
                    if doc_id not in doc_parent_cache:
                        try:
                            parents = self.chunking_service.load_parent_chunks(
                                workspace_id=workspace_id,
                                document_id=doc_id,
                            )
                            doc_parent_cache[doc_id] = {p.parent_id: p.text for p in parents}
                        except Exception as exc:
                            logger.warning(
                                "Failed to load parent chunks for document %s: %s",
                                doc_id,
                                exc,
                            )
                            doc_parent_cache[doc_id] = {}
                    item["parent_text"] = doc_parent_cache[doc_id].get(parent_id)
                else:
                    item["parent_text"] = None
        else:
            for item in reranked_candidates:
                item["parent_text"] = None

        # -------------------------------------------------------------
        # 6. Enrich with Document Version Metadata & Format Response
        # -------------------------------------------------------------
        doc_ids_to_lookup = {item.get("document_id") for item in reranked_candidates if item.get("document_id")}
        doc_version_map: Dict[str, Dict[str, Any]] = {}
        if doc_ids_to_lookup:
            try:
                from backend.database.connection import get_db_session
                from backend.database.models import Document
                with get_db_session() as db:
                    docs = db.query(Document).filter(Document.id.in_(doc_ids_to_lookup)).all()
                    for d in docs:
                        doc_version_map[d.id] = {
                            "family_id": d.family_id,
                            "family_title": d.family.title if d.family else None,
                            "version_number": d.version_number or 1,
                            "version_status": d.version_status or "current",
                            "is_current": (d.version_status == "current"),
                        }
            except Exception as exc:
                logger.debug("Failed to lookup document version metadata for search results: %s", exc)

        results = [
            SearchResultItem(
                chunk_id=item["chunk_id"],
                parent_id=item.get("parent_id", ""),
                document_id=item.get("document_id", ""),
                workspace_id=item.get("workspace_id", workspace_id),
                original_filename=item.get("original_filename", ""),
                page_start=item.get("page_start", 1),
                page_end=item.get("page_end", 1),
                section_title=item.get("section_title", ""),
                chunk_type=item.get("chunk_type", "text"),
                text=item.get("text", ""),
                parent_text=item.get("parent_text"),
                visibility_scope=item.get("visibility_scope", "company"),
                owner_user_id=item.get("owner_user_id"),
                family_id=doc_version_map.get(item.get("document_id", ""), {}).get("family_id"),
                family_title=doc_version_map.get(item.get("document_id", ""), {}).get("family_title"),
                version_number=doc_version_map.get(item.get("document_id", ""), {}).get("version_number", 1),
                version_status=doc_version_map.get(item.get("document_id", ""), {}).get("version_status", "current"),
                is_current=doc_version_map.get(item.get("document_id", ""), {}).get("is_current", True),
                dense_rank=item.get("dense_rank"),
                dense_distance=item.get("dense_distance"),
                bm25_rank=item.get("bm25_rank"),
                bm25_score=item.get("bm25_score"),
                rrf_score=item["rrf_score"],
                rrf_rank=item["rrf_rank"],
                cross_encoder_score=item.get("cross_encoder_score"),
                final_rank=item["final_rank"],
            )
            for item in reranked_candidates
        ]

        logger.info(
            "Hybrid search completed for '%s': returning %d reranked results",
            query,
            len(results),
        )

        return SearchResponse(
            query=query,
            workspace_id=workspace_id,
            total_results=len(results),
            retrieval_method="hybrid_rrf_cross_encoder",
            results=results,
        )


retrieval_service = RetrievalService()
