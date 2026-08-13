"""Evaluation and benchmarking module comparing 4 retrieval configurations.

Configurations:
- Configuration A: Dense vector-only (ChromaDB HNSW cosine)
- Configuration B: Sparse lexical-only (BM25Okapi)
- Configuration C: Hybrid retrieval (Vector + BM25 + Reciprocal Rank Fusion)
- Configuration D: Hybrid retrieval with Cross-Encoder reranking

Measures:
- Recall@5, Recall@10
- Precision@5
- Mean Reciprocal Rank (MRR)
- NDCG@10
- Latency (mean, median, P95)
- Cross-Encoder latency vs quality impact
- Candidate pool sensitivity analysis (10, 20, 30 candidates)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from backend.schemas.search import SearchRequest
from backend.services.bm25_service import bm25_service
from backend.services.chunking_service import chunking_service
from backend.services.embedding_service import embedding_service
from backend.services.pdf_extractor import pdf_extractor
from backend.services.reranking_service import reranking_service
from backend.services.retrieval_service import retrieval_service
from backend.services.vector_index_service import vector_index_service
from config.settings import get_settings
from evaluation.dataset_builder import DOCUMENTS_SPEC, create_pdf_file
from evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

logger = logging.getLogger("ekn.evaluation.retrieval")


class RetrievalBenchmark:
    """Orchestrates comparative benchmarking across the 4 retrieval architectures."""

    def __init__(self, workspace_id: str = "eval_workspace_01") -> None:
        self.workspace_id = workspace_id
        self.settings = get_settings()
        self.embedding_service = embedding_service
        self.vector_index_service = vector_index_service
        self.bm25_service = bm25_service
        self.reranking_service = reranking_service
        self.chunking_service = chunking_service
        self.retrieval_service = retrieval_service

        # In-memory mapping of document_id -> pages -> chunk_ids for relevance evaluation
        self.chunk_metadata_registry: Dict[str, Dict[str, Any]] = {}
        self.doc_page_to_chunks: Dict[Tuple[str, int], List[str]] = {}

    def setup_evaluation_corpus(self, work_dir: Path) -> Dict[str, Any]:
        """Generate test PDFs, extract, chunk, embed, and index into the evaluation workspace."""
        pdf_dir = work_dir / "eval_pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        total_pages = 0
        total_chunks = 0
        doc_count = len(DOCUMENTS_SPEC)

        for spec in DOCUMENTS_SPEC:
            doc_id = spec["doc_id"]
            filename = spec["filename"]
            pdf_path = pdf_dir / filename

            # 1. Create digital PDF
            create_pdf_file(pdf_path, spec["pages"])

            # 2. Extract pages via PyMuPDF
            extracted_pages = pdf_extractor.extract(pdf_path, document_id=doc_id)
            total_pages += len(extracted_pages)

            # 3. Hierarchical chunking
            parents, children = self.chunking_service.chunk_document(
                pages=extracted_pages,
                document_id=doc_id,
                workspace_id=self.workspace_id,
                original_filename=filename,
                file_hash=f"eval_hash_{doc_id}",
            )
            total_chunks += len(children)

            # 4. Save parent chunks
            self.chunking_service.save_parent_chunks(
                parents=parents,
                workspace_id=self.workspace_id,
                document_id=doc_id,
            )

            # 5. Embed & index children into ChromaDB
            if children:
                child_texts = [c.text for c in children]
                embeddings = self.embedding_service.generate_embeddings(child_texts)
                self.vector_index_service.index_child_chunks(
                    chunks=children,
                    embeddings=embeddings,
                    document_id=doc_id,
                )

                # 6. Index into BM25 corpus
                self.bm25_service.add_or_update_document_chunks(
                    workspace_id=self.workspace_id,
                    document_id=doc_id,
                    chunks=children,
                )

            # Register chunk metadata for ground-truth matching
            for c in children:
                cid = c.chunk_id
                self.chunk_metadata_registry[cid] = {
                    "document_id": doc_id,
                    "original_filename": filename,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "text": c.text,
                    "section_title": c.section_title,
                }
                for p in range(c.page_start, c.page_end + 1):
                    key = (doc_id, p)
                    if key not in self.doc_page_to_chunks:
                        self.doc_page_to_chunks[key] = []
                    self.doc_page_to_chunks[key].append(cid)

        logger.info(
            "Evaluation corpus initialized: %d documents, %d pages, %d child chunks in workspace '%s'",
            doc_count,
            total_pages,
            total_chunks,
            self.workspace_id,
        )
        return {
            "documents_indexed": doc_count,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
        }

    def _determine_relevant_chunk_ids(self, question_record: Dict[str, Any]) -> Set[str]:
        """Determine ground-truth relevant chunk IDs for a question record."""
        if question_record.get("is_unanswerable", False):
            return set()

        expected_doc = question_record.get("expected_document_id")
        expected_pages = question_record.get("expected_pages", [])

        relevant_chunks: Set[str] = set()
        if expected_doc and expected_pages:
            for p in expected_pages:
                chunks = self.doc_page_to_chunks.get((expected_doc, p), [])
                relevant_chunks.update(chunks)

        # Fallback: if no page match, match document
        if not relevant_chunks and expected_doc:
            for cid, meta in self.chunk_metadata_registry.items():
                if meta["document_id"] == expected_doc:
                    relevant_chunks.add(cid)

        return relevant_chunks

    def _determine_graded_relevance(
        self,
        question_record: Dict[str, Any],
        retrieved_ids: List[str],
    ) -> Dict[str, float]:
        """Compute graded relevance scores (0.0, 1.0, 2.0) for NDCG calculation."""
        relevance_scores: Dict[str, float] = {}
        if question_record.get("is_unanswerable", False):
            return relevance_scores

        expected_doc = question_record.get("expected_document_id")
        expected_pages = set(question_record.get("expected_pages", []))
        expected_answer = question_record.get("expected_answer", "").lower()

        for cid in retrieved_ids:
            meta = self.chunk_metadata_registry.get(cid)
            if not meta:
                relevance_scores[cid] = 0.0
                continue

            doc_match = meta["document_id"] == expected_doc
            pages = set(range(meta["page_start"], meta["page_end"] + 1))
            page_match = bool(pages.intersection(expected_pages))

            if doc_match and page_match:
                # Highly relevant if text mentions key phrase or answer
                chunk_text_lower = meta["text"].lower()
                answer_words = [w for w in expected_answer.split() if len(w) > 3]
                overlap = sum(1 for w in answer_words if w in chunk_text_lower)
                if overlap >= min(2, len(answer_words)):
                    relevance_scores[cid] = 2.0
                else:
                    relevance_scores[cid] = 1.0
            elif doc_match:
                relevance_scores[cid] = 0.5
            else:
                relevance_scores[cid] = 0.0

        return relevance_scores

    # -------------------------------------------------------------------------
    # Retrieval Execution Modes
    # -------------------------------------------------------------------------

    def retrieve_vector_only(self, query: str, top_k: int = 10) -> Tuple[List[str], float]:
        """Configuration A: Dense Vector-only search via ChromaDB HNSW cosine index."""
        start_time = time.perf_counter()
        query_vector = self.embedding_service.embed_query(query)
        where_filter = {"workspace_id": self.workspace_id}
        dense_results = self.vector_index_service.query_dense(
            query_embedding=query_vector,
            top_k=top_k,
            where_filter=where_filter,
        )
        latency = (time.perf_counter() - start_time) * 1000.0
        return [item["chunk_id"] for item in dense_results], latency

    def retrieve_bm25_only(self, query: str, top_k: int = 10) -> Tuple[List[str], float]:
        """Configuration B: Sparse Lexical-only search via BM25Okapi."""
        start_time = time.perf_counter()
        bm25_results = self.bm25_service.search(
            workspace_id=self.workspace_id,
            query=query,
            top_k=top_k,
        )
        latency = (time.perf_counter() - start_time) * 1000.0
        return [item["chunk_id"] for item in bm25_results], latency

    def retrieve_hybrid_rrf(self, query: str, top_k: int = 10) -> Tuple[List[str], float]:
        """Configuration C: Hybrid retrieval combining Dense + BM25 using Reciprocal Rank Fusion."""
        start_time = time.perf_counter()
        query_vector = self.embedding_service.embed_query(query)
        where_filter = {"workspace_id": self.workspace_id}

        dense_cands = self.vector_index_service.query_dense(
            query_embedding=query_vector,
            top_k=30,
            where_filter=where_filter,
        )
        bm25_cands = self.bm25_service.search(
            workspace_id=self.workspace_id,
            query=query,
            top_k=30,
        )

        k = self.settings.retrieval_rrf_k
        merged: Dict[str, float] = {}
        for item in dense_cands:
            cid = item["chunk_id"]
            merged[cid] = merged.get(cid, 0.0) + (1.0 / (k + item["dense_rank"]))

        for item in bm25_cands:
            cid = item["chunk_id"]
            merged[cid] = merged.get(cid, 0.0) + (1.0 / (k + item["bm25_rank"]))

        sorted_cands = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        latency = (time.perf_counter() - start_time) * 1000.0
        return [cid for cid, score in sorted_cands[:top_k]], latency

    def retrieve_hybrid_reranked(
        self,
        query: str,
        top_k: int = 10,
        rerank_pool_size: int = 30,
    ) -> Tuple[List[str], float, float]:
        """Configuration D: Hybrid retrieval with Cross-Encoder reranking.

        Returns:
            Tuple of (retrieved_chunk_ids, total_latency_ms, reranking_only_latency_ms)
        """
        start_total = time.perf_counter()

        # Step 1: Hybrid candidate generation
        query_vector = self.embedding_service.embed_query(query)
        where_filter = {"workspace_id": self.workspace_id}

        dense_cands = self.vector_index_service.query_dense(
            query_embedding=query_vector,
            top_k=rerank_pool_size,
            where_filter=where_filter,
        )
        bm25_cands = self.bm25_service.search(
            workspace_id=self.workspace_id,
            query=query,
            top_k=rerank_pool_size,
        )

        k = self.settings.retrieval_rrf_k
        merged: Dict[str, Dict[str, Any]] = {}
        for item in dense_cands:
            cid = item["chunk_id"]
            merged[cid] = {**item, "rrf_score": 1.0 / (k + item["dense_rank"])}

        for item in bm25_cands:
            cid = item["chunk_id"]
            rrf_contrib = 1.0 / (k + item["bm25_rank"])
            if cid in merged:
                merged[cid]["rrf_score"] += rrf_contrib
            else:
                merged[cid] = {**item, "rrf_score": rrf_contrib}

        fused = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)
        candidates_to_rerank = fused[:rerank_pool_size]

        # Step 2: Cross-Encoder attention scoring
        start_rerank = time.perf_counter()
        reranked = self.reranking_service.rerank(
            query=query,
            candidates=candidates_to_rerank,
            top_k=top_k,
        )
        rerank_latency = (time.perf_counter() - start_rerank) * 1000.0
        total_latency = (time.perf_counter() - start_total) * 1000.0

        return [item["chunk_id"] for item in reranked], total_latency, rerank_latency

    # -------------------------------------------------------------------------
    # Comprehensive Benchmark Runner
    # -------------------------------------------------------------------------

    def run_benchmark(
        self,
        dataset: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Execute all evaluation questions across Configurations A, B, C, D and compute metrics."""
        configs = ["config_a_vector", "config_b_bm25", "config_c_hybrid_rrf", "config_d_hybrid_reranked"]
        metrics_by_config: Dict[str, Dict[str, List[float]]] = {
            c: {"recall@5": [], "recall@10": [], "precision@5": [], "mrr": [], "ndcg@10": [], "latency": []}
            for c in configs
        }

        # Track category breakdowns
        category_mrr: Dict[str, Dict[str, List[float]]] = {c: {} for c in configs}
        rerank_latencies: List[float] = []

        # RRF vs Cross-Encoder before/after comparison
        rrf_ranks_before: List[float] = []
        ce_ranks_after: List[float] = []

        answerable_count = 0
        unanswerable_count = 0

        for record in dataset:
            q_id = record["id"]
            category = record["category"]
            query = record["question"]
            is_unanswerable = record.get("is_unanswerable", False)

            if is_unanswerable:
                unanswerable_count += 1
                continue

            answerable_count += 1
            relevant_chunks = self._determine_relevant_chunk_ids(record)

            # 1. Config A: Vector-only
            c_a_ids, c_a_lat = self.retrieve_vector_only(query, top_k=top_k)
            c_a_rel = self._determine_graded_relevance(record, c_a_ids)

            # 2. Config B: BM25-only
            c_b_ids, c_b_lat = self.retrieve_bm25_only(query, top_k=top_k)
            c_b_rel = self._determine_graded_relevance(record, c_b_ids)

            # 3. Config C: Hybrid RRF
            c_c_ids, c_c_lat = self.retrieve_hybrid_rrf(query, top_k=top_k)
            c_c_rel = self._determine_graded_relevance(record, c_c_ids)

            # 4. Config D: Hybrid + Reranking
            c_d_ids, c_d_lat, c_d_rerank_lat = self.retrieve_hybrid_reranked(query, top_k=top_k)
            c_d_rel = self._determine_graded_relevance(record, c_d_ids)
            rerank_latencies.append(c_d_rerank_lat)

            # Track RRF vs CE ranks for first relevant hit
            rrf_rr = reciprocal_rank(c_c_ids, relevant_chunks)
            ce_rr = reciprocal_rank(c_d_ids, relevant_chunks)
            rrf_ranks_before.append(rrf_rr)
            ce_ranks_after.append(ce_rr)

            # Compute and store metrics
            run_data = [
                ("config_a_vector", c_a_ids, c_a_rel, c_a_lat),
                ("config_b_bm25", c_b_ids, c_b_rel, c_b_lat),
                ("config_c_hybrid_rrf", c_c_ids, c_c_rel, c_c_lat),
                ("config_d_hybrid_reranked", c_d_ids, c_d_rel, c_d_lat),
            ]

            for cfg_name, ret_ids, rel_dict, lat in run_data:
                rec5 = recall_at_k(ret_ids, relevant_chunks, k=5)
                rec10 = recall_at_k(ret_ids, relevant_chunks, k=10)
                prec5 = precision_at_k(ret_ids, relevant_chunks, k=5)
                mrr_val = reciprocal_rank(ret_ids, relevant_chunks)
                ndcg10 = ndcg_at_k(ret_ids, rel_dict, k=10)

                metrics_by_config[cfg_name]["recall@5"].append(rec5)
                metrics_by_config[cfg_name]["recall@10"].append(rec10)
                metrics_by_config[cfg_name]["precision@5"].append(prec5)
                metrics_by_config[cfg_name]["mrr"].append(mrr_val)
                metrics_by_config[cfg_name]["ndcg@10"].append(ndcg10)
                metrics_by_config[cfg_name]["latency"].append(lat)

                if category not in category_mrr[cfg_name]:
                    category_mrr[cfg_name][category] = []
                category_mrr[cfg_name][category].append(mrr_val)

        # Aggregate metrics
        summary: Dict[str, Any] = {
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": len(dataset),
            "answerable_queries": answerable_count,
            "unanswerable_queries": unanswerable_count,
            "configurations": {},
            "category_performance": {},
            "cross_encoder_impact": {
                "mean_mrr_before_reranking_rrf": round(float(np.mean(rrf_ranks_before)), 4) if rrf_ranks_before else 0.0,
                "mean_mrr_after_cross_encoder": round(float(np.mean(ce_ranks_after)), 4) if ce_ranks_after else 0.0,
                "mrr_improvement_percent": round(
                    ((np.mean(ce_ranks_after) - np.mean(rrf_ranks_before)) / (np.mean(rrf_ranks_before) + 1e-9)) * 100.0, 2
                ) if rrf_ranks_before else 0.0,
                "mean_rerank_overhead_ms": round(float(np.mean(rerank_latencies)), 2) if rerank_latencies else 0.0,
                "median_rerank_overhead_ms": round(float(np.median(rerank_latencies)), 2) if rerank_latencies else 0.0,
                "p95_rerank_overhead_ms": round(float(np.percentile(rerank_latencies, 95)), 2) if rerank_latencies else 0.0,
            },
        }

        for cfg_name, vals in metrics_by_config.items():
            lats = vals["latency"]
            summary["configurations"][cfg_name] = {
                "recall@5": round(float(np.mean(vals["recall@5"])), 4),
                "recall@10": round(float(np.mean(vals["recall@10"])), 4),
                "precision@5": round(float(np.mean(vals["precision@5"])), 4),
                "mrr": round(float(np.mean(vals["mrr"])), 4),
                "ndcg@10": round(float(np.mean(vals["ndcg@10"])), 4),
                "latency_mean_ms": round(float(np.mean(lats)), 2),
                "latency_median_ms": round(float(np.median(lats)), 2),
                "latency_p95_ms": round(float(np.percentile(lats, 95)), 2) if len(lats) >= 5 else round(float(np.max(lats)), 2),
            }

        # Category performance breakdown
        all_categories = sorted(list(category_mrr["config_a_vector"].keys()))
        for cat in all_categories:
            summary["category_performance"][cat] = {
                "vector_mrr": round(float(np.mean(category_mrr["config_a_vector"][cat])), 4),
                "bm25_mrr": round(float(np.mean(category_mrr["config_b_bm25"][cat])), 4),
                "hybrid_rrf_mrr": round(float(np.mean(category_mrr["config_c_hybrid_rrf"][cat])), 4),
                "reranked_mrr": round(float(np.mean(category_mrr["config_d_hybrid_reranked"][cat])), 4),
            }

        return summary

    def benchmark_candidate_pool_sizes(
        self,
        dataset: List[Dict[str, Any]],
        candidate_sizes: List[int] = [10, 20, 30],
    ) -> Dict[int, Dict[str, float]]:
        """Evaluate trade-off between reranking candidate pool size, MRR, NDCG@10, and latency."""
        results: Dict[int, Dict[str, float]] = {}
        for pool_size in candidate_sizes:
            mrrs: List[float] = []
            ndcgs: List[float] = []
            latencies: List[float] = []

            for record in dataset:
                if record.get("is_unanswerable", False):
                    continue
                query = record["question"]
                relevant_chunks = self._determine_relevant_chunk_ids(record)

                ret_ids, tot_lat, _ = self.retrieve_hybrid_reranked(
                    query=query,
                    top_k=10,
                    rerank_pool_size=pool_size,
                )
                rel_dict = self._determine_graded_relevance(record, ret_ids)

                mrrs.append(reciprocal_rank(ret_ids, relevant_chunks))
                ndcgs.append(ndcg_at_k(ret_ids, rel_dict, k=10))
                latencies.append(tot_lat)

            results[pool_size] = {
                "mrr": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
                "ndcg@10": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
                "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
                "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2) if len(latencies) >= 5 else 0.0,
            }

        return results
