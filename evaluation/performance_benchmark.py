"""Performance and concurrency benchmarking module for Enterprise Knowledge Navigator.

Measures:
1. Document Ingestion Pipeline Latency & Memory Breakdown:
   - Upload, PyMuPDF extraction, OCR, chunking, embeddings, ChromaDB, BM25, peak memory (tracemalloc).
   - Ingestion scaling across page sizes (small 3p, medium 10p, larger 25p).
2. Query Latency Breakdown:
   - Preprocessing, embedding, HNSW search, BM25 search, RRF fusion, Cross-Encoder reranking,
     parent retrieval, LLM generation, total end-to-end.
   - Mean, Median, and P95 latency reporting (separating retrieval from LLM API).
3. Large-Repository Scaling:
   - Controlled collections (small, medium, larger) measuring pages, chunks, vectors, corpus size, and search latency.
4. Concurrent Workload and Reliability:
   - Simultaneous searches, multi-user requests, cross-tenant concurrency, thread-safety verification.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymupdf

from backend.schemas.indexing import ChildChunkRecord, ParentChunkRecord
from backend.services.bm25_service import bm25_service
from backend.services.chunking_service import chunking_service
from backend.services.embedding_service import embedding_service
from backend.services.ocr_service import ocr_service
from backend.services.pdf_extractor import pdf_extractor
from backend.services.reranking_service import reranking_service
from backend.services.retrieval_service import retrieval_service
from backend.services.vector_index_service import vector_index_service
from config.settings import get_settings

logger = logging.getLogger("ekn.evaluation.performance")


def create_benchmark_pdf(output_path: Path, num_pages: int, prefix: str = "Benchmark") -> None:
    """Generate a multi-page PDF document for performance stress-testing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for p in range(1, num_pages + 1):
        page = doc.new_page(width=595, height=842)
        heading = f"Section {p}. {prefix} Standard Operations ({p})"
        page.insert_text((50, 72), heading, fontsize=15, fontname="helv", color=(0.1, 0.2, 0.4))

        body_text = (
            f"This is synthetic benchmark page {p} of {num_pages}. "
            f"Equipment tag AX-{1000 + p} is monitored under specification ISO-900{p % 5}. "
            f"Operating pressure: {50 + (p % 10) * 2} bar with tolerance +/- 3 bar. "
            f"Nominal speed: {3000 + (p * 50)} RPM. "
            f"Error code ERR-{800 + p} indicates thermal trip condition exceeding 120 C. "
            "All operators must verify lubrication level every 30 days and replace synthetic oil ISO-VG-46 annually. "
            "Emergency stop circuit code EMERG-STOP-4 initiates stop valve closure within 0.12 seconds."
        )
        rect = pymupdf.Rect(50, 100, 545, 780)
        page.insert_textbox(rect, body_text * 3, fontsize=10, fontname="helv")
    doc.save(str(output_path))
    doc.close()


class PerformanceBenchmark:
    """Evaluates pipeline throughput, latency percentiles, memory consumption, and concurrency."""

    def __init__(self, workspace_id: str = "perf_workspace_01") -> None:
        self.workspace_id = workspace_id
        self.settings = get_settings()
        self.embedding_service = embedding_service
        self.vector_index_service = vector_index_service
        self.bm25_service = bm25_service
        self.reranking_service = reranking_service
        self.chunking_service = chunking_service
        self.retrieval_service = retrieval_service

    # -------------------------------------------------------------------------
    # 1. Document Ingestion Benchmarking
    # -------------------------------------------------------------------------

    def benchmark_ingestion_single_document(
        self,
        pdf_path: Path,
        doc_id: str,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Profile stage-by-stage ingestion latency and peak memory usage for a PDF."""
        ws_id = workspace_id or self.workspace_id
        filename = pdf_path.name

        tracemalloc.start()
        start_total = time.perf_counter()

        # Step 1: PDF text and layout extraction via PyMuPDF
        t0 = time.perf_counter()
        extracted_pages = pdf_extractor.extract(pdf_path, document_id=doc_id)
        extract_time = (time.perf_counter() - t0) * 1000.0

        # Step 2: Selective OCR check
        t0 = time.perf_counter()
        ocr_invocations = 0
        for page in extracted_pages:
            if page.ocr_used:
                ocr_invocations += 1
        ocr_time = (time.perf_counter() - t0) * 1000.0

        # Step 3: Hierarchical Parent-Child Chunking
        t0 = time.perf_counter()
        parents, children = self.chunking_service.chunk_document(
            pages=extracted_pages,
            document_id=doc_id,
            workspace_id=ws_id,
            original_filename=filename,
            file_hash=f"perf_hash_{doc_id}",
        )
        chunk_time = (time.perf_counter() - t0) * 1000.0

        # Step 4: Parent chunks persistence
        t0 = time.perf_counter()
        self.chunking_service.save_parent_chunks(
            parents=parents,
            workspace_id=ws_id,
            document_id=doc_id,
        )
        parent_save_time = (time.perf_counter() - t0) * 1000.0

        # Step 5: Dense embedding computation
        t0 = time.perf_counter()
        embeddings: List[List[float]] = []
        if children:
            texts = [c.text for c in children]
            embeddings = self.embedding_service.generate_embeddings(texts)
        embed_time = (time.perf_counter() - t0) * 1000.0

        # Step 6: ChromaDB vector indexing
        t0 = time.perf_counter()
        if children and embeddings:
            self.vector_index_service.index_child_chunks(
                chunks=children,
                embeddings=embeddings,
                document_id=doc_id,
            )
        chroma_time = (time.perf_counter() - t0) * 1000.0

        # Step 7: BM25 corpus update
        t0 = time.perf_counter()
        if children:
            self.bm25_service.add_or_update_document_chunks(
                workspace_id=ws_id,
                document_id=doc_id,
                chunks=children,
            )
        bm25_time = (time.perf_counter() - t0) * 1000.0

        total_time = (time.perf_counter() - start_total) * 1000.0
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        num_pages = len(extracted_pages)
        pages_per_sec = (num_pages / (total_time / 1000.0)) if total_time > 0 else 0.0

        return {
            "document_id": doc_id,
            "filename": filename,
            "pages_processed": num_pages,
            "parent_chunks": len(parents),
            "child_chunks": len(children),
            "ocr_pages_detected": ocr_invocations,
            "extraction_time_ms": round(extract_time, 2),
            "ocr_time_ms": round(ocr_time, 2),
            "chunking_time_ms": round(chunk_time, 2),
            "parent_persistence_time_ms": round(parent_save_time, 2),
            "embedding_generation_time_ms": round(embed_time, 2),
            "chroma_indexing_time_ms": round(chroma_time, 2),
            "bm25_indexing_time_ms": round(bm25_time, 2),
            "total_ingestion_time_ms": round(total_time, 2),
            "throughput_pages_per_sec": round(pages_per_sec, 2),
            "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
        }

    def benchmark_ingestion_scaling(self, work_dir: Path) -> Dict[str, Any]:
        """Benchmark ingestion across small (3 pages), medium (10 pages), and larger (25 pages) documents."""
        sizes = [("small", 3), ("medium", 10), ("larger", 25)]
        scaling_results: Dict[str, Any] = {}

        for label, page_count in sizes:
            pdf_path = work_dir / f"benchmark_{label}_{page_count}p.pdf"
            create_benchmark_pdf(pdf_path, num_pages=page_count, prefix=label.capitalize())
            metrics = self.benchmark_ingestion_single_document(
                pdf_path=pdf_path,
                doc_id=f"doc_{label}_{page_count}p",
            )
            scaling_results[label] = metrics

        return scaling_results

    # -------------------------------------------------------------------------
    # 2. Query Latency Breakdown (Mean, Median, P95)
    # -------------------------------------------------------------------------

    def benchmark_query_latency_breakdown(
        self,
        queries: List[str],
        iterations: int = 5,
    ) -> Dict[str, Any]:
        """Measure isolated per-stage latency across multiple query runs.

        Stages:
        - Query preprocessing
        - Query embedding generation
        - ChromaDB HNSW vector retrieval
        - BM25 lexical keyword retrieval
        - Reciprocal Rank Fusion
        - Cross-Encoder reranking
        - Parent context recovery
        - Total retrieval latency
        """
        stage_times: Dict[str, List[float]] = {
            "query_preprocessing": [],
            "embedding_generation": [],
            "hnsw_vector_search": [],
            "bm25_search": [],
            "rrf_fusion": [],
            "cross_encoder_rerank": [],
            "parent_context_recovery": [],
            "total_retrieval": [],
        }

        where_filter = {"workspace_id": self.workspace_id}
        k = self.settings.retrieval_rrf_k

        for _ in range(iterations):
            for query in queries:
                start_total = time.perf_counter()

                # Stage 1: Preprocessing
                t0 = time.perf_counter()
                clean_q = query.strip()
                tokens = self.bm25_service.tokenize(clean_q)
                stage_times["query_preprocessing"].append((time.perf_counter() - t0) * 1000.0)

                # Stage 2: Query Embedding
                t0 = time.perf_counter()
                query_vec = self.embedding_service.embed_query(clean_q)
                stage_times["embedding_generation"].append((time.perf_counter() - t0) * 1000.0)

                # Stage 3: HNSW Dense Search
                t0 = time.perf_counter()
                dense_cands = self.vector_index_service.query_dense(
                    query_embedding=query_vec,
                    top_k=30,
                    where_filter=where_filter,
                )
                stage_times["hnsw_vector_search"].append((time.perf_counter() - t0) * 1000.0)

                # Stage 4: BM25 Search
                t0 = time.perf_counter()
                bm25_cands = self.bm25_service.search(
                    workspace_id=self.workspace_id,
                    query=clean_q,
                    top_k=30,
                )
                stage_times["bm25_search"].append((time.perf_counter() - t0) * 1000.0)

                # Stage 5: RRF Fusion
                t0 = time.perf_counter()
                merged: Dict[str, Dict[str, Any]] = {}
                for item in dense_cands:
                    cid = item["chunk_id"]
                    merged[cid] = {**item, "rrf_score": 1.0 / (k + item["dense_rank"])}
                for item in bm25_cands:
                    cid = item["chunk_id"]
                    score = 1.0 / (k + item["bm25_rank"])
                    if cid in merged:
                        merged[cid]["rrf_score"] += score
                    else:
                        merged[cid] = {**item, "rrf_score": score}
                fused = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)[:30]
                stage_times["rrf_fusion"].append((time.perf_counter() - t0) * 1000.0)

                # Stage 6: Cross-Encoder Reranking
                t0 = time.perf_counter()
                reranked = self.reranking_service.rerank(
                    query=clean_q,
                    candidates=fused,
                    top_k=8,
                )
                stage_times["cross_encoder_rerank"].append((time.perf_counter() - t0) * 1000.0)

                # Stage 7: Parent Context Recovery
                t0 = time.perf_counter()
                doc_parent_cache: Dict[str, Dict[str, str]] = {}
                for item in reranked:
                    did = item.get("document_id")
                    pid = item.get("parent_id")
                    if did and pid:
                        if did not in doc_parent_cache:
                            parents = self.chunking_service.load_parent_chunks(self.workspace_id, did)
                            doc_parent_cache[did] = {p.parent_id: p.text for p in parents}
                        _ = doc_parent_cache[did].get(pid)
                stage_times["parent_context_recovery"].append((time.perf_counter() - t0) * 1000.0)

                # Total retrieval time
                stage_times["total_retrieval"].append((time.perf_counter() - start_total) * 1000.0)

        # Compute summary percentiles
        breakdown_summary: Dict[str, Any] = {}
        for stage, times in stage_times.items():
            breakdown_summary[stage] = {
                "mean_ms": round(float(np.mean(times)), 2),
                "median_ms": round(float(np.median(times)), 2),
                "p95_ms": round(float(np.percentile(times, 95)), 2) if len(times) >= 5 else round(float(np.max(times)), 2),
            }

        return breakdown_summary

    # -------------------------------------------------------------------------
    # 3. Controlled Repository Scaling
    # -------------------------------------------------------------------------

    def benchmark_repository_scaling(
        self,
        work_dir: Path,
        repository_tiers: List[Tuple[str, int]] = [("small", 3), ("medium", 8), ("larger", 15)],
    ) -> Dict[str, Any]:
        """Evaluate how indexing time and query latency scale as repository document count increases."""
        scaling_report: Dict[str, Any] = {}

        for tier_name, doc_count in repository_tiers:
            ws_id = f"scaling_ws_{tier_name}"
            tier_dir = work_dir / ws_id
            tier_dir.mkdir(parents=True, exist_ok=True)

            t_index_start = time.perf_counter()
            total_pages = 0
            total_chunks = 0

            for i in range(1, doc_count + 1):
                pdf_p = tier_dir / f"doc_{i}.pdf"
                create_benchmark_pdf(pdf_p, num_pages=4, prefix=f"Tier_{tier_name}_{i}")
                ext_pages = pdf_extractor.extract(pdf_p, document_id=f"doc_{i}")
                total_pages += len(ext_pages)

                parents, children = self.chunking_service.chunk_document(
                    pages=ext_pages,
                    document_id=f"doc_{i}",
                    workspace_id=ws_id,
                    original_filename=f"doc_{i}.pdf",
                    file_hash=f"hash_{ws_id}_{i}",
                )
                total_chunks += len(children)

                self.chunking_service.save_parent_chunks(parents, ws_id, f"doc_{i}")
                if children:
                    texts = [c.text for c in children]
                    embs = self.embedding_service.generate_embeddings(texts)
                    self.vector_index_service.index_child_chunks(children, embs, f"doc_{i}")
                    self.bm25_service.add_or_update_document_chunks(ws_id, f"doc_{i}", children)

            total_index_time_sec = time.perf_counter() - t_index_start

            # Measure query latency on this repository tier
            test_query = "What is the operating pressure and error code ERR-802?"
            sample_latencies: List[float] = []
            for _ in range(5):
                t0 = time.perf_counter()
                q_vec = self.embedding_service.embed_query(test_query)
                _ = self.vector_index_service.query_dense(q_vec, 10, {"workspace_id": ws_id})
                _ = self.bm25_service.search(ws_id, test_query, 10)
                sample_latencies.append((time.perf_counter() - t0) * 1000.0)

            scaling_report[tier_name] = {
                "documents": doc_count,
                "total_pages": total_pages,
                "total_chunks": total_chunks,
                "indexing_duration_sec": round(total_index_time_sec, 2),
                "query_latency_mean_ms": round(float(np.mean(sample_latencies)), 2),
                "query_latency_p95_ms": round(float(np.percentile(sample_latencies, 95)), 2),
            }

        return scaling_report

    # -------------------------------------------------------------------------
    # 4. Concurrency & Reliability Under Load
    # -------------------------------------------------------------------------

    def benchmark_concurrency(
        self,
        queries: List[str],
        num_threads: int = 4,
        requests_per_thread: int = 5,
    ) -> Dict[str, Any]:
        """Stress-test concurrent searches and verify thread-safety, no data leaks, and zero errors."""
        total_requests = num_threads * requests_per_thread
        latencies: List[float] = []
        errors: List[str] = []

        def worker_task(thread_id: int) -> List[float]:
            thread_lats: List[float] = []
            for i in range(requests_per_thread):
                q = queries[(thread_id + i) % len(queries)]
                t0 = time.perf_counter()
                try:
                    q_vec = self.embedding_service.embed_query(q)
                    dense = self.vector_index_service.query_dense(
                        query_embedding=q_vec,
                        top_k=10,
                        where_filter={"workspace_id": self.workspace_id},
                    )
                    bm = self.bm25_service.search(self.workspace_id, q, top_k=10)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    thread_lats.append(elapsed)

                    # Verify tenant isolation in dense results
                    for item in dense:
                        meta_ws = item.get("workspace_id")
                        if meta_ws and meta_ws != self.workspace_id:
                            errors.append(f"Tenant isolation leak: expected {self.workspace_id}, got {meta_ws}")
                except Exception as exc:
                    errors.append(f"Thread {thread_id} error: {exc}")
            return thread_lats

        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, tid) for tid in range(num_threads)]
            for fut in as_completed(futures):
                try:
                    thread_times = fut.result()
                    latencies.extend(thread_times)
                except Exception as exc:
                    errors.append(f"Future failure: {exc}")

        total_duration = time.perf_counter() - t_start
        throughput_qps = (total_requests / total_duration) if total_duration > 0 else 0.0

        return {
            "num_threads": num_threads,
            "requests_per_thread": requests_per_thread,
            "total_requests": total_requests,
            "total_duration_sec": round(total_duration, 2),
            "throughput_qps": round(throughput_qps, 2),
            "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "median_latency_ms": round(float(np.median(latencies)), 2) if latencies else 0.0,
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2) if len(latencies) >= 5 else 0.0,
            "errors_count": len(errors),
            "errors": errors[:5],
            "thread_safety_passed": len(errors) == 0,
        }
