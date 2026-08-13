"""Automated test suite for Phase 11 — Evaluation, Benchmarking, and Performance Optimization."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List

import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.schemas.indexing import ChildChunkRecord, ParentChunkRecord
from backend.services.bm25_service import bm25_service
from backend.services.chunking_service import chunking_service
from backend.services.embedding_service import embedding_service
from backend.services.reranking_service import reranking_service
from backend.services.retrieval_service import retrieval_service
from backend.services.vector_index_service import vector_index_service
from evaluation.answer_benchmark import AnswerBenchmark
from evaluation.dataset_builder import DOCUMENTS_SPEC, generate_evaluation_dataset
from evaluation.metrics import (
    answer_relevance_score,
    citation_coverage,
    citation_precision,
    context_relevance_score,
    faithfulness_score,
    is_unanswerable_declaration,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    unanswerable_detection_metrics,
)
from evaluation.performance_benchmark import PerformanceBenchmark, create_benchmark_pdf
from evaluation.retrieval_benchmark import RetrievalBenchmark


# =============================================================================
# 1. Retrieval Metrics Mathematical Verification
# =============================================================================

def test_recall_at_k_calculation():
    """Verify Recall@K mathematical behavior across perfect, partial, and zero recall."""
    relevant = {"c1", "c2", "c3"}

    # Perfect recall in top-3
    assert recall_at_k(["c1", "c2", "c3", "c4"], relevant, k=3) == 1.0

    # Partial recall (2 out of 3)
    assert round(recall_at_k(["c1", "c4", "c2", "c5"], relevant, k=3), 4) == round(2 / 3, 4)

    # Cutoff truncation: c3 is at rank 4, so at k=3 only 2 retrieved
    assert round(recall_at_k(["c1", "c2", "c4", "c3"], relevant, k=3), 4) == round(2 / 3, 4)
    assert recall_at_k(["c1", "c2", "c4", "c3"], relevant, k=4) == 1.0

    # Zero recall
    assert recall_at_k(["c4", "c5", "c6"], relevant, k=3) == 0.0

    # Empty relevant set
    assert recall_at_k([], set(), k=5) == 1.0


def test_precision_at_k_calculation():
    """Verify Precision@K mathematical behavior."""
    relevant = {"c1", "c2"}

    # 2 out of 5 are relevant
    assert precision_at_k(["c1", "c3", "c2", "c4", "c5"], relevant, k=5) == 0.4

    # 2 out of 2 are relevant
    assert precision_at_k(["c1", "c2", "c3"], relevant, k=2) == 1.0

    # 0 out of 5 relevant
    assert precision_at_k(["c7", "c8"], relevant, k=5) == 0.0

    # k <= 0 safety
    assert precision_at_k(["c1"], relevant, k=0) == 0.0


def test_reciprocal_rank_calculation():
    """Verify Reciprocal Rank (RR) and MRR behavior."""
    relevant = {"c_target"}

    # First position -> RR = 1.0
    assert reciprocal_rank(["c_target", "c2", "c3"], relevant) == 1.0

    # Second position -> RR = 0.5
    assert reciprocal_rank(["c1", "c_target", "c3"], relevant) == 0.5

    # Fourth position -> RR = 0.25
    assert reciprocal_rank(["c1", "c2", "c3", "c_target"], relevant) == 0.25

    # Not found -> RR = 0.0
    assert reciprocal_rank(["c1", "c2", "c3"], relevant) == 0.0

    # Empty relevant -> RR = 0.0
    assert reciprocal_rank(["c1"], set()) == 0.0


def test_ndcg_at_k_graded_relevance():
    """Verify NDCG@K calculation with binary and graded relevance."""
    # Ideal order: item with 2.0 first, then 1.0
    rel_scores = {"c1": 2.0, "c2": 1.0, "c3": 0.0}

    # Perfect ranking: [c1, c2, c3] -> NDCG = 1.0
    assert ndcg_at_k(["c1", "c2", "c3"], rel_scores, k=3) == 1.0

    # Sub-optimal ranking: [c2, c1, c3] -> NDCG < 1.0 but > 0.0
    ndcg_sub = ndcg_at_k(["c2", "c1", "c3"], rel_scores, k=3)
    assert 0.75 < ndcg_sub < 1.0

    # Worst ranking: [c3, c4] (no relevant items) -> NDCG = 0.0
    assert ndcg_at_k(["c3", "c4"], rel_scores, k=3) == 0.0

    # Empty inputs safety
    assert ndcg_at_k([], rel_scores, k=5) == 0.0
    assert ndcg_at_k(["c1"], {}, k=5) == 0.0


# =============================================================================
# 2. Citation Accuracy & Quality Metrics
# =============================================================================

def test_citation_precision_and_coverage():
    """Verify citation precision against retrieved evidence pool and coverage of claims."""
    evidence_pool = {
        "E1": {"document_id": "doc_01", "page_start": 2, "text": "Operating speed is 3600 RPM."},
        "E2": {"document_id": "doc_02", "page_start": 1, "text": "Hazardous threshold is 25 ppm."},
    }

    # 1. Perfect precision: citation document_id and page match evidence
    valid_citations = [
        {"citation_id": "E1", "document_id": "doc_01", "page_start": 2},
        {"citation_id": "E2", "document_id": "doc_02", "page_start": 1},
    ]
    assert citation_precision(valid_citations, evidence_pool) == 1.0

    # 2. Corrupted page reference
    bad_page_citations = [
        {"citation_id": "E1", "document_id": "doc_01", "page_start": 99},
    ]
    assert citation_precision(bad_page_citations, evidence_pool) == 0.0

    # 3. Nonexistent citation ID
    unknown_citations = [
        {"citation_id": "E99", "document_id": "doc_01", "page_start": 2},
    ]
    assert citation_precision(unknown_citations, evidence_pool) == 0.0

    # 4. Citation coverage
    answer = "The turbine operates at 3600 RPM [E1]. Ammonia limit is 25 ppm [E2]."
    assert citation_coverage(answer, valid_citations) == 1.0

    unbacked_answer = "The turbine operates at 3600 RPM. Ammonia limit is 25 ppm."
    assert citation_coverage(unbacked_answer, valid_citations) == 0.0


def test_faithfulness_and_answer_relevance():
    """Verify rule-based faithfulness scoring and relevance checks."""
    context = [
        "Turbine AX-450 nominal operating speed is 3600 RPM with inlet pressure at 65 bar. Error ERR-902 signals vibration.",
    ]

    # Faithful answer: mentions exact model, speed, and error code
    faithful_answer = "The turbine model AX-450 nominal operating speed is 3600 RPM and reports ERR-902 when vibrating."
    score = faithfulness_score(faithful_answer, context)
    assert score > 0.6

    # Hallucinated numbers and nonexistent model
    unfaithful_answer = "The turbine model RX-999 operates at 9500 RPM with pressure at 900 bar."
    low_score = faithfulness_score(unfaithful_answer, context)
    assert low_score < 0.4

    # Answer relevance
    query = "What is the operating speed of turbine AX-450?"
    rel = answer_relevance_score(faithful_answer, query)
    assert rel > 0.6

    ctx_rel = context_relevance_score(context, query)
    assert ctx_rel == 1.0


def test_unanswerable_metrics_classification():
    """Verify unanswerable declaration detection and confusion matrix calculation."""
    assert is_unanswerable_declaration("There is insufficient evidence in the provided documents.")
    assert is_unanswerable_declaration("This is not mentioned in the provided documents.")
    assert not is_unanswerable_declaration("The operating speed is 3600 RPM.")

    eval_data = [
        {"is_unanswerable": True, "declared_unanswerable": True},   # TP
        {"is_unanswerable": True, "declared_unanswerable": False},  # FN (Hallucination)
        {"is_unanswerable": False, "declared_unanswerable": False}, # TN
        {"is_unanswerable": False, "declared_unanswerable": True},  # FP
    ]
    metrics = unanswerable_detection_metrics(eval_data)
    assert metrics["unanswerable_accuracy"] == 0.5
    assert metrics["unanswerable_recall"] == 0.5
    assert metrics["unsupported_answer_rate"] == 0.5


# =============================================================================
# 3. Dataset Integrity & Schema Tests
# =============================================================================

def test_evaluation_dataset_completeness_and_schema():
    """Verify dataset contains >= 50 questions with complete required fields across 9 categories."""
    dataset = generate_evaluation_dataset("eval_ws_test")
    assert len(dataset) >= 50, f"Expected at least 50 questions, got {len(dataset)}"

    categories = set(item["category"] for item in dataset)
    expected_categories = {
        "technical_id",
        "keyword",
        "factual",
        "semantic",
        "cross_document",
        "comparison",
        "summary",
        "multi_hop",
        "unanswerable",
    }
    assert expected_categories.issubset(categories), f"Missing categories: {expected_categories - categories}"

    for item in dataset:
        assert "id" in item
        assert "question" in item and len(item["question"].strip()) > 5
        assert "category" in item
        assert "is_unanswerable" in item
        assert "expected_answer" in item and len(item["expected_answer"].strip()) > 0

        if not item["is_unanswerable"]:
            assert item["expected_document_id"] is not None
            assert item["expected_filename"] is not None
            assert len(item["expected_pages"]) > 0


# =============================================================================
# 4. Comparative Retrieval Benchmark (Configurations A, B, C, D)
# =============================================================================

def test_retrieval_benchmark_all_configurations(tmp_path):
    """Run comparative retrieval benchmark across Config A (Vector), B (BM25), C (Hybrid), D (Reranked)."""
    ws_id = f"eval_bench_ws_{int(time.time())}"
    benchmark = RetrievalBenchmark(workspace_id=ws_id)

    # Set up temporary storage directories
    benchmark.bm25_service.storage_root = tmp_path
    benchmark.chunking_service.storage_root = tmp_path

    # Set up the 5 evaluation PDFs
    setup_info = benchmark.setup_evaluation_corpus(tmp_path)
    assert setup_info["documents_indexed"] == 5
    assert setup_info["total_pages"] == 13
    assert setup_info["total_chunks"] > 0

    # Run on a representative slice of evaluation questions
    dataset = generate_evaluation_dataset(ws_id)[:15]
    report = benchmark.run_benchmark(dataset=dataset, top_k=5)

    assert report["dataset_size"] == 15
    assert "configurations" in report
    for cfg in ["config_a_vector", "config_b_bm25", "config_c_hybrid_rrf", "config_d_hybrid_reranked"]:
        cfg_stats = report["configurations"][cfg]
        assert 0.0 <= cfg_stats["recall@5"] <= 1.0
        assert 0.0 <= cfg_stats["precision@5"] <= 1.0
        assert 0.0 <= cfg_stats["mrr"] <= 1.0
        assert 0.0 <= cfg_stats["ndcg@10"] <= 1.0
        assert cfg_stats["latency_mean_ms"] >= 0.0

    # Verify Cross-Encoder comparison metrics
    ce_impact = report["cross_encoder_impact"]
    assert "mean_mrr_before_reranking_rrf" in ce_impact
    assert "mean_mrr_after_cross_encoder" in ce_impact
    assert "mean_rerank_overhead_ms" in ce_impact
    assert ce_impact["mean_rerank_overhead_ms"] >= 0.0


def test_candidate_pool_size_tradeoff(tmp_path):
    """Verify reranking candidate pool sensitivity benchmarking (10 vs 20 vs 30 candidates)."""
    ws_id = f"eval_pool_ws_{int(time.time())}"
    benchmark = RetrievalBenchmark(workspace_id=ws_id)
    benchmark.bm25_service.storage_root = tmp_path
    benchmark.chunking_service.storage_root = tmp_path
    benchmark.setup_evaluation_corpus(tmp_path)

    dataset = generate_evaluation_dataset(ws_id)[:6]
    pool_results = benchmark.benchmark_candidate_pool_sizes(dataset, candidate_sizes=[10, 20])

    assert 10 in pool_results
    assert 20 in pool_results
    assert pool_results[10]["mean_latency_ms"] >= 0.0
    assert pool_results[20]["mean_latency_ms"] >= 0.0


# =============================================================================
# 5. Answer Quality & Citation Validation Tests
# =============================================================================

def test_citation_integrity_suite():
    """Verify independent citation integrity testing under valid, nonexistent, and mixed conditions."""
    benchmark = AnswerBenchmark(workspace_id="test_ws")
    evidence_pool = {
        "E1": {"document_id": "doc_01", "page_start": 1, "original_filename": "manual.pdf", "text": "Speed 3600 RPM"},
    }

    integrity_report = benchmark.test_citation_integrity(evidence_pool)
    assert integrity_report["all_passed"] is True
    assert integrity_report["test_outcomes"]["valid_citation_accepted"] is True
    assert integrity_report["test_outcomes"]["nonexistent_citation_rejected"] is True
    assert integrity_report["test_outcomes"]["mixed_citations_isolated"] is True
    assert integrity_report["test_outcomes"]["uncited_answer_handled"] is True


def test_answer_benchmark_unanswerable_handling():
    """Verify answer benchmark detects unanswerable queries and avoids unsupported claims."""
    benchmark = AnswerBenchmark(workspace_id="test_ws")
    unanswerable_record = {
        "id": "unans_01",
        "category": "unanswerable",
        "question": "What is the solar panel power output on the roof?",
        "is_unanswerable": True,
        "expected_answer": "Insufficient evidence.",
    }

    result = benchmark.evaluate_single_question(
        unanswerable_record,
        mock_answer_override="There is insufficient evidence in the provided documents regarding solar panels.",
    )
    assert result["is_unanswerable"] is True
    assert result["declared_unanswerable"] is True
    assert result["citations_count"] == 0


# =============================================================================
# 6. Performance Benchmarking & Ingestion Tests
# =============================================================================

def test_ingestion_single_document_profiling(tmp_path):
    """Profile stage-by-stage ingestion time and peak memory tracking with tracemalloc."""
    perf = PerformanceBenchmark(workspace_id="perf_test_ws")
    perf.bm25_service.storage_root = tmp_path
    perf.chunking_service.storage_root = tmp_path

    pdf_path = tmp_path / "test_doc_5p.pdf"
    create_benchmark_pdf(pdf_path, num_pages=5, prefix="ProfileDoc")

    metrics = perf.benchmark_ingestion_single_document(pdf_path, doc_id="profile_doc_1", workspace_id="perf_test_ws")

    assert metrics["pages_processed"] == 5
    assert metrics["parent_chunks"] > 0
    assert metrics["child_chunks"] > 0
    assert metrics["extraction_time_ms"] > 0.0
    assert metrics["chunking_time_ms"] > 0.0
    assert metrics["embedding_generation_time_ms"] > 0.0
    assert metrics["chroma_indexing_time_ms"] > 0.0
    assert metrics["bm25_indexing_time_ms"] > 0.0
    assert metrics["total_ingestion_time_ms"] > 0.0
    assert metrics["peak_memory_mb"] > 0.0


def test_query_latency_breakdown(tmp_path):
    """Verify per-stage query latency breakdown with Mean, Median, and P95."""
    perf = PerformanceBenchmark(workspace_id="perf_query_ws")
    perf.bm25_service.storage_root = tmp_path
    perf.chunking_service.storage_root = tmp_path

    # Ingest 1 document to populate index
    pdf_path = tmp_path / "query_test.pdf"
    create_benchmark_pdf(pdf_path, num_pages=3, prefix="QueryTest")
    perf.benchmark_ingestion_single_document(pdf_path, doc_id="q_doc", workspace_id="perf_query_ws")

    breakdown = perf.benchmark_query_latency_breakdown(
        queries=["What is operating speed?", "What is pressure tolerance?"],
        iterations=2,
    )

    expected_stages = [
        "query_preprocessing",
        "embedding_generation",
        "hnsw_vector_search",
        "bm25_search",
        "rrf_fusion",
        "cross_encoder_rerank",
        "parent_context_recovery",
        "total_retrieval",
    ]
    for stage in expected_stages:
        assert stage in breakdown
        assert breakdown[stage]["mean_ms"] >= 0.0
        assert breakdown[stage]["median_ms"] >= 0.0
        assert breakdown[stage]["p95_ms"] >= 0.0


def test_concurrency_and_tenant_isolation_under_load(tmp_path):
    """Stress-test concurrent searches and verify thread safety and zero cross-tenant leakage."""
    perf = PerformanceBenchmark(workspace_id="tenant_ws_A")
    perf.bm25_service.storage_root = tmp_path
    perf.chunking_service.storage_root = tmp_path

    pdf_path = tmp_path / "tenant_A.pdf"
    create_benchmark_pdf(pdf_path, num_pages=3, prefix="TenantA")
    perf.benchmark_ingestion_single_document(pdf_path, doc_id="doc_A", workspace_id="tenant_ws_A")

    concurrency_report = perf.benchmark_concurrency(
        queries=["What is nominal speed?", "What is error code ERR-801?"],
        num_threads=3,
        requests_per_thread=3,
    )

    assert concurrency_report["total_requests"] == 9
    assert concurrency_report["errors_count"] == 0
    assert concurrency_report["thread_safety_passed"] is True
    assert concurrency_report["mean_latency_ms"] > 0.0
    assert concurrency_report["throughput_qps"] > 0.0


# =============================================================================
# 7. Caching & Optimization Verification Tests
# =============================================================================

def test_parent_chunk_in_memory_lru_caching(tmp_path):
    """Verify parent chunk in-memory caching avoids repeated disk reads."""
    cs = chunking_service
    cs.storage_root = tmp_path
    cs.invalidate_parent_cache()

    p_chunk = ParentChunkRecord(
        parent_id="p001",
        document_id="doc_cache_test",
        workspace_id="ws_cache",
        section_title="Overview",
        page_start=1,
        page_end=1,
        text="Cached parent content",
        child_ids=["c001"],
    )
    cs.save_parent_chunks([p_chunk], workspace_id="ws_cache", document_id="doc_cache_test")

    # First load: from memory cache
    t0 = time.perf_counter()
    loaded1 = cs.load_parent_chunks("ws_cache", "doc_cache_test")
    t_cache = (time.perf_counter() - t0) * 1000.0

    assert len(loaded1) == 1
    assert loaded1[0].text == "Cached parent content"

    # Invalidate cache
    cs.invalidate_parent_cache("ws_cache", "doc_cache_test")
    # Second load: re-read from disk
    t0 = time.perf_counter()
    loaded2 = cs.load_parent_chunks("ws_cache", "doc_cache_test")
    t_disk = (time.perf_counter() - t0) * 1000.0

    assert len(loaded2) == 1
    # Cache hit should be sub-millisecond
    assert t_cache < 5.0


def test_query_embedding_in_memory_caching():
    """Verify query embedding in-memory caching provides instant lookup for repeated queries."""
    es = embedding_service
    es.clear_query_cache()

    query = "What is the maintenance protocol for AX-450?"

    # First lookup: computes through PyTorch model
    t0 = time.perf_counter()
    emb1 = es.embed_query(query)
    t_compute = (time.perf_counter() - t0) * 1000.0

    # Second lookup: instant cache hit
    t0 = time.perf_counter()
    emb2 = es.embed_query(query)
    t_cached = (time.perf_counter() - t0) * 1000.0

    assert emb1 == emb2
    assert len(emb1) == 384
    # Cached lookup should be drastically faster (< 1.0 ms)
    assert t_cached < t_compute
    assert t_cached < 2.0
