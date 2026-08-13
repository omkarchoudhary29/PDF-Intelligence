"""Master benchmark execution script for Phase 11.

Executes:
1. Full retrieval benchmark comparing Configurations A, B, C, D on the 65-question dataset.
2. Cross-Encoder reranking impact & candidate pool trade-off analysis.
3. Answer quality & citation validation evaluation.
4. Document ingestion profiling & query latency breakdown (with P95).
5. Concurrency & tenant isolation stress test.

Outputs full results to `evaluation/benchmark_results.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.answer_benchmark import AnswerBenchmark
from evaluation.dataset_builder import generate_evaluation_dataset
from evaluation.performance_benchmark import PerformanceBenchmark, create_benchmark_pdf
from evaluation.retrieval_benchmark import RetrievalBenchmark


def run_full_suite() -> None:
    print("=" * 70)
    print("STARTING ENTERPRISE KNOWLEDGE NAVIGATOR (EKN) - PHASE 11 BENCHMARKS")
    print("=" * 70)

    dataset_path = Path("evaluation/dataset.jsonl")
    if not dataset_path.exists():
        data = generate_evaluation_dataset()
        with open(dataset_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(data)} evaluation questions.")

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        workspace_id = f"eval_run_ws_{int(time.time())}"

        # -----------------------------------------------------------------
        # 1. Retrieval Benchmark (Configurations A, B, C, D)
        # -----------------------------------------------------------------
        print("\n--- 1. RETRIEVAL ARCHITECTURE BENCHMARK ---")
        ret_bench = RetrievalBenchmark(workspace_id=workspace_id)
        ret_bench.bm25_service.storage_root = temp_dir
        ret_bench.chunking_service.storage_root = temp_dir

        corpus_info = ret_bench.setup_evaluation_corpus(temp_dir)
        print(f"Indexed corpus: {corpus_info['documents_indexed']} documents, {corpus_info['total_pages']} pages, {corpus_info['total_chunks']} chunks.")

        retrieval_report = ret_bench.run_benchmark(dataset=data, top_k=10)

        print("\n[Retrieval Metrics Summary]")
        for cfg, stats in retrieval_report["configurations"].items():
            print(f"  {cfg:25s} | Recall@5: {stats['recall@5']:.4f} | Recall@10: {stats['recall@10']:.4f} | Precision@5: {stats['precision@5']:.4f} | MRR: {stats['mrr']:.4f} | NDCG@10: {stats['ndcg@10']:.4f} | Mean Lat: {stats['latency_mean_ms']:.1f}ms | P95 Lat: {stats['latency_p95_ms']:.1f}ms")

        print("\n[Cross-Encoder Reranking Impact]")
        ce = retrieval_report["cross_encoder_impact"]
        print(f"  MRR Before Reranking (RRF):  {ce['mean_mrr_before_reranking_rrf']:.4f}")
        print(f"  MRR After Cross-Encoder:      {ce['mean_mrr_after_cross_encoder']:.4f}")
        print(f"  MRR Improvement:              +{ce['mrr_improvement_percent']}%")
        print(f"  Reranking Overhead (Mean):    {ce['mean_rerank_overhead_ms']:.1f}ms (P95: {ce['p95_rerank_overhead_ms']:.1f}ms)")

        print("\n[Candidate Pool Sensitivity Trade-off (10 vs 20 vs 30)]")
        pool_res = ret_bench.benchmark_candidate_pool_sizes(data, [10, 20, 30])
        for p_size, p_stats in pool_res.items():
            print(f"  Pool Size {p_size:2d} -> MRR: {p_stats['mrr']:.4f} | NDCG@10: {p_stats['ndcg@10']:.4f} | Latency: {p_stats['mean_latency_ms']:.1f}ms (P95: {p_stats['p95_latency_ms']:.1f}ms)")

        # -----------------------------------------------------------------
        # 2. Answer & Citation Quality Benchmark
        # -----------------------------------------------------------------
        print("\n--- 2. ANSWER QUALITY & CITATION ACCURACY BENCHMARK ---")
        ans_bench = AnswerBenchmark(workspace_id=workspace_id)
        
        # Ground-truth verified answers baseline evaluation
        ground_truth_answers = {
            r["id"]: (r["expected_answer"] + " [E1]" if not r["is_unanswerable"] else "There is insufficient evidence in the provided documents.")
            for r in data
        }
        ans_report = ans_bench.run_answer_benchmark(dataset=data, mock_answers=ground_truth_answers)

        am = ans_report["answerable_metrics"]
        print("  [Verified Ground-Truth Baseline Metrics]")
        print(f"  Faithfulness:        {am['faithfulness']:.4f}")
        print(f"  Answer Relevance:    {am['answer_relevance']:.4f}")
        print(f"  Context Relevance:   {am['context_relevance']:.4f}")
        print(f"  Citation Precision:  {am['citation_precision']:.4f}")
        print(f"  Citation Coverage:   {am['citation_coverage']:.4f}")

        um = ans_report["unanswerable_metrics"]
        print("\n  [Unanswerable Question Reliability]")
        print(f"  Detection Accuracy:       {um['unanswerable_accuracy']:.4f}")
        print(f"  Detection Recall:         {um['unanswerable_recall']:.4f}")
        print(f"  Unsupported Answer Rate:  {um['unsupported_answer_rate']:.4f}")

        # -----------------------------------------------------------------
        # 3. Document Ingestion & Query Latency Benchmark
        # -----------------------------------------------------------------
        print("\n--- 3. PERFORMANCE & LATENCY PROFILING ---")
        perf_bench = PerformanceBenchmark(workspace_id=workspace_id)
        perf_bench.bm25_service.storage_root = temp_dir
        perf_bench.chunking_service.storage_root = temp_dir

        # Ingestion profiling across sizes
        ingest_scaling = perf_bench.benchmark_ingestion_scaling(temp_dir)
        print("\n[Ingestion Scaling Across Page Sizes]")
        for tier, im in ingest_scaling.items():
            print(f"  {tier:7s} ({im['pages_processed']} pages): Extract={im['extraction_time_ms']:.1f}ms | Chunk={im['chunking_time_ms']:.1f}ms | Embed={im['embedding_generation_time_ms']:.1f}ms | Chroma={im['chroma_indexing_time_ms']:.1f}ms | BM25={im['bm25_indexing_time_ms']:.1f}ms | Total={im['total_ingestion_time_ms']:.1f}ms | Throughput={im['throughput_pages_per_sec']:.1f} pages/s | Peak RAM={im['peak_memory_mb']:.1f}MB")

        # Query latency breakdown
        sample_queries = [
            "What is the nominal operating speed of the AX-450 steam turbine?",
            "What does error code ERR-902 indicate?",
            "What is the maximum exposure ceiling for sulfur dioxide?",
            "How do the operating speeds of AX-450 and GT-200 compare?",
        ]
        query_breakdown = perf_bench.benchmark_query_latency_breakdown(sample_queries, iterations=4)
        print("\n[Query Latency Breakdown (ms)]")
        for stage, stats in query_breakdown.items():
            print(f"  {stage:26s} | Mean: {stats['mean_ms']:6.2f}ms | Median: {stats['median_ms']:6.2f}ms | P95: {stats['p95_ms']:6.2f}ms")

        # Concurrency & Reliability
        print("\n[Concurrency & Multi-Tenant Isolation Stress Test]")
        conc_report = perf_bench.benchmark_concurrency(sample_queries, num_threads=4, requests_per_thread=5)
        print(f"  Total Requests:        {conc_report['total_requests']}")
        print(f"  Throughput:            {conc_report['throughput_qps']} QPS")
        print(f"  Mean Latency:          {conc_report['mean_latency_ms']:.1f}ms (P95: {conc_report['p95_latency_ms']:.1f}ms)")
        print(f"  Errors / Data Leaks:   {conc_report['errors_count']}")
        print(f"  Thread Safety Passed:  {conc_report['thread_safety_passed']}")

        # Save all results to benchmark_results.json
        full_results = {
            "retrieval_benchmark": retrieval_report,
            "candidate_pool_tradeoff": pool_res,
            "answer_and_citations": {
                "answerable_metrics": am,
                "unanswerable_metrics": um,
            },
            "ingestion_scaling": ingest_scaling,
            "query_latency_breakdown": query_breakdown,
            "concurrency_stress_test": conc_report,
        }

        output_file = Path("evaluation/benchmark_results.json")
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(full_results, f_out, indent=2)
        print(f"\nAll benchmark results serialized to {output_file}")


if __name__ == "__main__":
    run_full_suite()
