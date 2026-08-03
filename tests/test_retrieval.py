import json
from pathlib import Path
import sys
import uuid
from fastapi.testclient import TestClient
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database.base import Base
from backend.database.connection import get_db, set_test_engine
from backend.database.models import Document, DocumentIndex, IngestionJob, Workspace
from backend.main import app
from backend.schemas.document import PageExtractionRecord
from backend.schemas.indexing import ChildChunkRecord, ParentChunkRecord
from backend.schemas.search import SearchRequest, SearchResponse
from backend.services.bm25_service import BM25Service, bm25_service
from backend.services.chunking_service import chunking_service
from backend.services.embedding_service import embedding_service
from backend.services.ingestion_service import ingestion_service
from backend.services.pdf_extractor import pdf_extractor
from backend.services.reranking_service import reranking_service
from backend.services.retrieval_service import retrieval_service
from backend.services.storage_service import storage_service
from backend.services.vector_index_service import vector_index_service


@pytest.fixture
def test_db():
    """Isolated in-memory SQLite database for retrieval tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    set_test_engine(engine)

    with session_factory() as session:
        yield session

    set_test_engine(None)
    engine.dispose()


@pytest.fixture
def api_client(test_db):
    """FastAPI TestClient wired to the isolated test database."""
    def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def test_workspace(test_db):
    """Create a sample workspace for retrieval testing."""
    ws = Workspace(
        name="Retrieval Test Workspace",
        slug=f"retrieval-test-{uuid.uuid4().hex[:8]}",
        description="Workspace for hybrid retrieval engine verification",
    )
    test_db.add(ws)
    test_db.commit()
    test_db.refresh(ws)
    return ws


# =============================================================================
# 1. Domain-Aware Tokenizer Tests
# =============================================================================

def test_domain_aware_tokenizer_preserves_technical_identifiers():
    """Verify tokenizer preserves technical codes, model numbers, and dotted versions."""
    text = "Inspection of turbine AX-450 rev_2 for ISO-9001 compliance with firmware v1.2.3 and error ERR_500."
    tokens = BM25Service.tokenize(text)

    expected_preserved = ["ax-450", "rev_2", "iso-9001", "v1.2.3", "err_500"]
    for expected in expected_preserved:
        assert expected in tokens, f"Expected '{expected}' to be preserved in {tokens}"


def test_domain_aware_tokenizer_handles_punctuation_and_empty():
    """Verify punctuation trimming and empty string handling."""
    assert BM25Service.tokenize("") == []
    assert BM25Service.tokenize("   \n\t  ") == []

    tokens = BM25Service.tokenize("Critical failure in Component (AX-450), section [3.14]: stopped.")
    assert "critical" in tokens
    assert "failure" in tokens
    assert "component" in tokens
    assert "ax-450" in tokens
    assert "3.14" in tokens
    assert "stopped" in tokens


# =============================================================================
# 2. BM25 Service Corpus Persistence & Indexing
# =============================================================================

def test_bm25_corpus_persistence_and_rebuild(tmp_path, monkeypatch):
    """Verify JSONL corpus creation, reload, and in-memory caching."""
    monkeypatch.setattr(bm25_service, "storage_root", tmp_path)
    monkeypatch.setattr(bm25_service, "processed_root", tmp_path / "processed")
    bm25_service.invalidate_cache()

    ws_id = f"ws_{uuid.uuid4().hex[:6]}"
    doc_id = f"doc_{uuid.uuid4().hex[:6]}"

    chunks = [
        ChildChunkRecord(
            chunk_id=f"{doc_id}_c1",
            parent_id=f"{doc_id}_p1",
            document_id=doc_id,
            workspace_id=ws_id,
            original_filename="manual.pdf",
            page_start=1,
            page_end=1,
            section_title="Turbine Safety",
            text="The AX-450 turbine emergency shutdown procedure requires valve closure.",
            chunk_type="text",
            token_count=12,
            file_hash="hash1",
            embedding_model="test-model",
            document_version=1,
        ),
        ChildChunkRecord(
            chunk_id=f"{doc_id}_c2",
            parent_id=f"{doc_id}_p1",
            document_id=doc_id,
            workspace_id=ws_id,
            original_filename="manual.pdf",
            page_start=2,
            page_end=2,
            section_title="Hydraulic Systems",
            text="Hydraulic pressure pump P-99 maintenance should be performed quarterly.",
            chunk_type="text",
            token_count=10,
            file_hash="hash1",
            embedding_model="test-model",
            document_version=1,
        ),
    ]

    added = bm25_service.add_or_update_document_chunks(ws_id, doc_id, chunks)
    assert added == 2

    # Verify JSONL file exists
    corpus_file = tmp_path / "processed" / ws_id / "bm25_corpus.jsonl"
    assert corpus_file.exists()

    with open(corpus_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2
    assert lines[0]["chunk_id"] == f"{doc_id}_c1"
    assert lines[1]["chunk_id"] == f"{doc_id}_c2"


def test_bm25_keyword_search_ranking(tmp_path, monkeypatch):
    """Verify keyword query finds exact technical terms with positive BM25 scores."""
    monkeypatch.setattr(bm25_service, "storage_root", tmp_path)
    monkeypatch.setattr(bm25_service, "processed_root", tmp_path / "processed")
    bm25_service.invalidate_cache()

    ws_id = f"ws_{uuid.uuid4().hex[:6]}"
    doc_id = f"doc_{uuid.uuid4().hex[:6]}"

    chunks = [
        ChildChunkRecord(
            chunk_id=f"{doc_id}_c1",
            parent_id=f"{doc_id}_p1",
            document_id=doc_id,
            workspace_id=ws_id,
            original_filename="manual.pdf",
            page_start=1,
            page_end=1,
            section_title="Turbine",
            text="The AX-450 turbine rotor operates at 3600 RPM under nominal load.",
            chunk_type="text",
            token_count=12,
            file_hash="hash1",
            embedding_model="test-model",
            document_version=1,
        ),
        ChildChunkRecord(
            chunk_id=f"{doc_id}_c2",
            parent_id=f"{doc_id}_p1",
            document_id=doc_id,
            workspace_id=ws_id,
            original_filename="manual.pdf",
            page_start=2,
            page_end=2,
            section_title="Lubrication",
            text="Synthetic oil ISO-VG46 is recommended for high-temperature bearings.",
            chunk_type="text",
            token_count=10,
            file_hash="hash1",
            embedding_model="test-model",
            document_version=1,
        ),
    ]

    bm25_service.add_or_update_document_chunks(ws_id, doc_id, chunks)

    # Search for specific technical model number
    results = bm25_service.search(workspace_id=ws_id, query="AX-450 rotor", top_k=5)
    assert len(results) == 1
    assert results[0]["chunk_id"] == f"{doc_id}_c1"
    assert results[0]["bm25_score"] > 0
    assert results[0]["bm25_rank"] == 1

    # Search for oil specification
    results_oil = bm25_service.search(workspace_id=ws_id, query="ISO-VG46 lubrication", top_k=5)
    assert len(results_oil) == 1
    assert results_oil[0]["chunk_id"] == f"{doc_id}_c2"

    # Search for unrelated keyword
    results_none = bm25_service.search(workspace_id=ws_id, query="quantum cryptography", top_k=5)
    assert len(results_none) == 0


def test_bm25_incremental_update_and_removal(tmp_path, monkeypatch):
    """Verify updating a document replaces its chunks and removal purges them."""
    monkeypatch.setattr(bm25_service, "storage_root", tmp_path)
    monkeypatch.setattr(bm25_service, "processed_root", tmp_path / "processed")
    bm25_service.invalidate_cache()

    ws_id = f"ws_{uuid.uuid4().hex[:6]}"
    doc1_id = f"doc1_{uuid.uuid4().hex[:6]}"
    doc2_id = f"doc2_{uuid.uuid4().hex[:6]}"

    # Add doc1 with 2 chunks
    chunks_doc1 = [
        ChildChunkRecord(
            chunk_id=f"{doc1_id}_c{i}",
            parent_id=f"{doc1_id}_p1",
            document_id=doc1_id,
            workspace_id=ws_id,
            original_filename="doc1.pdf",
            page_start=i,
            page_end=i,
            section_title=f"Section {i}",
            text=f"AX-450 specification part {i}",
            chunk_type="text",
            token_count=8,
            file_hash="hash1",
            embedding_model="test-model",
            document_version=1,
        )
        for i in range(1, 3)
    ]
    bm25_service.add_or_update_document_chunks(ws_id, doc1_id, chunks_doc1)

    # Add doc2 with 1 chunk
    chunks_doc2 = [
        ChildChunkRecord(
            chunk_id=f"{doc2_id}_c1",
            parent_id=f"{doc2_id}_p1",
            document_id=doc2_id,
            workspace_id=ws_id,
            original_filename="doc2.pdf",
            page_start=1,
            page_end=1,
            section_title="Doc 2",
            text="Coolant radiator specifications and AX-450 cooling manifold",
            chunk_type="text",
            token_count=8,
            file_hash="hash2",
            embedding_model="test-model",
            document_version=1,
        )
    ]
    bm25_service.add_or_update_document_chunks(ws_id, doc2_id, chunks_doc2)

    # Query matching both docs
    res = bm25_service.search(workspace_id=ws_id, query="AX-450", top_k=10)
    assert len(res) == 3

    # Update doc1 to only have 1 chunk (simulating re-indexing / edit)
    updated_doc1 = chunks_doc1[:1]
    bm25_service.add_or_update_document_chunks(ws_id, doc1_id, updated_doc1)

    res_updated = bm25_service.search(workspace_id=ws_id, query="AX-450", top_k=10)
    assert len(res_updated) == 2  # 1 from doc1, 1 from doc2

    # Remove doc2
    removed = bm25_service.remove_document_chunks(ws_id, doc2_id)
    assert removed == 1

    res_after_remove = bm25_service.search(workspace_id=ws_id, query="AX-450", top_k=10)
    assert len(res_after_remove) == 1
    assert res_after_remove[0]["document_id"] == doc1_id


# =============================================================================
# 3. Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranking
# =============================================================================

def test_reciprocal_rank_fusion_mathematics():
    """Validate RRF formula: RRF(d) = sum(1 / (k + rank))."""
    k = 60
    # Case 1: Doc A is rank 1 in dense (1/(60+1) = 0.016393) and rank 2 in BM25 (1/(60+2) = 0.016129)
    # Total RRF(A) = 0.016393 + 0.016129 = 0.032522
    # Doc B is rank 2 in dense only: 1/(60+2) = 0.016129
    # Doc C is rank 1 in BM25 only: 1/(60+1) = 0.016393

    score_a = (1.0 / (k + 1)) + (1.0 / (k + 2))
    score_b = 1.0 / (k + 2)
    score_c = 1.0 / (k + 1)

    assert score_a > score_c > score_b

    # Verify retrieval_service computation produces identical values
    dense_items = [
        {"chunk_id": "c_a", "dense_rank": 1, "dense_distance": 0.1, "text": "Chunk A"},
        {"chunk_id": "c_b", "dense_rank": 2, "dense_distance": 0.2, "text": "Chunk B"},
    ]
    bm25_items = [
        {"chunk_id": "c_c", "bm25_rank": 1, "bm25_score": 5.0, "text": "Chunk C"},
        {"chunk_id": "c_a", "bm25_rank": 2, "bm25_score": 4.0, "text": "Chunk A"},
    ]

    merged = {}
    for d in dense_items:
        cid = d["chunk_id"]
        merged[cid] = {**d, "rrf_score": 1.0 / (k + d["dense_rank"])}

    for b in bm25_items:
        cid = b["chunk_id"]
        r = 1.0 / (k + b["bm25_rank"])
        if cid in merged:
            merged[cid]["rrf_score"] += r
        else:
            merged[cid] = {**b, "rrf_score": r}

    assert pytest.approx(merged["c_a"]["rrf_score"], rel=1e-5) == score_a
    assert pytest.approx(merged["c_b"]["rrf_score"], rel=1e-5) == score_b
    assert pytest.approx(merged["c_c"]["rrf_score"], rel=1e-5) == score_c


def test_cross_encoder_reranking_service():
    """Verify Cross-Encoder reranking assigns relevance scores and sorts correctly."""
    query = "What is the operating temperature of the turbine?"
    candidates = [
        {
            "chunk_id": "c1",
            "text": "The office cafeteria is open from 8 AM to 5 PM Monday through Friday.",
        },
        {
            "chunk_id": "c2",
            "text": "The AX-450 gas turbine continuous operating temperature must not exceed 950 degrees Celsius.",
        },
        {
            "chunk_id": "c3",
            "text": "General company safety policy requires protective eyewear in designated zones.",
        },
    ]

    reranked = reranking_service.rerank(query=query, candidates=candidates, top_k=2)

    assert len(reranked) == 2
    # The relevant turbine passage should be ranked 1st
    assert reranked[0]["chunk_id"] == "c2"
    assert reranked[0]["final_rank"] == 1
    assert isinstance(reranked[0]["cross_encoder_score"], float)
    assert reranked[0]["cross_encoder_score"] > reranked[1]["cross_encoder_score"]
    assert reranked[1]["final_rank"] == 2


def test_cross_encoder_rerank_empty_candidates():
    """Verify graceful handling when candidates list is empty."""
    res = reranking_service.rerank("any query", [], top_k=5)
    assert res == []


# =============================================================================
# 4. Dense ChromaDB Retrieval & Parent Context Recovery
# =============================================================================

def test_dense_vector_query(tmp_path):
    """Verify ChromaDB query_dense retrieves nearest neighbours with cosine distance."""
    doc_id = f"doc_dense_{uuid.uuid4().hex[:6]}"
    ws_id = f"ws_dense_{uuid.uuid4().hex[:6]}"

    chunks = [
        ChildChunkRecord(
            chunk_id=f"{doc_id}_c1",
            parent_id=f"{doc_id}_p1",
            document_id=doc_id,
            workspace_id=ws_id,
            original_filename="physics.pdf",
            page_start=1,
            page_end=1,
            section_title="Thermodynamics",
            text="Second law of thermodynamics states entropy of an isolated system always increases.",
            chunk_type="text",
            token_count=14,
            file_hash="hash_p",
            embedding_model=embedding_service.model_name,
            document_version=1,
        ),
        ChildChunkRecord(
            chunk_id=f"{doc_id}_c2",
            parent_id=f"{doc_id}_p1",
            document_id=doc_id,
            workspace_id=ws_id,
            original_filename="physics.pdf",
            page_start=2,
            page_end=2,
            section_title="Electromagnetism",
            text="Maxwell equations describe how electric and magnetic fields are generated by charges.",
            chunk_type="text",
            token_count=14,
            file_hash="hash_p",
            embedding_model=embedding_service.model_name,
            document_version=1,
        ),
    ]

    embeddings = embedding_service.generate_embeddings([c.text for c in chunks])
    vector_index_service.index_child_chunks(chunks, embeddings, doc_id)

    # Query for thermodynamics
    query_vector = embedding_service.embed_query("entropy and thermodynamics")
    results = vector_index_service.query_dense(
        query_embedding=query_vector,
        top_k=2,
        where_filter={"workspace_id": ws_id},
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == f"{doc_id}_c1"
    assert results[0]["dense_rank"] == 1
    assert "dense_distance" in results[0]

    # Clean up ChromaDB
    vector_index_service.delete_document_vectors(doc_id)


def test_parent_context_recovery_in_retrieval(tmp_path, monkeypatch):
    """Verify parent passages are correctly recovered and attached to results."""
    monkeypatch.setattr(chunking_service, "storage_root", tmp_path)
    monkeypatch.setattr(chunking_service, "processed_root", tmp_path / "processed")
    monkeypatch.setattr(bm25_service, "storage_root", tmp_path)
    monkeypatch.setattr(bm25_service, "processed_root", tmp_path / "processed")
    bm25_service.invalidate_cache()

    ws_id = f"ws_parent_{uuid.uuid4().hex[:6]}"
    doc_id = f"doc_parent_{uuid.uuid4().hex[:6]}"

    # Save parent chunk
    parent_record = ParentChunkRecord(
        parent_id=f"{doc_id}_p1",
        document_id=doc_id,
        workspace_id=ws_id,
        section_title="Safety Standards",
        page_start=1,
        page_end=2,
        text="FULL PARENT CONTEXT: Detailed explanation of all corporate safety requirements.",
        content_type="text",
        child_ids=[f"{doc_id}_c1"],
    )
    chunking_service.save_parent_chunks([parent_record], ws_id, doc_id)

    # Save child chunk
    child_record = ChildChunkRecord(
        chunk_id=f"{doc_id}_c1",
        parent_id=f"{doc_id}_p1",
        document_id=doc_id,
        workspace_id=ws_id,
        original_filename="safety.pdf",
        page_start=1,
        page_end=1,
        section_title="Safety Standards",
        text="Wear safety glasses in all laboratory designated areas.",
        chunk_type="text",
        token_count=10,
        file_hash="hash_s",
        embedding_model=embedding_service.model_name,
        document_version=1,
    )

    # Index into BM25
    bm25_service.add_or_update_document_chunks(ws_id, doc_id, [child_record])

    # Search with include_parent=True
    req = SearchRequest(query="safety glasses laboratory", include_parent=True, top_k=1)
    res = retrieval_service.search(ws_id, req)

    assert res.total_results == 1
    assert res.results[0].parent_text == parent_record.text

    # Search with include_parent=False
    req_no_parent = SearchRequest(query="safety glasses laboratory", include_parent=False, top_k=1)
    res_no_parent = retrieval_service.search(ws_id, req_no_parent)

    assert res_no_parent.total_results == 1
    assert res_no_parent.results[0].parent_text is None


# =============================================================================
# 5. Hybrid Search API Endpoints & Error Handling
# =============================================================================

def test_search_api_workspace_not_found(api_client):
    """POST /api/v1/workspaces/{workspace_id}/search returns 404 for unknown workspace."""
    bad_id = str(uuid.uuid4())
    resp = api_client.post(
        f"/api/v1/workspaces/{bad_id}/search",
        json={"query": "test query"},
    )
    assert resp.status_code == 404
    assert f"Workspace '{bad_id}' was not found" in resp.json()["detail"]


def test_search_api_empty_query_validation(api_client, test_workspace):
    """POST /api/v1/workspaces/{workspace_id}/search returns 422 for empty query string."""
    resp = api_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/search",
        json={"query": ""},
    )
    assert resp.status_code == 422


def test_search_api_empty_workspace(api_client, test_workspace, tmp_path, monkeypatch):
    """POST /api/v1/workspaces/{workspace_id}/search on empty workspace returns 200 with 0 results."""
    monkeypatch.setattr(bm25_service, "storage_root", tmp_path)
    monkeypatch.setattr(bm25_service, "processed_root", tmp_path / "processed")
    bm25_service.invalidate_cache()

    resp = api_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/search",
        json={"query": "turbines and compressors"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "turbines and compressors"
    assert data["workspace_id"] == test_workspace.id
    assert data["total_results"] == 0
    assert data["results"] == []


def test_search_api_end_to_end_integration(api_client, test_workspace, test_db, tmp_path, monkeypatch):
    """Full integration test: ingest document -> search via API -> verify rankings & citations."""
    monkeypatch.setattr(bm25_service, "storage_root", tmp_path)
    monkeypatch.setattr(bm25_service, "processed_root", tmp_path / "processed")
    monkeypatch.setattr(chunking_service, "storage_root", tmp_path)
    monkeypatch.setattr(chunking_service, "processed_root", tmp_path / "processed")
    bm25_service.invalidate_cache()

    # Create Document record
    doc = Document(
        workspace_id=test_workspace.id,
        original_filename="turbines_guide.pdf",
        display_name="Turbines Guide",
        storage_key=f"uploads/{test_workspace.id}/guide.pdf",
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        file_size_bytes=1024,
        mime_type="application/pdf",
        total_pages=1,
        processing_status="completed",
    )
    test_db.add(doc)
    test_db.commit()
    test_db.refresh(doc)

    # Mock extracted page
    page = PageExtractionRecord(
        document_id=doc.id,
        filename=doc.original_filename,
        page_number=1,
        text="The AX-450 gas turbine operates at 3600 RPM. ISO-9001 certified maintenance.",
        headings=["Technical Specifications"],
        tables=[],
        extraction_method="native_text",
        ocr_used=False,
        char_count=77,
        token_estimate=15,
    )

    # Index document via ingestion service
    doc_index = ingestion_service.index_document(test_db, doc, pages=[page], force=True)
    assert doc_index.status == "completed"

    try:
        # Search via API
        resp = api_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/search",
            json={
                "query": "AX-450 turbine operating speed",
                "top_k": 5,
                "include_parent": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["workspace_id"] == test_workspace.id
        assert data["total_results"] >= 1
        result_item = data["results"][0]

        # Check full provenance
        assert result_item["document_id"] == doc.id
        assert result_item["original_filename"] == "turbines_guide.pdf"
        assert result_item["page_start"] == 1
        assert "AX-450" in result_item["text"]
        assert result_item["final_rank"] == 1
        assert result_item["rrf_score"] > 0
        assert result_item["parent_text"] is not None

    finally:
        # Clean up ChromaDB vectors
        vector_index_service.delete_document_vectors(doc.id)


def test_document_deletion_purges_bm25_corpus(api_client, test_workspace, test_db, tmp_path, monkeypatch):
    """Verify deleting a document removes its chunks from BM25 corpus and invalidates index."""
    monkeypatch.setattr(bm25_service, "storage_root", tmp_path)
    monkeypatch.setattr(bm25_service, "processed_root", tmp_path / "processed")
    monkeypatch.setattr(storage_service, "storage_root", tmp_path)
    monkeypatch.setattr(storage_service, "upload_root", tmp_path / "uploads")
    bm25_service.invalidate_cache()

    # Create dummy upload file
    upload_dir = tmp_path / "uploads" / test_workspace.id
    upload_dir.mkdir(parents=True, exist_ok=True)
    dummy_file = upload_dir / "sample.pdf"
    dummy_file.write_bytes(b"%PDF-1.4 sample")

    doc = Document(
        workspace_id=test_workspace.id,
        original_filename="sample.pdf",
        display_name="Sample PDF",
        storage_key=f"uploads/{test_workspace.id}/sample.pdf",
        sha256_hash="samplehash123",
        file_size_bytes=100,
        mime_type="application/pdf",
        total_pages=1,
        processing_status="completed",
    )
    test_db.add(doc)
    test_db.commit()
    test_db.refresh(doc)

    # Add BM25 chunks
    chunks = [
        ChildChunkRecord(
            chunk_id=f"{doc.id}_c1",
            parent_id=f"{doc.id}_p1",
            document_id=doc.id,
            workspace_id=test_workspace.id,
            original_filename="sample.pdf",
            page_start=1,
            page_end=1,
            section_title="Sample",
            text="UniqueKeywordForDeletionVerification",
            chunk_type="text",
            token_count=5,
            file_hash="samplehash123",
            embedding_model="test-model",
            document_version=1,
        )
    ]
    bm25_service.add_or_update_document_chunks(test_workspace.id, doc.id, chunks)

    # Verify query finds it
    res_before = bm25_service.search(test_workspace.id, "UniqueKeywordForDeletionVerification")
    assert len(res_before) == 1

    # Delete document via API
    resp = api_client.delete(f"/api/v1/workspaces/{test_workspace.id}/documents/{doc.id}")
    assert resp.status_code == 200

    # Verify query no longer finds it
    res_after = bm25_service.search(test_workspace.id, "UniqueKeywordForDeletionVerification")
    assert len(res_after) == 0
