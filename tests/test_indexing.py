from datetime import datetime
import json
from pathlib import Path
import sys
import uuid
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database.base import Base
from backend.database.connection import get_chroma_client, set_test_engine
from backend.database.models import Document, DocumentIndex, IngestionJob, Workspace
from backend.schemas.indexing import ChildChunkRecord
from backend.services.chunking_service import chunking_service
from backend.services.embedding_service import embedding_service
from backend.services.ingestion_service import ingestion_service
from backend.services.pdf_extractor import pdf_extractor
from backend.services.storage_service import storage_service
from backend.services.vector_index_service import vector_index_service
from backend.worker import backfill_unindexed_documents


@pytest.fixture
def test_db():
    """Isolated in-memory SQLite database for indexing tests."""
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
def test_workspace(test_db):
    """Create a sample workspace for tests."""
    ws = Workspace(
        name="Vector Test Workspace",
        slug=f"vec-test-{uuid.uuid4().hex[:8]}",
        description="Workspace for vector indexing verification",
    )
    test_db.add(ws)
    test_db.commit()
    test_db.refresh(ws)
    return ws


def test_embedding_generation_dimension_and_normalization():
    """Verify embedding dimensions and cosine normalization (L2 norm = 1.0)."""
    texts = [
        "Enterprise document intelligence platform.",
        "Hierarchical chunking preserves parent context and page numbers.",
    ]

    embeddings = embedding_service.generate_embeddings(texts, batch_size=2)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384
    assert embedding_service.dimension == 384

    # Verify L2 normalization: sqrt(sum(v^2)) == 1.0 within float tolerance
    norm_0 = np.linalg.norm(np.array(embeddings[0]))
    norm_1 = np.linalg.norm(np.array(embeddings[1]))
    assert pytest.approx(norm_0, rel=1e-3) == 1.0
    assert pytest.approx(norm_1, rel=1e-3) == 1.0


def test_chromadb_collection_and_cosine_distance():
    """Verify collection initialization and cosine space metadata."""
    collection = vector_index_service.get_collection()
    assert collection is not None
    assert collection.name == vector_index_service.collection_name
    assert collection.metadata.get("hnsw:space") == "cosine"


def test_batch_vector_upsert_and_metadata_persistence():
    """Test batch indexing into ChromaDB and retrieval of stored metadata."""
    doc_id = f"doc_upsert_{uuid.uuid4().hex[:8]}"
    workspace_id = "ws_test"

    chunks = [
        ChildChunkRecord(
            chunk_id=f"{doc_id}_c0001",
            parent_id=f"{doc_id}_p0001",
            document_id=doc_id,
            workspace_id=workspace_id,
            original_filename="manual.pdf",
            page_start=1,
            page_end=1,
            section_title="Introduction",
            text="First child chunk discussing system foundations.",
            chunk_type="text",
            token_count=10,
            file_hash="hash_upsert_1",
            embedding_model=embedding_service.model_name,
            document_version=1,
        ),
        ChildChunkRecord(
            chunk_id=f"{doc_id}_c0002",
            parent_id=f"{doc_id}_p0001",
            document_id=doc_id,
            workspace_id=workspace_id,
            original_filename="manual.pdf",
            page_start=1,
            page_end=1,
            section_title="Introduction",
            text="Second child chunk describing vector indexing pipelines.",
            chunk_type="text",
            token_count=12,
            file_hash="hash_upsert_1",
            embedding_model=embedding_service.model_name,
            document_version=1,
        ),
    ]

    embeddings = embedding_service.generate_embeddings([c.text for c in chunks])
    indexed_count = vector_index_service.index_child_chunks(chunks, embeddings, doc_id)

    assert indexed_count == 2
    assert vector_index_service.get_document_vector_count(doc_id) == 2

    # Query collection to verify metadata fields
    col = vector_index_service.get_collection()
    res = col.get(ids=[f"{doc_id}_c0001"])
    assert len(res["ids"]) == 1
    meta = res["metadatas"][0]
    assert meta["document_id"] == doc_id
    assert meta["section_title"] == "Introduction"
    assert meta["chunk_type"] == "text"
    assert meta["page_start"] == 1

    # Clean up
    vector_index_service.delete_document_vectors(doc_id)
    assert vector_index_service.get_document_vector_count(doc_id) == 0


def test_idempotent_reindexing_and_obsolete_vector_cleanup():
    """Verify that re-indexing does not duplicate vectors and cleans up obsolete chunk IDs."""
    doc_id = f"doc_idempotent_{uuid.uuid4().hex[:8]}"
    workspace_id = "ws_test"

    # Initial indexing: 3 chunks
    chunks_v1 = [
        ChildChunkRecord(
            chunk_id=f"{doc_id}_c{i:04d}",
            parent_id=f"{doc_id}_p0001",
            document_id=doc_id,
            workspace_id=workspace_id,
            original_filename="guide.pdf",
            page_start=1,
            page_end=1,
            section_title="Overview",
            text=f"Version 1 text content chunk {i}",
            chunk_type="text",
            token_count=10,
            file_hash="hash_v1",
            embedding_model=embedding_service.model_name,
            document_version=1,
        )
        for i in range(1, 4)
    ]

    emb_v1 = embedding_service.generate_embeddings([c.text for c in chunks_v1])
    vector_index_service.index_child_chunks(chunks_v1, emb_v1, doc_id)
    assert vector_index_service.get_document_vector_count(doc_id) == 3

    # Re-indexing with same chunks (idempotent upsert): vector count remains 3
    vector_index_service.index_child_chunks(chunks_v1, emb_v1, doc_id)
    assert vector_index_service.get_document_vector_count(doc_id) == 3

    # Re-indexing with fewer chunks (e.g. document edited): obsolete chunks should be pruned
    chunks_v2 = chunks_v1[:1]  # Only 1 chunk now
    emb_v2 = embedding_service.generate_embeddings([c.text for c in chunks_v2])
    vector_index_service.index_child_chunks(chunks_v2, emb_v2, doc_id)

    assert vector_index_service.get_document_vector_count(doc_id) == 1

    # Clean up
    vector_index_service.delete_document_vectors(doc_id)


def test_end_to_end_ingestion_and_indexing_pipeline(test_db, test_workspace, tmp_path, monkeypatch):
    """Test full pipeline execution: PDF extraction -> chunking -> vector indexing -> search ready."""
    import pymupdf

    monkeypatch.setattr(ingestion_service.settings, "storage_local_root", str(tmp_path))
    monkeypatch.setattr(storage_service, "storage_root", tmp_path)
    monkeypatch.setattr(storage_service, "upload_root", tmp_path / "uploads")
    monkeypatch.setattr(storage_service, "temp_root", tmp_path / "temp")
    monkeypatch.setattr(pdf_extractor, "storage_root", tmp_path)
    monkeypatch.setattr(pdf_extractor, "processed_root", tmp_path / "processed")
    monkeypatch.setattr(pdf_extractor, "temp_root", tmp_path / "temp")
    monkeypatch.setattr(ingestion_service, "storage_root", tmp_path)
    monkeypatch.setattr(chunking_service.settings, "storage_local_root", str(tmp_path))

    # Generate test PDF
    pdf_bytes = b"%PDF-1.4 sample pdf content for end to end verification"
    doc_pdf = pymupdf.open()
    page1 = doc_pdf.new_page(width=595, height=842)
    page1.insert_text((50, 50), "Overview Section\nEKN provides deep PDF extraction and vector search.", fontsize=12)
    page2 = doc_pdf.new_page(width=595, height=842)
    page2.insert_text((50, 50), "Specifications\nAll embeddings are generated locally on CPU.", fontsize=12)
    pdf_data = doc_pdf.tobytes()
    doc_pdf.close()

    # Store file
    uploads_dir = tmp_path / "uploads" / test_workspace.id
    uploads_dir.mkdir(parents=True, exist_ok=True)
    pdf_file = uploads_dir / "test_e2e.pdf"
    with open(pdf_file, "wb") as f:
        f.write(pdf_data)

    doc = Document(
        workspace_id=test_workspace.id,
        original_filename="test_e2e.pdf",
        display_name="test_e2e.pdf",
        storage_key=f"uploads/{test_workspace.id}/test_e2e.pdf",
        sha256_hash="e2e_sha256_hash",
        file_size_bytes=len(pdf_data),
        mime_type="application/pdf",
        processing_status="queued",
    )
    test_db.add(doc)
    test_db.commit()
    test_db.refresh(doc)

    job = IngestionJob(document_id=doc.id, status="queued")
    test_db.add(job)
    test_db.commit()
    test_db.refresh(job)

    # Claim and execute job
    claimed = ingestion_service.claim_next_job(test_db)
    assert claimed.id == job.id

    success = ingestion_service.execute_job(test_db, claimed)
    assert success is True

    test_db.refresh(doc)
    test_db.refresh(job)

    assert doc.processing_status == "completed"
    assert doc.total_pages == 2
    assert job.status == "completed"

    # Verify DocumentIndex record
    doc_index = test_db.query(DocumentIndex).filter(DocumentIndex.document_id == doc.id).first()
    assert doc_index is not None
    assert doc_index.status == "completed"
    assert doc_index.total_parents >= 1
    assert doc_index.total_children >= 1
    assert doc_index.embedding_dimension == 384
    assert doc_index.indexed_file_hash == doc.sha256_hash

    # Verify Search Readiness status
    status_resp = ingestion_service.get_document_index_status(test_db, test_workspace.id, doc.id)
    assert status_resp.is_search_ready is True
    assert status_resp.index.status == "completed"

    # Clean up ChromaDB vectors
    vector_index_service.delete_document_vectors(doc.id)


def test_incremental_indexing_skips_unchanged_file(test_db, test_workspace, tmp_path, monkeypatch):
    """Test that incremental indexing skips regeneration when hash and model match, unless forced."""
    monkeypatch.setattr(ingestion_service.settings, "storage_local_root", str(tmp_path))

    doc = Document(
        workspace_id=test_workspace.id,
        original_filename="doc_inc.pdf",
        display_name="doc_inc.pdf",
        storage_key=f"uploads/{test_workspace.id}/doc_inc.pdf",
        sha256_hash="hash_inc_123",
        file_size_bytes=1000,
        mime_type="application/pdf",
        processing_status="completed",
        total_pages=1,
    )
    test_db.add(doc)
    test_db.commit()
    test_db.refresh(doc)

    # Pre-populate completed index
    doc_index = DocumentIndex(
        document_id=doc.id,
        status="completed",
        total_parents=2,
        total_children=5,
        embedding_model=ingestion_service.settings.embedding_model,
        embedding_dimension=384,
        indexed_file_hash=doc.sha256_hash,
        completed_at=datetime.now(),
    )
    test_db.add(doc_index)
    test_db.commit()

    # Re-index without force: should return existing index without error
    result = ingestion_service.index_document(test_db, doc, pages=[], force=False)
    assert result.total_parents == 2
    assert result.total_children == 5
    assert result.status == "completed"


def test_backfill_unindexed_documents(test_db, test_workspace, tmp_path, monkeypatch):
    """Test backfilling existing Phase 3 extracted documents missing vector indexes."""
    from backend.schemas.document import PageExtractionRecord

    monkeypatch.setattr(ingestion_service.settings, "storage_local_root", str(tmp_path))
    monkeypatch.setattr(ingestion_service, "storage_root", tmp_path)
    monkeypatch.setattr(chunking_service.settings, "storage_local_root", str(tmp_path))

    doc = Document(
        workspace_id=test_workspace.id,
        original_filename="backfill_target.pdf",
        display_name="backfill_target.pdf",
        storage_key=f"uploads/{test_workspace.id}/backfill_target.pdf",
        sha256_hash="backfill_sha",
        file_size_bytes=2000,
        mime_type="application/pdf",
        processing_status="completed",
        total_pages=1,
        extracted_pages_path=f"processed/{test_workspace.id}/bf_doc/extracted_pages.jsonl",
    )
    test_db.add(doc)
    test_db.commit()
    test_db.refresh(doc)

    # Write extracted_pages.jsonl
    pages_dir = tmp_path / "processed" / test_workspace.id / "bf_doc"
    pages_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = pages_dir / "extracted_pages.jsonl"

    page_rec = PageExtractionRecord(
        document_id=doc.id,
        filename=doc.original_filename,
        page_number=1,
        text="Backfill test document content explaining operations.",
        headings=["Operations"],
        tables=[],
        extraction_method="native_text",
        ocr_used=False,
    )
    with open(jsonl_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(page_rec.model_dump()) + "\n")

    # Run backfill
    backfilled_count = backfill_unindexed_documents()
    assert backfilled_count == 1

    test_db.refresh(doc)
    doc_index = test_db.query(DocumentIndex).filter(DocumentIndex.document_id == doc.id).first()
    assert doc_index is not None
    assert doc_index.status == "completed"
    assert doc_index.total_children >= 1

    # Clean up vectors
    vector_index_service.delete_document_vectors(doc.id)
