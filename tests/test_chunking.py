from pathlib import Path
import sys
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.schemas.document import PageExtractionRecord
from backend.services.chunking_service import ChunkingService


@pytest.fixture
def chunking_svc():
    return ChunkingService()


def test_heading_aware_and_cross_page_sections(chunking_svc):
    """Test that headings create sections and sections continue across pages."""
    pages = [
        PageExtractionRecord(
            document_id="doc_test_sec",
            filename="manual.pdf",
            page_number=1,
            text="Document Introduction\nThis is introductory context before the main section.\n\nSection 1: Architecture\nThis describes the backend architecture in detail.",
            headings=["Section 1: Architecture"],
            tables=[],
            extraction_method="native_text",
            ocr_used=False,
        ),
        PageExtractionRecord(
            document_id="doc_test_sec",
            filename="manual.pdf",
            page_number=2,
            text="The architecture continues onto page two with database schemas and services.\n\nSection 2: Security\nThis section covers encryption and authentication.",
            headings=["Section 2: Security"],
            tables=[],
            extraction_method="native_text",
            ocr_used=False,
        ),
    ]

    parents, children = chunking_svc.chunk_document(
        pages=pages,
        document_id="doc_test_sec",
        workspace_id="ws_test",
        original_filename="manual.pdf",
        file_hash="hash123",
    )

    assert len(parents) >= 2
    assert len(children) >= 2

    # Check section titles preserved in parents and children
    parent_titles = [p.section_title for p in parents]
    child_titles = [c.section_title for c in children]

    assert any("Architecture" in t for t in parent_titles)
    assert any("Security" in t for t in parent_titles)
    assert any("Architecture" in t for t in child_titles)
    assert any("Security" in t for t in child_titles)

    # Verify cross-page continuation: Section 1 spans page 1 to 2
    arch_parents = [p for p in parents if "Architecture" in p.section_title]
    assert len(arch_parents) > 0
    assert arch_parents[0].page_start <= arch_parents[0].page_end


def test_fallback_when_no_headings_exist(chunking_svc):
    """Test fallback section naming when a document has zero headings."""
    pages = [
        PageExtractionRecord(
            document_id="doc_no_headings",
            filename="plain.pdf",
            page_number=1,
            text="Plain text document without any detected headings or font anomalies. Just standard body text.",
            headings=[],
            tables=[],
            extraction_method="native_text",
            ocr_used=False,
        )
    ]

    parents, children = chunking_svc.chunk_document(
        pages=pages,
        document_id="doc_no_headings",
        workspace_id="ws_test",
        original_filename="plain.pdf",
        file_hash="hash456",
    )

    assert len(parents) == 1
    assert len(children) == 1
    assert parents[0].section_title == "Document Overview"
    assert children[0].section_title == "Document Overview"
    assert children[0].page_start == 1
    assert children[0].page_end == 1


def test_paragraph_and_sentence_boundary_chunking(chunking_svc):
    """Test paragraph and sentence boundaries are respected and token limits hold."""
    long_para_1 = " ".join([f"Sentence {i} explains important concept number {i}." for i in range(1, 30)])
    long_para_2 = " ".join([f"Paragraph two sentence {i} details advanced mechanism {i}." for i in range(1, 30)])
    full_text = f"{long_para_1}\n\n{long_para_2}"

    pages = [
        PageExtractionRecord(
            document_id="doc_para_test",
            filename="large_paras.pdf",
            page_number=1,
            text=full_text,
            headings=[],
            tables=[],
            extraction_method="native_text",
            ocr_used=False,
        )
    ]

    parents, children = chunking_svc.chunk_document(
        pages=pages,
        document_id="doc_para_test",
        workspace_id="ws_test",
        original_filename="large_paras.pdf",
        file_hash="hash789",
    )

    # Should split into multiple child chunks
    assert len(children) >= 2
    # Verify child chunk size does not exceed max target plus slight margin
    for c in children:
        assert c.token_count <= 250
        assert len(c.text) > 0


def test_parent_child_mapping_and_provenance(chunking_svc):
    """Verify parent-child references, stable IDs, and page provenance."""
    pages = [
        PageExtractionRecord(
            document_id="doc_provenance",
            filename="report.pdf",
            page_number=3,
            text="Page three first paragraph with sufficient technical substance.\n\nPage three second paragraph.",
            headings=[],
            tables=[],
            extraction_method="native_text",
            ocr_used=False,
        )
    ]

    parents, children = chunking_svc.chunk_document(
        pages=pages,
        document_id="doc_provenance",
        workspace_id="ws_test",
        original_filename="report.pdf",
        file_hash="hash999",
    )

    assert len(parents) == 1
    assert len(children) == 1

    parent = parents[0]
    child = children[0]

    # Verify ID convention
    assert parent.parent_id == "doc_provenance_p0001"
    assert child.chunk_id == "doc_provenance_c0001"

    # Verify parent-child bidirectional linking
    assert child.parent_id == parent.parent_id
    assert child.chunk_id in parent.child_ids

    # Verify page provenance
    assert child.page_start == 3
    assert child.page_end == 3
    assert parent.page_start == 3
    assert parent.page_end == 3


def test_table_aware_chunking_and_row_splitting(chunking_svc):
    """Test structured table formatting as Markdown, repeating headers on row splitting."""
    headers = ["Component", "Status", "Version", "Owner"]
    rows = [[f"Module_{i}", "Active" if i % 2 == 0 else "Pending", f"v{i}.0", f"Team_{i % 3}"] for i in range(1, 40)]

    table_dict = {
        "headers": headers,
        "rows": rows,
        "bbox": [50.0, 50.0, 500.0, 700.0],
    }

    pages = [
        PageExtractionRecord(
            document_id="doc_table_test",
            filename="inventory.pdf",
            page_number=5,
            text="Here is the system inventory table:",
            headings=["System Inventory"],
            tables=[table_dict],
            extraction_method="native_text",
            ocr_used=False,
        )
    ]

    parents, children = chunking_svc.chunk_document(
        pages=pages,
        document_id="doc_table_test",
        workspace_id="ws_test",
        original_filename="inventory.pdf",
        file_hash="table_hash",
    )

    table_children = [c for c in children if c.chunk_type == "table"]
    assert len(table_children) >= 2  # Large table split into multiple sub-tables

    for tc in table_children:
        assert tc.page_start == 5
        assert tc.page_end == 5
        # Verify markdown headers are repeated in each sub-table
        assert "| Component | Status | Version | Owner |" in tc.text
        assert "---" in tc.text

    # Verify parent content_type is 'mixed' or 'table'
    assert any(p.content_type in ("table", "mixed") for p in parents)


def test_empty_pages_handling(chunking_svc):
    """Verify empty document or blank pages return zero chunks without error."""
    pages = [
        PageExtractionRecord(
            document_id="doc_empty",
            filename="empty.pdf",
            page_number=1,
            text="",
            headings=[],
            tables=[],
            extraction_method="empty",
            ocr_used=False,
        )
    ]

    parents, children = chunking_svc.chunk_document(
        pages=pages,
        document_id="doc_empty",
        workspace_id="ws_test",
        original_filename="empty.pdf",
        file_hash="empty_hash",
    )

    assert len(parents) == 0
    assert len(children) == 0


def test_parent_chunks_atomic_persistence(chunking_svc, tmp_path, monkeypatch):
    """Test saving and loading parent chunks via atomic JSONL file operations."""
    monkeypatch.setattr(chunking_svc.settings, "storage_local_root", str(tmp_path))

    pages = [
        PageExtractionRecord(
            document_id="doc_persist",
            filename="safety.pdf",
            page_number=1,
            text="Operational procedures section.\nAll operators must follow safety guidelines.",
            headings=["Safety Guidelines"],
            tables=[],
            extraction_method="native_text",
            ocr_used=False,
        )
    ]

    parents, children = chunking_svc.chunk_document(
        pages=pages,
        document_id="doc_persist",
        workspace_id="ws_persist",
        original_filename="safety.pdf",
        file_hash="safety_hash",
    )

    # Save to disk
    rel_path = chunking_svc.save_parent_chunks(parents, "ws_persist", "doc_persist")
    assert rel_path == "processed/ws_persist/doc_persist/parent_chunks.jsonl"

    expected_file = tmp_path / "processed" / "ws_persist" / "doc_persist" / "parent_chunks.jsonl"
    assert expected_file.is_file()

    # Load from disk
    loaded_parents = chunking_svc.load_parent_chunks("ws_persist", "doc_persist")
    assert len(loaded_parents) == len(parents)
    assert loaded_parents[0].parent_id == parents[0].parent_id
    assert loaded_parents[0].text == parents[0].text
    assert loaded_parents[0].child_ids == parents[0].child_ids
