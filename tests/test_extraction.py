import json
from pathlib import Path
import shutil
import sys
import tempfile

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw
import pytest
import pymupdf

from backend.schemas.document import PageExtractionRecord
from backend.services.ocr_service import ocr_service
from backend.services.pdf_extractor import pdf_extractor


def create_digital_pdf(output_path: Path, pages_content: list[str]) -> None:
    """Helper to generate a clean digital text PDF with multiple pages."""
    doc = pymupdf.open()
    for content in pages_content:
        page = doc.new_page()
        page.insert_text((50, 72), content, fontsize=12)
    doc.save(str(output_path))
    doc.close()


def create_structured_pdf(output_path: Path) -> None:
    """Helper to generate a PDF with clear headings, paragraphs, and a drawn table."""
    doc = pymupdf.open()
    page = doc.new_page()

    # Title / Heading
    page.insert_text((50, 60), "1. Executive Overview", fontsize=18)
    page.insert_text((50, 90), "This document outlines standard operating procedures.", fontsize=11)

    # Subheading
    page.insert_text((50, 130), "2. Operating Parameters", fontsize=15)
    page.insert_text((50, 160), "Parameters are detailed in the operational matrix below.", fontsize=11)

    # Draw table
    rect = pymupdf.Rect(50, 180, 350, 260)
    page.draw_rect(rect)
    page.draw_line(pymupdf.Point(50, 210), pymupdf.Point(350, 210))
    page.draw_line(pymupdf.Point(200, 180), pymupdf.Point(200, 260))

    page.insert_text((60, 200), "Component")
    page.insert_text((210, 200), "Tolerance")
    page.insert_text((60, 240), "Pressure Valve")
    page.insert_text((210, 240), "+/- 5 bar")

    doc.save(str(output_path))
    doc.close()


def create_scanned_image_pdf(output_path: Path, image_text: str = "SCANNED EMERGENCY SHUTDOWN") -> None:
    """Helper to generate a scanned PDF containing an image with raster text."""
    # Create raster image using PIL
    img = Image.new("RGB", (600, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 80), image_text, fill=(0, 0, 0))

    img_temp = output_path.parent / f"tmp_img_{output_path.name}.png"
    img.save(img_temp)

    # Insert raster image into PDF page
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(pymupdf.Rect(50, 100, 550, 300), filename=str(img_temp))

    doc.save(str(output_path))
    doc.close()
    img_temp.unlink(missing_ok=True)


@pytest.fixture
def extraction_env():
    """Setup isolated storage directories for extraction testing."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    storage_root = temp_path / "storage"
    processed_root = storage_root / "processed"
    temp_work_root = storage_root / "temp"

    storage_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    temp_work_root.mkdir(parents=True, exist_ok=True)

    # Patch extractor paths
    pdf_extractor.storage_root = storage_root
    pdf_extractor.processed_root = processed_root
    pdf_extractor.temp_root = temp_work_root

    yield {
        "temp_path": temp_path,
        "storage_root": storage_root,
        "processed_root": processed_root,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_normal_digital_and_multipage_pdf_extraction(extraction_env):
    """Test 1, 2 & 3: Normal digital PDF extraction, multi-page parsing, and 1-indexed page preservation."""
    temp_path = extraction_env["temp_path"]
    pdf_file = temp_path / "three_page_manual.pdf"
    pages = [
        "First page content with technical instructions.",
        "Second page content with maintenance schedules.",
        "Third page content with regulatory compliance references.",
    ]
    create_digital_pdf(pdf_file, pages)

    result = pdf_extractor.extract_document(
        pdf_path=pdf_file,
        document_id="doc_test_101",
        workspace_id="ws_test_01",
        original_filename="three_page_manual.pdf",
    )

    assert result.total_pages == 3
    assert result.pages_extracted == 3
    assert result.ocr_pages_count == 0

    # Verify JSONL output file exists
    jsonl_path = extraction_env["storage_root"] / result.relative_output_path
    assert jsonl_path.is_file()

    with open(jsonl_path, "r", encoding="utf-8") as f_in:
        lines = [json.loads(line) for line in f_in]

    assert len(lines) == 3
    for idx, record in enumerate(lines, start=1):
        assert record["page_number"] == idx
        assert record["document_id"] == "doc_test_101"
        assert record["extraction_method"] == "native_text"
        assert record["ocr_used"] is False
        assert pages[idx - 1] in record["text"]


def test_empty_page_handling(extraction_env):
    """Test 4: PDF containing an intentionally empty page."""
    temp_path = extraction_env["temp_path"]
    pdf_file = temp_path / "with_blank_page.pdf"

    doc = pymupdf.open()
    doc.new_page()  # Page 1: Blank
    p2 = doc.new_page()  # Page 2: Text
    p2.insert_text((50, 50), "Content on page two.", fontsize=12)
    doc.save(str(pdf_file))
    doc.close()

    result = pdf_extractor.extract_document(
        pdf_path=pdf_file,
        document_id="doc_blank_01",
        workspace_id="ws_test_01",
        original_filename="with_blank_page.pdf",
    )

    assert result.total_pages == 2
    jsonl_path = extraction_env["storage_root"] / result.relative_output_path
    with open(jsonl_path, "r", encoding="utf-8") as f_in:
        lines = [json.loads(line) for line in f_in]

    assert lines[0]["page_number"] == 1
    assert lines[0]["extraction_method"] == "empty"
    assert lines[0]["text"] == ""

    assert lines[1]["page_number"] == 2
    assert lines[1]["extraction_method"] == "native_text"
    assert "Content on page two." in lines[1]["text"]


def test_corrupted_pdf_handling(extraction_env):
    """Test 5: Corrupted PDF file raises informative ValueError."""
    temp_path = extraction_env["temp_path"]
    corrupt_file = temp_path / "broken.pdf"
    corrupt_file.write_bytes(b"%PDF-1.4\nCorrupted garbage data that is not a valid PDF")

    with pytest.raises(ValueError, match="Corrupted or invalid PDF"):
        pdf_extractor.extract_document(
            pdf_path=corrupt_file,
            document_id="doc_corrupt_01",
            workspace_id="ws_test_01",
            original_filename="broken.pdf",
        )


def test_encrypted_pdf_handling(extraction_env):
    """Test 6: Encrypted / password-protected PDF raises ValueError."""
    temp_path = extraction_env["temp_path"]
    enc_file = temp_path / "protected.pdf"

    doc = pymupdf.open()
    doc.new_page().insert_text((50, 50), "Confidential Secret")
    # Save with user and owner passwords
    doc.save(
        str(enc_file),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="OwnerPass123",
        user_pw="UserPass123",
    )
    doc.close()

    with pytest.raises(ValueError, match="encrypted or password-protected"):
        pdf_extractor.extract_document(
            pdf_path=enc_file,
            document_id="doc_enc_01",
            workspace_id="ws_test_01",
            original_filename="protected.pdf",
        )


def test_heading_and_table_detection(extraction_env):
    """Test 7 & 8: Structure extraction detecting headings and tables."""
    temp_path = extraction_env["temp_path"]
    pdf_file = temp_path / "structured_doc.pdf"
    create_structured_pdf(pdf_file)

    result = pdf_extractor.extract_document(
        pdf_path=pdf_file,
        document_id="doc_struct_01",
        workspace_id="ws_test_01",
        original_filename="structured_doc.pdf",
    )

    assert result.total_pages == 1
    assert result.headings_count >= 1
    assert result.tables_count >= 1

    jsonl_path = extraction_env["storage_root"] / result.relative_output_path
    with open(jsonl_path, "r", encoding="utf-8") as f_in:
        page = json.loads(f_in.readline())

    # Check headings captured
    assert any("Executive Overview" in h for h in page["headings"])

    # Check table structure captured
    assert len(page["tables"]) >= 1
    tbl = page["tables"][0]
    assert tbl["col_count"] >= 2
    assert tbl["row_count"] >= 2


def test_scanned_page_detection_and_ocr(extraction_env):
    """Test 9 & 10: Scanned image PDF triggers selective OCR when Tesseract is available."""
    if not ocr_service.is_available():
        pytest.skip("Tesseract is not available on this machine; skipping live OCR test.")

    temp_path = extraction_env["temp_path"]
    pdf_file = temp_path / "scanned_doc.pdf"
    create_scanned_image_pdf(pdf_file, image_text="EMERGENCY SHUTDOWN PROCEDURE")

    result = pdf_extractor.extract_document(
        pdf_path=pdf_file,
        document_id="doc_scanned_01",
        workspace_id="ws_test_01",
        original_filename="scanned_doc.pdf",
    )

    assert result.total_pages == 1
    assert result.ocr_pages_count == 1

    jsonl_path = extraction_env["storage_root"] / result.relative_output_path
    with open(jsonl_path, "r", encoding="utf-8") as f_in:
        page = json.loads(f_in.readline())

    assert page["extraction_method"] == "ocr"
    assert page["ocr_used"] is True
    assert "EMERGENCY" in page["text"].upper() or "SHUTDOWN" in page["text"].upper()


def test_graceful_missing_ocr_behavior(extraction_env, monkeypatch):
    """Test 11: When OCR is unavailable, scanned pages record 'scanned_no_ocr' without failing."""
    # Force OCR availability to False
    monkeypatch.setattr(ocr_service, "is_available", lambda: False)

    temp_path = extraction_env["temp_path"]
    pdf_file = temp_path / "scanned_offline_ocr.pdf"
    create_scanned_image_pdf(pdf_file, image_text="SYSTEM HEALTH WARNING")

    result = pdf_extractor.extract_document(
        pdf_path=pdf_file,
        document_id="doc_scanned_offline_01",
        workspace_id="ws_test_01",
        original_filename="scanned_offline_ocr.pdf",
    )

    assert result.total_pages == 1
    assert result.ocr_pages_count == 0

    jsonl_path = extraction_env["storage_root"] / result.relative_output_path
    with open(jsonl_path, "r", encoding="utf-8") as f_in:
        page = json.loads(f_in.readline())

    assert page["extraction_method"] == "scanned_no_ocr"
    assert page["ocr_used"] is False
