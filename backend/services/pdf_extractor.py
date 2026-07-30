import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
import uuid

import pymupdf

from backend.schemas.document import ExtractedTable, PageExtractionRecord
from backend.services.ocr_service import ocr_service
from config.settings import get_settings

logger = logging.getLogger("ekn.extractor")


class PDFExtractionResult:
    def __init__(
        self,
        total_pages: int,
        pages_extracted: int,
        ocr_pages_count: int,
        headings_count: int,
        tables_count: int,
        relative_output_path: str,
    ):
        self.total_pages = total_pages
        self.pages_extracted = pages_extracted
        self.ocr_pages_count = ocr_pages_count
        self.headings_count = headings_count
        self.tables_count = tables_count
        self.relative_output_path = relative_output_path


class PDFExtractor:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.storage_root = Path(self.settings.storage_local_root).resolve()
        self.processed_root = Path(self.settings.processed_dir).resolve()
        self.temp_root = Path(self.settings.temp_upload_dir).resolve()

        self.processed_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def extract_headings_from_page(self, page: pymupdf.Page) -> List[str]:
        """Detect section titles and headings using font size and weight layout heuristics."""
        page_dict = page.get_text("dict")
        lines_info = []

        for block in page_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    line_text = "".join(span.get("text", "") for span in spans).strip()
                    if not line_text:
                        continue

                    max_size = max(span.get("size", 10.0) for span in spans) if spans else 10.0
                    is_bold = any(
                        bool(span.get("flags", 0) & (1 << 4) or "bold" in span.get("font", "").lower())
                        for span in spans
                    )
                    lines_info.append({"text": line_text, "size": max_size, "bold": is_bold})

        if not lines_info:
            return []

        # Find median font size for body text baseline
        all_sizes = sorted([item["size"] for item in lines_info])
        median_size = all_sizes[len(all_sizes) // 2] if all_sizes else 10.0

        headings = []
        for item in lines_info:
            text = item["text"]
            # Exclude long text paragraphs or trivial strings
            if len(text) < 3 or len(text) > 130:
                continue
            # Exclude lines ending with standard mid-sentence punctuation
            if text.endswith((".", ",", ";")):
                continue

            # Heading condition: notably larger than median or bold with at least median size
            is_larger = item["size"] >= median_size * 1.2
            is_prominent_bold = item["bold"] and (item["size"] >= median_size * 1.05)

            if is_larger or is_prominent_bold:
                headings.append(text)

        return headings

    def extract_tables_from_page(self, page: pymupdf.Page, page_number: int) -> List[Dict[str, Any]]:
        """Extract structured tables using PyMuPDF's table finder."""
        tables_data = []
        try:
            # use_layout=False avoids ONNX runtime segfaults on complex vector drawings in PyMuPDF 2.1
            tab_finder = page.find_tables(use_layout=False)
            if tab_finder and tab_finder.tables:
                for idx, tab in enumerate(tab_finder.tables, start=1):
                    raw_rows = tab.extract()
                    if not raw_rows:
                        continue

                    # Clean empty cell values to empty string
                    clean_rows = [[cell if cell is not None else "" for cell in row] for row in raw_rows]
                    headers = [str(h) for h in tab.header.names] if hasattr(tab.header, "names") and tab.header.names else []

                    # If headers are empty, use first row if reasonable
                    if not headers and clean_rows:
                        headers = [str(c) for c in clean_rows[0]]

                    table_record = ExtractedTable(
                        table_id=f"p{page_number}_tbl_{idx}",
                        page_number=page_number,
                        row_count=len(clean_rows),
                        col_count=len(clean_rows[0]) if clean_rows else 0,
                        headers=headers,
                        rows=clean_rows,
                    )
                    tables_data.append(table_record.model_dump())
        except Exception as exc:
            logger.warning("Table extraction error on page %d: %s", page_number, exc)

        return tables_data

    def extract_document(
        self,
        pdf_path: Path,
        document_id: str,
        workspace_id: str,
        original_filename: str,
    ) -> PDFExtractionResult:
        """Extract all pages from a PDF document incrementally into structured JSON Lines."""
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        try:
            with open(pdf_path, "rb") as f_in:
                file_bytes = f_in.read()
        except Exception as read_err:
            raise ValueError(f"Unable to read PDF file from disk: {read_err}") from read_err

        # 1. Open document safely from memory stream
        doc = None
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except pymupdf.FileDataError as exc:
            raise ValueError(f"Corrupted or invalid PDF file: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Unable to open PDF document: {exc}") from exc

        # 2. Check for encryption / password protection
        if doc.is_encrypted and doc.authenticate("") == 0:
            doc.close()
            raise ValueError("PDF document is encrypted or password-protected.")

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise ValueError("PDF document has 0 pages.")

        # Setup temporary output file for incremental streaming writes
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.processed_root.mkdir(parents=True, exist_ok=True)
        temp_out_id = f"extract_{uuid.uuid4().hex}.jsonl"
        temp_out_path = self.temp_root / temp_out_id

        ocr_pages_count = 0
        headings_count = 0
        tables_count = 0
        pages_extracted = 0

        try:
            with open(temp_out_path, "w", encoding="utf-8") as f_out:
                for page_idx in range(total_pages):
                    page_number = page_idx + 1
                    try:
                        page = doc[page_idx]

                        # A. Native text extraction
                        native_text = page.get_text("text").strip()
                        char_count = len(native_text)

                        # B. Structure & Heading detection
                        headings = self.extract_headings_from_page(page)
                        headings_count += len(headings)

                        # C. Table extraction
                        tables = self.extract_tables_from_page(page, page_number)
                        tables_count += len(tables)

                        # D. Scanned page detection & Selective OCR
                        images = page.get_images()
                        has_images = bool(images)
                        ocr_used = False

                        if char_count < self.settings.ocr_min_text_chars_per_page:
                            # Deficient digital text
                            if has_images and ocr_service.is_available():
                                success, ocr_text, err = ocr_service.run_ocr(page)
                                if success and ocr_text:
                                    final_text = ocr_text
                                    method = "ocr"
                                    ocr_used = True
                                    ocr_pages_count += 1
                                else:
                                    final_text = native_text
                                    method = "ocr_failed" if err else "native_text"
                            elif has_images:
                                final_text = native_text
                                method = "scanned_no_ocr"
                            elif char_count == 0:
                                final_text = ""
                                method = "empty"
                            else:
                                final_text = native_text
                                method = "native_text"
                        else:
                            final_text = native_text
                            method = "native_text"

                        # Build structured page record
                        record = PageExtractionRecord(
                            document_id=document_id,
                            filename=original_filename,
                            page_number=page_number,
                            text=final_text,
                            headings=headings,
                            tables=tables,
                            extraction_method=method,
                            ocr_used=ocr_used,
                        )

                        # Write JSONL line incrementally
                        f_out.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
                        f_out.flush()
                        pages_extracted += 1

                    except Exception as page_err:
                        logger.error("Error extracting page %d of %s: %s", page_number, original_filename, page_err)
                        # Fallback record for problematic page so the worker continues
                        fallback_record = PageExtractionRecord(
                            document_id=document_id,
                            filename=original_filename,
                            page_number=page_number,
                            text="",
                            headings=[],
                            tables=[],
                            extraction_method="extraction_error",
                            ocr_used=False,
                        )
                        f_out.write(json.dumps(fallback_record.model_dump(), ensure_ascii=False) + "\n")
                        f_out.flush()
                        pages_extracted += 1

            # 3. Finalize output: move from temp to permanent processed storage
            dest_dir = self.processed_root / workspace_id / document_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            final_jsonl_path = dest_dir / "extracted_pages.jsonl"

            shutil.move(str(temp_out_path), str(final_jsonl_path))

            # Store relative path from storage_root
            rel_output_path = str(final_jsonl_path.relative_to(self.storage_root)).replace("\\", "/")

            logger.info(
                "Document %s extracted successfully: %d pages (%d OCR, %d headings, %d tables) -> %s",
                original_filename,
                pages_extracted,
                ocr_pages_count,
                headings_count,
                tables_count,
                rel_output_path,
            )

            return PDFExtractionResult(
                total_pages=total_pages,
                pages_extracted=pages_extracted,
                ocr_pages_count=ocr_pages_count,
                headings_count=headings_count,
                tables_count=tables_count,
                relative_output_path=rel_output_path,
            )

        finally:
            if doc is not None:
                doc.close()
            # Clean up temp file if still present
            if temp_out_path.exists():
                try:
                    temp_out_path.unlink()
                except OSError:
                    pass

    def extract(
        self,
        pdf_path: Path,
        document_id: str = "doc_01",
        workspace_id: str = "eval_ws",
    ) -> List[PageExtractionRecord]:
        """Convenience method to extract and directly return a list of PageExtractionRecord objects."""
        filename = pdf_path.name
        res = self.extract_document(
            pdf_path=pdf_path,
            document_id=document_id,
            workspace_id=workspace_id,
            original_filename=filename,
        )
        jsonl_path = (self.storage_root / res.relative_output_path).resolve()
        records: List[PageExtractionRecord] = []
        if jsonl_path.is_file():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        records.append(PageExtractionRecord(**json.loads(line_str)))
        return records


pdf_extractor = PDFExtractor()
