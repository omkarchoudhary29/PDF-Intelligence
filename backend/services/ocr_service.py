import io
import logging
from typing import Optional, Tuple

from PIL import Image
import pymupdf
import pytesseract

from config.settings import get_settings

logger = logging.getLogger("ekn.ocr")


class OCRService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._ocr_checked = False
        self._ocr_available = False

    def is_available(self) -> bool:
        """Check whether OCR is enabled and Tesseract is accessible on the system."""
        if not self.settings.ocr_enabled:
            return False

        if self._ocr_checked:
            return self._ocr_available

        try:
            cmd = self.settings.tesseract_cmd or "tesseract"
            pytesseract.pytesseract.tesseract_cmd = cmd
            version = pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR initialized successfully (version: %s)", version)
            self._ocr_available = True
        except Exception as exc:
            logger.warning("Tesseract OCR is not available on this system: %s. Scanned pages will not be OCR-processed.", exc)
            self._ocr_available = False

        self._ocr_checked = True
        return self._ocr_available

    def run_ocr(self, page: pymupdf.Page, dpi: int = 150) -> Tuple[bool, str, Optional[str]]:
        """Render a page to image and perform OCR using Tesseract.

        Returns:
            (success, text, error_message)
        """
        if not self.is_available():
            return False, "", "OCR is disabled or Tesseract is not installed on this system."

        try:
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(image)
            clean_text = text.strip()
            return True, clean_text, None
        except Exception as exc:
            logger.warning("OCR failed on page %s: %s", getattr(page, "number", "unknown"), exc)
            return False, "", str(exc)


ocr_service = OCRService()
