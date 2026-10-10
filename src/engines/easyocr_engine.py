"""
EasyOCR Engine (local, offline, free).

Uses EasyOCR's neural network for OCR, which tends to be more accurate
for handwriting than Tesseract. Bundles its own models and does not
depend on external Tesseract installation.

EasyOCR supports both English and Indonesian models out of the box.
"""

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    EASYOCR_AVAILABLE = False

import cv2
import numpy as np
import logging
from typing import List

from src.engine.engine_service import EngineInterface, EngineConfig
from src.models.scan_result import (
    ScanResult, Region, RegionType, Annotation,
    Symbol, LayoutInfo
)

logger = logging.getLogger(__name__)


class EasyOCREngine(EngineInterface):
    """EasyOCR engine wrapper."""

    _reader = None  # Singleton reader

    def __init__(self, languages: List[str] = None):
        if languages is None:
            languages = ['en', 'id']
        self.languages = languages

    def _get_reader(self):
        """Lazily initialize EasyOCR reader (loads models on first use)."""
        if self._reader is None:
            logger.info(f"Initializing EasyOCR reader for languages: {self.languages}")
            self._reader = easyocr.Reader(
                self.languages,
                gpu=False,
                verbose=False,
                # Use larger minimum size for better handling of small text
                # in low-resolution scans
            )
        return self._reader

    def name(self) -> str:
        return "easyocr-1.7"

    def is_available(self) -> bool:
        if not EASYOCR_AVAILABLE:
            return False
        try:
            self._get_reader()
            return True
        except Exception as e:
            logger.warning(f"EasyOCR not available: {e}")
            return False

    def process(self, image_path: str, config: EngineConfig) -> ScanResult:
        """Process image with EasyOCR."""
        reader = self._get_reader()

        # Load and preprocess
        img = cv2.imread(image_path)
        if img is None:
            return ScanResult(status="error", warnings=[f"Cannot read image: {image_path}"])

        original_h, original_w = img.shape[:2]

        # EasyOCR benefits from larger images for handwriting
        new_w = 1600
        new_h = int(original_h * new_w / original_w)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Scale factor for bbox mapping
        scale_x = original_w / new_w
        scale_y = original_h / new_h

        # Run OCR with detail=1 for bounding boxes
        result = reader.readtext(
            gray,
            detail=1,
            paragraph=False,
            min_size=5,
            width_ths=0.5,
            low_text=0.4,
        )

        if not result:
            return ScanResult(
                status="no_text_detected",
                model=self.name(),
                warnings=["EasyOCR detected no text — image may be blank"],
            )

        # Convert results to our Region format
        regions = []
        confs = []
        full_lines = []

        for item in result:
            bbox_pts = item[0]  # List of [x, y] tuples (4 points)
            text = item[1]
            conf = float(item[2])
            confs.append(conf)

            # Map bbox back to original coordinates
            orig_bbox = []
            for pt in bbox_pts:
                x = int(pt[0] * scale_x)
                y = int(pt[1] * scale_y)
                orig_bbox.append([x, y])

            # Detect if the region might be struck-through
            # (check if there's a line intersecting the bbox area)
            annotation = Annotation()

            # EasyOCR sometimes detects symbols like @, +, = — mark these
            if any(c in text for c in ['@', '=', '×', 'x', '+', '★', '→']) and len(text) <= 3:
                annotation.symbols.append(Symbol(
                    text=text,
                    bbox=orig_bbox,
                    confidence=conf,
                ))

            # Check for currency markers
            if any(c in text.lower() for c in ['rp', 'k', 'rb', 'ribu']):
                annotation.symbols.append(Symbol(
                    text=text,
                    bbox=orig_bbox,
                    confidence=conf,
                ))

            region = Region(
                text=text.strip(),
                bbox=orig_bbox,
                confidence=conf,
                region_type=RegionType.UNKNOWN,
                annotation=annotation,
                engine=self.name(),
            )
            regions.append(region)

            # Track lines for raw text reconstruction
            full_lines.append(text)

        avg_conf = float(np.mean(confs)) if confs else 0.0

        # Sort regions by position (top-to-bottom, left-to-right)
        regions.sort(key=lambda r: (
            np.mean([r.bbox[0][1], r.bbox[2][1]]),  # y center
            np.mean([r.bbox[0][0], r.bbox[2][0]])   # x center
        ))

        # Reconstruct raw text (line by line approximation)
        raw_text = '\n'.join(full_lines)

        result_obj = ScanResult(
            status="ok",
            model=self.name(),
            confidence=round(avg_conf, 4),
            raw_text=raw_text,
            regions=regions,
            warnings=[],
        )

        logger.info(
            f"EasyOCR result: conf={avg_conf:.3f}, regions={len(regions)}"
        )

        return result_obj


# Also alias for compatibility
class EasyOCRNeuralEngine(EasyOCREngine):
    """Explicit alias for the neural-based EasyOCR engine."""
    pass
