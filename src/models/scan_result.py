"""
Phase 1: Image Understanding - Data Models

Output schema for the OCR engine. This models the structured result
from raw OCR processing, before interpretation/matching (Phase 2-4).

Key design principles:
- Every region has a confidence score (0.0 - 1.0)
- Text is never modified or cleaned — raw OCR output preserved
- Structural information (struck-through, symbols, position) is kept
- Regions are spatial — bbox in original image coordinates
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
from enum import Enum
import json


class RegionType(Enum):
    """Type classification for OCR regions (filled in Phase 2)."""
    UNKNOWN = "unknown"
    DATE = "date"
    CUSTOMER_NAME = "customer_name"
    PRODUCT_NAME = "product_name"
    QUANTITY = "quantity"
    UNIT_PRICE = "unit_price"
    SUBTOTAL = "subtotal"
    TOTAL = "total"
    DISCOUNT = "discount"
    NOTE = "note"


@dataclass
class Symbol:
    """A structural symbol detected near text (e.g., currency marker, separator)."""
    text: str  # The symbol character(s), e.g. "@", "Rp", "x"
    bbox: List[List[int]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    confidence: float = 1.0


@dataclass
class Annotation:
    """
    Additional structural info for a region.
    Populated by Vision LLM or preprocessing analysis, not basic OCR.
    """
    has_circle: bool = False  # Detected circle/oval around the text (often qty)
    circle_bbox: Optional[List[List[int]]] = None
    has_arrow: bool = False
    arrow_direction: Optional[str] = None  # "up", "down", "left", "right", "cross"
    has_strikethrough: bool = False
    strikethrough_start: Optional[List[int]] = None  # [x, y]
    strikethrough_end: Optional[List[int]] = None  # [x, y]
    symbols: List[Symbol] = field(default_factory=list)


@dataclass
class Region:
    """A detected text region from OCR."""
    text: str
    bbox: List[List[int]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] in image coordinates
    confidence: float  # 0.0 - 1.0
    region_type: RegionType = RegionType.UNKNOWN
    annotation: Annotation = field(default_factory=Annotation)
    engine: str = "unknown"  # which OCR engine produced this
    # For debugging: the raw symbol-level breakdown if available
    symbol_boxes: Optional[List[dict]] = None


@dataclass
class LayoutInfo:
    """Layout analysis of the page."""
    page_width: int
    page_height: int
    orientation: str = "portrait"  # or "landscape"
    columns: int = 1
    writing_direction: str = "ltr"  # left-to-right
    line_height_px: int = 0
    margin_left: int = 0
    margin_right: int = 0
    margin_top: int = 0
    margin_bottom: int = 0


@dataclass
class PreprocessingInfo:
    """Record of preprocessing operations applied."""
    operations: List[str] = field(default_factory=list)
    original_dimensions: Tuple[int, int] = (0, 0)
    processed_dimensions: Tuple[int, int] = (0, 0)


@dataclass
class ScanResult:
    """
    Complete Phase 1 result.
    This is what the OCR engine service returns.
    Phase 2 (Interpretation) consumes this.
    """
    status: str = "pending"  # "ok", "error", "uncertain", "no_text_detected"
    model: str = "unknown"  # which engine/version was used
    confidence: float = 0.0  # overall average confidence
    raw_text: str = ""
    regions: List[Region] = field(default_factory=list)
    layout: Optional[LayoutInfo] = None
    preprocessing: PreprocessingInfo = field(default_factory=PreprocessingInfo)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "status": self.status,
            "model": self.model,
            "confidence": round(self.confidence, 4),
            "raw_text": self.raw_text,
            "regions": [
                {
                    "text": r.text,
                    "bbox": r.bbox,
                    "confidence": round(r.confidence, 4),
                    "type": r.region_type.value,
                    "engine": r.engine,
                    "annotation": {
                        "has_circle": r.annotation.has_circle,
                        "circle_bbox": r.annotation.circle_bbox,
                        "has_arrow": r.annotation.has_arrow,
                        "arrow_direction": r.annotation.arrow_direction,
                        "has_strikethrough": r.annotation.has_strikethrough,
                        "strikethrough_start": r.annotation.strikethrough_start,
                        "strikethrough_end": r.annotation.strikethrough_end,
                        "symbols": [{"text": s.text, "bbox": s.bbox, "confidence": s.confidence} for s in r.annotation.symbols],
                    },
                    "symbol_boxes": r.symbol_boxes,
                }
                for r in self.regions
            ],
            "layout": {
                "page_width": self.layout.page_width if self.layout else 0,
                "page_height": self.layout.page_height if self.layout else 0,
                "orientation": self.layout.orientation if self.layout else "portrait",
                "columns": self.layout.columns if self.layout else 1,
                "writing_direction": self.layout.writing_direction if self.layout else "ltr",
                "line_height_px": self.layout.line_height_px if self.layout else 0,
                "margins": {
                    "left": self.layout.margin_left if self.layout else 0,
                    "right": self.layout.margin_right if self.layout else 0,
                    "top": self.layout.margin_top if self.layout else 0,
                    "bottom": self.layout.margin_bottom if self.layout else 0,
                } if self.layout else {},
            },
            "preprocessing": {
                "operations": self.preprocessing.operations,
                "original_dimensions": list(self.preprocessing.original_dimensions),
                "processed_dimensions": list(self.preprocessing.processed_dimensions),
            },
            "warnings": self.warnings,
        }
