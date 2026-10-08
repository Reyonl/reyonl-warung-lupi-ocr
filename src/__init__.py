"""
Warung Lupi OCR Engine package.

Public API:
    from src import EngineService, TesseractEngine, EasyOCREngine
    from src.interpreter import InterpretationService, InterpretedResult
    from src.models import ScanResult, Region
"""

from src.engine.engine_service import EngineService, EngineConfig, EngineInterface
from src.engines.tesseract_engine import TesseractEngine
from src.engines.easyocr_engine import EasyOCREngine
from src.interpreter.service import (
    InterpretationService,
    InterpretedResult,
    InterpretedItem,
    InterpretedCustomer,
)
from src.draft.service import (
    DraftTransactionService,
    DraftTransaction,
    DraftTransactionItem,
)
from src.models.scan_result import (
    ScanResult,
    Region,
    RegionType,
    Annotation,
    Symbol,
    LayoutInfo,
    PreprocessingInfo,
)

__all__ = [
    "EngineService",
    "EngineConfig",
    "EngineInterface",
    "TesseractEngine",
    "EasyOCREngine",
    "InterpretationService",
    "InterpretedResult",
    "InterpretedItem",
    "InterpretedCustomer",
    "ScanResult",
    "Region",
    "RegionType",
    "Annotation",
    "Symbol",
    "LayoutInfo",
    "PreprocessingInfo",
]
