"""
Warung Lupi OCR Engine package.

Public API:
    from src import (
        EngineService, TesseractEngine, EasyOCREngine, ClaudeVisionEngine,
        GeminiVisionEngine, InterpretationService, InterpretedResult,
        DraftTransactionService, DraftTransaction,
        OCRConfig, AuditTrail,
        WarungLupiAPIClient,
    )

Full pipeline:
    from src import EngineService, InterpretationService, DraftTransactionService, OCRConfig

    config = OCRConfig.load()
    # Phase 1: OCR
    service = EngineService()
    scan = service.process("image.jpg", engine=config.engine.preferred_engine)

    # Phase 2-4: Interpret + Match
    interpreter = InterpretationService()
    interpreted = interpreter.interpret(scan)
    interpreted = interpreter.enhance_with_products(interpreted, products)
    interpreted = interpreter.enhance_with_customers(interpreted, customers)

    # Phase 5: Draft
    draft_service = DraftTransactionService()
    draft = draft_service.create_draft_from_interpretation(interpreted)

    # Phase 9: API payload
    payload = draft_service.confirm_draft(draft)
    # client = WarungLupiAPIClient(config.api.base_url, config.api.api_token)
    # response = client.create_transaction(payload)
"""

from src.engine.engine_service import EngineService, EngineConfig, EngineInterface
from src.engines.tesseract_engine import TesseractEngine
from src.engines.easyocr_engine import EasyOCREngine
from src.engines.claude_vision_engine import ClaudeVisionEngine
from src.engines.gemini_vision_engine import GeminiVisionEngine
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
from src.config import OCRConfig, EngineConfig as OCRConfigEngineConfig, InterpreterConfig
from src.audit.trail import AuditTrail
from src.api.client import WarungLupiAPIClient, WarungLupiAPIError

__all__ = [
    "EngineService",
    "EngineConfig",
    "EngineInterface",
    "TesseractEngine",
    "EasyOCREngine",
    "ClaudeVisionEngine",
    "GeminiVisionEngine",
    "InterpretationService",
    "InterpretedResult",
    "InterpretedItem",
    "InterpretedCustomer",
    "DraftTransactionService",
    "DraftTransaction",
    "DraftTransactionItem",
    "OCRConfig",
    "InterpreterConfig",
    "AuditTrail",
    "WarungLupiAPIClient",
    "WarungLupiAPIError",
    "ScanResult",
    "Region",
    "RegionType",
    "Annotation",
    "Symbol",
    "LayoutInfo",
    "PreprocessingInfo",
]
