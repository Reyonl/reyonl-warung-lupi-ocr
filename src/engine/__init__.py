"""
Engine interface and service module exports.
"""

from src.engine.engine_service import (
    EngineInterface,
    EngineConfig,
    EngineService,
)
from src.engines.tesseract_engine import TesseractEngine
from src.engines.easyocr_engine import EasyOCREngine

__all__ = [
    'EngineInterface',
    'EngineConfig',
    'EngineService',
    'TesseractEngine',
    'EasyOCREngine',
]
