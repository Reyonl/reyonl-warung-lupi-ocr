"""
OCR Engine Service — Orchestrates multiple OCR engines and selects the best result.

Design:
- EngineInterface defines the contract
- Each engine (Tesseract, EasyOCR, Claude) implements independently
- EngineService picks the engine with highest confidence for the image
- All results are normalized to ScanResult (Phase 1 schema)

Usage:
    service = EngineService()
    result = service.process(image_path, engine="tesseract")  # or "auto"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple
import os
import logging
import cv2
import numpy as np

from src.models.scan_result import (
    ScanResult, Region, RegionType, Annotation,
    LayoutInfo, PreprocessingInfo
)
from src.preprocess.processor import ImageProcessor

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for an OCR engine."""
    name: str
    type: str  # "local" or "cloud"
    enabled: bool = True
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    timeout: int = 30  # seconds
    cost_per_image: Optional[float] = None  # for cloud engines
    # Preprocessing preferences
    preferred_pp: List[str] = None  # preprocessing methods this engine likes
    tesseract_lang: str = "eng"
    tesseract_config: str = "--psm 6 --oem 3"

    def __post_init__(self):
        if self.preferred_pp is None:
            self.preferred_pp = ["gray_norm"]


class EngineInterface(ABC):
    """Abstract interface all OCR engines must implement."""

    @abstractmethod
    def process(self, image_path: str, config: EngineConfig) -> ScanResult:
        """
        Process an image and return structured OCR output.

        Args:
            image_path: Path to the image file
            config: EngineConfig with parameters

        Returns:
            ScanResult with regions, confidence, raw_text, etc.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return engine name (e.g., 'tesseract-5.5')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the engine can be used (binary installed, API key valid, etc.)."""
        pass


class EngineService:
    """
    Main orchestrator for OCR processing.
    Manages engine selection, preprocessing, and result normalization.
    """

    def __init__(self, engines_dir: str = None):
        self.engines: List[EngineInterface] = []
        self.processor = ImageProcessor()

    def auto_register_default_engines(self, config: Optional["OCRConfig"] = None) -> None:
        """
        Auto-register all available OCR engines based on configuration.
        - Gemini Vision (if GOOGLE_API_KEY)
        - Claude Vision (if ANTHROPIC_API_KEY)
        - EasyOCR (if installed)
        - Tesseract (if binary found)
        """
        from src.engines.tesseract_engine import TesseractEngine
        from src.engines.easyocr_engine import EasyOCREngine

        # Try EasyOCR
        try:
            easy = EasyOCREngine()
            if easy.is_available():
                self.register_engine(easy)
        except Exception as e:
            logger.warning(f"EasyOCR registration failed: {e}")

        # Try Tesseract
        try:
            tess = TesseractEngine()
            if tess.is_available():
                self.register_engine(tess)
        except Exception as e:
            logger.warning(f"Tesseract registration failed: {e}")

        # Try Gemini Vision (if config has API key)
        if config and config.engine.google_api_key:
            try:
                from src.engines.gemini_vision_engine import GeminiVisionEngine
                gemini = GeminiVisionEngine(
                    api_key=config.engine.google_api_key,
                    model=config.engine.gemini_model
                )
                if gemini.is_available():
                    self.register_engine(gemini)
            except Exception as e:
                logger.warning(f"Gemini Vision registration failed: {e}")

        # Try Claude Vision (if config has API key)
        if config and config.engine.anthropic_api_key:
            try:
                from src.engines.claude_vision_engine import ClaudeVisionEngine
                claude = ClaudeVisionEngine(
                    api_key=config.engine.anthropic_api_key,
                    model=config.engine.claude_model
                )
                if claude.is_available():
                    self.register_engine(claude)
            except Exception as e:
                logger.warning(f"Claude Vision registration failed: {e}")

    def register_engine(self, engine: EngineInterface) -> None:
        """Register an OCR engine."""
        self.engines.append(engine)
        logger.info(f"Registered engine: {engine.name()}")

    def get_available_engines(self) -> List[str]:
        """List available engine names."""
        return [e.name() for e in self.engines if e.is_available()]

    def _select_image(self, image_path: str, methods: List[str]) -> Tuple[object, str]:
        """
        Try different preprocessing methods and pick the one that
        gives the best OCR input. Returns (image_array, method_used).
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        h, w = img.shape[:2]
        original_size = (w, h)

        # Try preferred preprocessing method first
        # The ImageProcessor handles all CV preprocessing
        best_img = img
        best_method = "none"

        # For OCR, grayscale + normalization typically works best
        # for handwriting detection
        for method in ["gray_norm", "gray", "none"]:
            if method == "none":
                candidate = img
            else:
                candidate = self.processor.apply(image_path, [method])
                if candidate is None:
                    candidate = img

            # Quick sanity check: ensure image has content
            gray_check = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY) if len(candidate.shape) == 3 else candidate
            if gray_check.std() > 5:  # has some variation
                best_img = candidate
                best_method = method

        # Resize for OCR (max dimension ~1600 for good balance of speed/accuracy)
        best_img, final_size = self.processor.resize_for_ocr(best_img, max_dim=1600)

        return best_img, best_method

    def process(
        self,
        image_path: str,
        engine: str = "auto",
        config: Optional[EngineConfig] = None,
    ) -> ScanResult:
        """
        Process an image through the OCR pipeline.

        Args:
            image_path: Path to input image
            engine: "auto" (best available), or specific engine name
            config: Optional EngineConfig override

        Returns:
            ScanResult — Phase 1 output
        """
        import time
        start_time = time.time()

        if not os.path.exists(image_path):
            return ScanResult(
                status="error",
                warnings=[f"Image not found: {image_path}"],
            )

        # Select preprocessing
        processed_img, pp_method = self._select_image(image_path, ["gray_norm"])
        orig_h, orig_w = cv2.imread(image_path).shape[:2]

        preprocessing_info = PreprocessingInfo(
            operations=[pp_method],
            original_dimensions=(orig_w, orig_h),
            processed_dimensions=(processed_img.shape[1], processed_img.shape[0]),
        )

        # Engine selection
        target_engine: Optional[EngineInterface] = None

        if engine == "auto":
            # Auto-select: prioritize by accuracy (cloud first if configured)
            # Priority: gemini-vision > claude-vision > easyocr > tesseract > others
            auto_priority = ["gemini", "claude", "easyocr", "tesseract", "anthropic"]
            for prefix in auto_priority:
                for e in self.engines:
                    engine_name = e.name().lower()
                    if engine_name.startswith(prefix) and e.is_available():
                        target_engine = e
                        break
                if target_engine:
                    break
            # Fallback: any available engine
            if target_engine is None:
                for e in self.engines:
                    if e.is_available():
                        target_engine = e
                        break
            if target_engine is None:
                return ScanResult(
                    status="error",
                    warnings=[f"No available OCR engine found"],
                    preprocessing=preprocessing_info,
                )
        else:
            # Find engine by name (supports prefix match: "tesseract" matches "tesseract-5.5.3")
            matched = False
            for e in self.engines:
                engine_full_name = e.name()
                if (engine_full_name == engine or engine_full_name.startswith(engine)) and e.is_available():
                    target_engine = e
                    matched = True
                    break
            if not matched:
                return ScanResult(
                    status="error",
                    warnings=[f"Engine '{engine}' not available"],
                    preprocessing=preprocessing_info,
                )

        logger.info(
            f"Processing '{image_path}' with engine '{target_engine.name()}' "
            f"(preprocessing: {pp_method})"
        )

        # Run engine
        result = target_engine.process(image_path, config or EngineConfig(
            name=target_engine.name(), type="local"
        ))

        # Attach preprocessing info
        result.preprocessing = preprocessing_info

        # Compute layout analysis
        result.layout = self._analyze_layout(image_path, result)

        # Compute overall confidence
        if result.regions:
            result.confidence = float(np.mean([r.confidence for r in result.regions]))
        else:
            result.confidence = 0.0

        elapsed = time.time() - start_time
        logger.info(
            f"OCR complete: {len(result.regions)} regions, "
            f"confidence={result.confidence:.3f}, time={elapsed:.2f}s"
        )

        # Add warning if no text detected
        if not result.regions or result.confidence < 0.1:
            result.warnings.append("No text detected — image may be too blurry or blank")
            result.status = "no_text_detected"
        elif result.confidence < 0.5:
            result.status = "uncertain"
            result.warnings.append(
                "Low confidence results — review recommended"
            )

        return result

    def _analyze_layout(self, image_path: str, result: ScanResult) -> LayoutInfo:
        """
        Analyze page layout: orientation, columns, margins.
        """
        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        orientation = "portrait" if h > w else "landscape"

        # Column detection: look at x-distribution of regions
        if result.regions:
            x_centers = [np.mean([r.bbox[0][0], r.bbox[2][0]]) for r in result.regions]
            x_min = min(x_centers)
            x_max = max(x_centers)

            if x_max - x_min > w * 0.6:
                # Check if bimodal (2 columns)
                mid = (x_min + x_max) / 2
                left_count = sum(1 for x in x_centers if x < mid)
                right_count = sum(1 for x in x_centers if x >= mid)

                columns = 2 if left_count > 3 and right_count > 3 else 1
            else:
                columns = 1

            # Line height estimate
            y_centers = sorted([np.mean([r.bbox[0][1], r.bbox[2][1]]) for r in result.regions])
            if len(y_centers) > 2:
                gaps = [y_centers[i+1] - y_centers[i] for i in range(len(y_centers)-1)]
                line_height = int(np.median(gaps)) if gaps else 40
            else:
                line_height = 40
        else:
            columns = 1
            line_height = 40

        return LayoutInfo(
            page_width=w,
            page_height=h,
            orientation=orientation,
            columns=columns,
            line_height_px=line_height,
            margin_left=20,
            margin_right=20,
            margin_top=20,
            margin_bottom=20,
        )
