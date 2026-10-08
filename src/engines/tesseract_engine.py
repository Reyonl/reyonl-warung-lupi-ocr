"""
Tesseract OCR Engine (local, offline, free).

Implements EngineInterface for Tesseract 5.x via pytesseract.
Supports both eng and Indonesian (id) traineddata files.
Optimized for handwritten text in Indonesian context.
"""

import pytesseract
import cv2
import numpy as np
import logging
from typing import Optional
import os

from src.engine.engine_service import EngineInterface, EngineConfig
from src.models.scan_result import (
    ScanResult, Region, RegionType, Annotation,
)

logger = logging.getLogger(__name__)

# Blacklist chars commonly misread from handwriting noise
# Brackets and braces often confused with handwriting strokes
BLACKLIST_CONFIG = "-c tessedit_char_blacklist=|<>"


class TesseractEngine(EngineInterface):
    """Tesseract OCR engine wrapper."""

    TESSERACT_CMD: Optional[str] = None
    _lang_cache: Optional[dict] = None

    def __init__(self):
        self._find_tesseract_binary()
        self._find_indonesian_lang()

    def _find_tesseract_binary(self):
        """Locate tesseract binary."""
        candidates = [
            pytesseract.pytesseract.tesseract_cmd,
            "tesseract",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]

        for candidate in candidates:
            if candidate and os.path.isfile(str(candidate)):
                pytesseract.pytesseract.tesseract_cmd = candidate
                self.TESSERACT_CMD = candidate
                logger.info(f"Tesseract binary: {candidate}")
                return

        from shutil import which
        path_found = which("tesseract")
        if path_found:
            pytesseract.pytesseract.tesseract_cmd = path_found
            self.TESSERACT_CMD = path_found
            logger.info(f"Tesseract binary (from PATH): {path_found}")
            return

        logger.warning("Tesseract binary not found!")
        self.TESSERACT_CMD = None

    def _find_indonesian_lang(self):
        """Find available languages and pick the best for Indonesian text."""
        local_tessdata = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tessdata"
        )
        global_tessdata = r"C:\Program Files\Tesseract-OCR\tessdata"

        self._lang_cache = {}

        if os.path.isdir(local_tessdata):
            for f in os.listdir(local_tessdata):
                if f.endswith(".traineddata"):
                    lang = f.replace(".traineddata", "")
                    self._lang_cache[lang] = True

        if os.path.isdir(global_tessdata):
            for f in os.listdir(global_tessdata):
                if f.endswith(".traineddata"):
                    lang = f.replace(".traineddata", "")
                    if lang not in self._lang_cache:
                        self._lang_cache[lang] = True

        if "ind" in self._lang_cache:
            self.lang = "ind"
        elif "eng" in self._lang_cache:
            self.lang = "eng"
            logger.info("Indonesian traineddata not found - using English.")
        else:
            self.lang = "eng"

        logger.info(f"Tesseract languages: {list(self._lang_cache.keys())}")
        logger.info(f"Selected language: {self.lang}")

    def name(self) -> str:
        version = "unknown"
        try:
            version = pytesseract.get_tesseract_version()
        except Exception:
            pass
        return f"tesseract-{version}"

    def is_available(self) -> bool:
        return self.TESSERACT_CMD is not None

    def process(self, image_path: str, config: EngineConfig) -> ScanResult:
        """Process image with Tesseract and return structured result."""
        if self.TESSERACT_CMD is None:
            return ScanResult(status="error", warnings=["Tesseract not found"])

        img = cv2.imread(image_path)
        if img is None:
            return ScanResult(
                status="error",
                warnings=[f"Cannot read image: {image_path}"]
            )

        original_h, original_w = img.shape[:2]

        # Preprocessing: grayscale + normalize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Gaussian blur + adaptive threshold for noisy handwriting scans
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        gray_thresh = cv2.adaptiveThreshold(
            gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 5
        )

        # Resize
        new_w = 1600
        new_h = int(original_h * new_w / original_w)
        gray_resized = cv2.resize(
            gray_thresh, (new_w, new_h), interpolation=cv2.INTER_CUBIC
        )

        scale_x = original_w / new_w
        scale_y = original_h / new_h

        # Set TESSDATA_PREFIX for local traineddata
        local_tessdata = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tessdata"
        )
        if os.path.isdir(local_tessdata):
            os.environ["TESSDATA_PREFIX"] = local_tessdata

        # Tesseract configs optimized for handwriting
        bl = "|<>[]{}"
        configs = [
            ("psm6_oem3", f"--psm 6 --oem 3 -c tessedit_char_blacklist={bl}"),
            ("psm4_oem3", f"--psm 4 --oem 3 -c tessedit_char_blacklist={bl}"),
            ("psm11_oem3", "--psm 11 --oem 3"),
            ("psm7_oem3", "--psm 7 --oem 3"),
        ]

        best_result: Optional[ScanResult] = None
        best_conf = 0

        for cfg_name, cfg_str in configs:
            try:
                data = pytesseract.image_to_data(
                    gray_resized,
                    lang=self.lang,
                    config=cfg_str,
                    output_type=pytesseract.Output.DICT
                )

                full_text = pytesseract.image_to_string(
                    gray_resized,
                    lang=self.lang,
                    config=cfg_str
                )

                regions = []
                confs = []
                for i in range(len(data["text"])):
                    text_val = data["text"][i].strip()
                    if not text_val:
                        continue

                    conf = float(data["conf"][i])
                    if conf < 0:
                        conf = 0.0
                    confs.append(conf)

                    x = int(data["left"][i] * scale_x)
                    y = int(data["top"][i] * scale_y)
                    w = int(data["width"][i] * scale_x)
                    h = int(data["height"][i] * scale_y)

                    bbox = [
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h],
                    ]

                    region = Region(
                        text=text_val,
                        bbox=bbox,
                        confidence=conf / 100.0,
                        region_type=RegionType.UNKNOWN,
                        annotation=Annotation(),
                        engine=self.name(),
                    )
                    regions.append(region)

                avg_conf = float(np.mean(confs)) / 100.0 if confs else 0

                if avg_conf > best_conf:
                    best_conf = avg_conf
                    warning_msg = []
                    if len(regions) <= 2:
                        warning_msg.append("Few text regions detected - image quality may be low")
                    best_result = ScanResult(
                        status="ok",
                        model=f"tesseract-{cfg_name}",
                        confidence=avg_conf,
                        raw_text=full_text,
                        regions=regions,
                        warnings=warning_msg,
                    )

            except Exception as e:
                logger.warning(f"Tesseract config {cfg_name} failed: {e}")
                continue

        if best_result is None:
            return ScanResult(
                status="error",
                model="tesseract",
                warnings=["All Tesseract configurations failed"],
            )

        logger.info(
            f"Tesseract result: conf={best_result.confidence:.3f}, "
            f"regions={len(best_result.regions)}"
        )
        return best_result
