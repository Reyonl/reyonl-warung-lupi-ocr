"""
Gemini Vision OCR Engine (production-grade, optional).

Uses Google's Gemini API with vision capabilities for handwriting recognition.
Much more accurate than Tesseract/EasyOCR for handwritten text, especially
Indonesian notebook handwriting with noise/blur/perspective issues.

This engine is OPTIONAL — the pipeline falls back to Tesseract/EasyOCR
when Gemini Vision is not configured.

Environment variables:
    GOOGLE_API_KEY     — required for Gemini Vision
    GEMINI_VISION_MODEL — model to use (default: gemini-3.5-flash)

Pricing (as of 2026-10, in IDR approx):
    gemini-2.0-flash:  Rp 0.60/token
    gemini-2.5-pro:    Rp 1.20/token
    For OCR of a single notebook image (~5-10MB): ~Rp 2,000-5,000 per image

Usage:
    from src.engines.gemini_vision_engine import GeminiVisionEngine
    engine = GeminiVisionEngine(api_key="your-google-api-key")
    scan = engine.process("image.jpg", config)
"""
import base64
import json
import os
import re
import logging
import time
from typing import Optional
from src.engine.engine_service import EngineInterface, EngineConfig
from src.models.scan_result import (
    ScanResult, Region, RegionType, Annotation,
)

logger = logging.getLogger(__name__)


class GeminiVisionEngine(EngineInterface):
    """
    Gemini Vision API engine for high-accuracy handwriting OCR.

    Uses Google's Gemini API (Vision capability).
    Much superior to Tesseract/EasyOCR for:
    - Handwritten text recognition
    - Context-aware interpretation
    - Structural understanding (struck-through, circled, arrows)
    - Indonesian language context
    """

    DEFAULT_MODEL = "gemini-3.5-flash"  # Most reliable; 3.8-flash often overloaded (503)
    # Fallback models: reliable fast ones first, then slower/higher-quality
    FALLBACK_MODELS = [
        "gemini-3.5-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.8-flash",
        "gemini-flash-latest",
    ]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model = model or os.environ.get("GEMINI_VISION_MODEL", self.DEFAULT_MODEL)
        self._client = None

    def name(self) -> str:
        return f"gemini-vision-{self.model}"

    def is_available(self) -> bool:
        """Check if Gemini Vision is configured and API key is present."""
        if not self.api_key:
            logger.warning("Gemini Vision not configured (missing GOOGLE_API_KEY)")
            return False
        try:
            import google.genai  # type: ignore
        except ImportError:
            logger.warning("google-genai package not installed for Gemini Vision")
            return False
        return True

    def _get_client(self):
        """Lazy-load the Gemini client to avoid startup cost."""
        if self._client is None:
            from google import genai
            from google.genai import types as gtypes
            self._client = genai.Client(api_key=self.api_key)
            self._genai_types = gtypes
        return self._client

    def process(self, image_path: str, config: EngineConfig) -> ScanResult:
        """Process image through Gemini Vision API with retry + fallback."""
        if not self.is_available():
            return ScanResult(
                status="error",
                warnings=["Gemini Vision not available (missing GOOGLE_API_KEY or google-genai package)"],
            )

        # Read image
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
        except Exception as e:
            return ScanResult(
                status="error",
                warnings=[f"Cannot read image: {e}"],
            )

        prompt = self._build_prompt()

        # Try primary model, then fallbacks with retry
        models_to_try = [self.model] + [m for m in self.FALLBACK_MODELS if m != self.model]
        last_error = None

        for model_name in models_to_try:
            for attempt in range(3):  # 3 retries per model
                try:
                    client = self._get_client()
                    from google.genai import types as gtypes

                    response = client.models.generate_content(
                        model=model_name,
                        contents=[
                            gtypes.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                            gtypes.Part.from_text(text=prompt),
                        ],
                        config=self._genai_types.GenerateContentConfig(
                            temperature=0.1,
                            max_output_tokens=8192,
                        ),
                    )

                    text_response = response.text

                    # Parse JSON
                    json_str = text_response.strip()
                    if json_str.startswith("```json"):
                        json_str = json_str[7:]
                    if json_str.endswith("```"):
                        json_str = json_str[:-3]
                    json_str = json_str.strip()

                    result_dict = json.loads(json_str)

                    # Build ScanResult
                    scan = ScanResult(
                        status="ok" if result_dict.get("confidence", 0) > 0.3 else "uncertain",
                        model=f"gemini-vision-{model_name}",
                        confidence=result_dict.get("confidence", 0.5),
                        raw_text=result_dict.get("raw_text", ""),
                        warnings=result_dict.get("warnings", []),
                    )

                    for region_data in result_dict.get("regions", []):
                        region = Region(
                            text=region_data["text"],
                            bbox=region_data["bbox"],
                            confidence=region_data.get("confidence", 0.5),
                            region_type=RegionType(region_data.get("type", "unknown")),
                            annotation=Annotation(
                                has_circle=region_data.get("has_circle", False),
                                circle_bbox=region_data.get("circle_bbox"),
                                has_arrow=region_data.get("has_arrow", False),
                                has_strikethrough=region_data.get("has_strikethrough", False),
                                symbols=[
                                    type("Symbol", (), {
                                        "text": s["text"], "bbox": s["bbox"],
                                        "confidence": s.get("confidence", 1.0)
                                    })()
                                    for s in region_data.get("symbols", [])
                                ],
                            ),
                            engine=scan.model,
                        )
                        scan.regions.append(region)

                    # Track token usage
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        scan.warnings.append(
                            f"tokens: input={response.usage_metadata.input_tokens}, "
                            f"output={response.usage_metadata.output_tokens}"
                        )

                    # Add retry note if we had to retry
                    if attempt > 0 or model_name != self.model:
                        scan.warnings.append(f"recovered via {model_name} (attempt {attempt+1})")

                    logger.info(f"Gemini Vision: conf={scan.confidence:.3f}, regions={len(scan.regions)}, model={model_name}")
                    return scan

                except Exception as e:
                    error_str = str(e)
                    last_error = e

                    # 503 (high demand) — try next model or retry
                    if "503" in error_str or "UNAVAILABLE" in error_str.upper():
                        if attempt < 2:
                            time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s
                            continue
                        else:
                            logger.warning(f"Model {model_name} unavailable after 3 retries, trying fallback...")
                            break  # try next model
                    elif "404" in error_str or "NOT_FOUND" in error_str.upper():
                        logger.warning(f"Model {model_name} not found, trying fallback...")
                        break  # try next model immediately
                    else:
                        # Other errors (JSON parse, network, etc.) — retry same model
                        if attempt < 2:
                            time.sleep(1)
                            continue
                        break

        # All retries exhausted
        return ScanResult(
            status="error",
            model=self.name(),
            warnings=[f"All Gemini Vision models failed after retries: {str(last_error)[:200]}"],
        )

    def _build_prompt(self) -> str:
        """Build the vision prompt for handwriting OCR with Indonesian context."""
        return """
Anda adalah sistem OCR yang ahli mengenali tulisan tangan buku catatan warung.
Anda akan mendapatkan gambar foto halaman buku catatan bon/bukti warung yang
ditulis tangan menggunakan pulpen di atas kertas bergaris.

Anda HARUS merespons dengan JSON saja (bukan markdown, bukan prose), menggunakan
struktur JSON berikut:

{
  "raw_text": "seluruh teks yang terbaca",
  "confidence": 0.95,
  "regions": [
    {
      "text": "kata atau frasa",
      "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
      "confidence": 0.95,
      "type": "product_name|quantity|unit_price|date|customer_name|note|total|unknown",
      "has_circle": false,
      "circle_bbox": null,
      "has_arrow": false,
      "has_strikethrough": false,
      "symbols": [{"text": "@", "bbox": [[x,y],[x,y],[x,y],[x,y]], "confidence": 0.9}]
    }
  ],
  "warnings": []
}

Instruksi kunci:
1. Ekstrak SEMUA teks termasuk: nama produk, jumlah, harga, nama customer, tanggal, catatan
2. Klasifikasikan tiap region: product_name=jualan, quantity=jumlah/barang,
   unit_price=harga satuan, date=tanggal, customer_name=nama orang
3. Deteksi penanda struktural: lingkaran di sekitar teks (has_circle=true), panah, coretan (strikethrough)
4. Untuk teks Indonesia/Warung Lupi: kenali singkatan seperti "OKT" (Oktober), "Kamis",
   "sore", "bon", "madu", "kopi", "teh", "rokok", "makan"
5. Angka: pisahkan jumlah (kecil: 1-99) dari harga (besar: ribuan+)
6. Beri confidence tinggi (>0.8) jika yakin, rendah (<0.5) jika ragu
7. Jika tidak bisa terbaca, isi confidence rendah dan beri warning
8. Koordinat bbox dalam pixel asli image

Contoh format harga: 3.500, 3500, 3rb, Rp3500, 3,5rb, 3k, 3500,-
Contoh tanggal: 08/10, 8 Okt, Kamis 8 Oktober 2026, tgl 8
Contoh jumlah: 2x, x2, (2)
"""
