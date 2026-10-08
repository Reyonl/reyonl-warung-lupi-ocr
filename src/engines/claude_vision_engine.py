"""
Claude Vision OCR Engine (production-grade, optional).

Uses Anthropic's Claude API with vision capabilities for handwriting recognition.
Much more accurate than Tesseract/EasyOCR for handwritten text,
but requires API key (not bundled — configured via environment variable).

This engine is OPTIONAL — the pipeline falls back to Tesseract/EasyOCR
when Claude Vision is not configured.

Environment variables:
    ANTHROPIC_API_KEY — required for Claude Vision
    CLAUDE_VISION_MODEL — model to use (default: claude-3-5-sonnet-20241022)

Pricing (as of 2026):
    Haiku 4.5: $1.00/M input, $5.00/M output (fast, cheap)
    Sonnet 4.6: $3.00/M input, $15.00/M output (high accuracy)
    Opus 4.7: $5.00/M input, $25.00/M output (best quality)
    
    For OCR of a single notebook image (~5-10MB): ~$0.05-0.15 per image with Sonnet
"""

import logging
import base64
import json
import os
from typing import Optional
from src.engine.engine_service import EngineInterface, EngineConfig
from src.models.scan_result import (
    ScanResult, Region, RegionType, Annotation,
)

logger = logging.getLogger(__name__)


class ClaudeVisionEngine(EngineInterface):
    """
    Claude Vision API engine for high-accuracy handwriting OCR.

    Uses Anthropic's Claude with vision capabilities.
    Far superior to Tesseract/EasyOCR for:
    - Handwritten text recognition
    - Context-aware interpretation
    - Structural understanding (struck-through, circled, arrows)

    Requires ANTHROPIC_API_KEY environment variable.
    """

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model or os.environ.get("CLAUDE_VISION_MODEL", self.DEFAULT_MODEL)

    def name(self) -> str:
        return f"claude-vision-{self.model}"

    def is_available(self) -> bool:
        """Check if Claude Vision is configured and API key is present."""
        if not self.api_key:
            logger.warning("Claude Vision not configured (missing API key)")
            return False
        try:
            import anthropic  # type: ignore
        except ImportError:
            logger.warning("anthropic package not installed for Claude Vision")
            return False
        return True

    def process(self, image_path: str, config: EngineConfig) -> ScanResult:
        """Process image through Claude Vision API."""
        if not self.is_available():
            return ScanResult(
                status="error",
                warnings=["Claude Vision not available (missing API key or anthropic package)"],
            )

        try:
            import anthropic
        except ImportError:
            return ScanResult(
                status="error",
                warnings=["anthropic package not installed"],
            )

        # Read image
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            return ScanResult(
                status="error",
                warnings=[f"Cannot read image: {e}"],
            )

        # Prepare the vision prompt
        prompt = self._build_prompt()

        client = anthropic.Anthropic(api_key=self.api_key)

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=8000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                    ],
                }],
            )

            # Parse the response
            text_response = response.content[0].text

            # Extract JSON from the response
            # Claude may wrap in markdown code blocks
            json_str = text_response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()

            result_dict = json.loads(json_str)

            # Convert to ScanResult
            scan = ScanResult(
                status="ok" if result_dict.get("confidence", 0) > 0.3 else "uncertain",
                model=self.name(),
                confidence=result_dict.get("confidence", 0.0),
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
                            type("Symbol", (), {"text": s["text"], "bbox": s["bbox"], "confidence": s.get("confidence", 1.0)})()
                            for s in region_data.get("symbols", [])
                        ],
                    ),
                    engine=self.name(),
                )
                scan.regions.append(region)

            return scan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude Vision JSON: {e}")
            # Re-call without JSON parsing — extract text directly
            return self._fallback_text_extraction(image_path, text_response)
        except Exception as e:
            logger.error(f"Claude Vision processing failed: {e}")
            return ScanResult(
                status="error",
                model=self.name(),
                warnings=[f"Claude Vision error: {str(e)[:200]}"],
            )

    def _build_prompt(self) -> str:
        """Build the vision prompt for handwriting OCR."""
        return """
You are an expert OCR system specializing in Indonesian warung (street shop) bookkeeping notes.
Analyze this image of handwritten notes carefully and extract all text content.

Return your response as raw JSON (no markdown wrapping) with this exact structure:

{
  "raw_text": "Full transcribed text of the page",
  "confidence": 0.95,
  "regions": [
    {
      "text": "word_or_phrase",
      "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
      "confidence": 0.95,
      "type": "product_name|quantity|unit_price|date|customer_name|note|total|unknown",
      "has_circle": false,
      "circle_bbox": null,
      "has_arrow": false,
      "has_strikethrough": false,
      "symbols": [{"text": "@", "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "confidence": 0.9}]
    }
  ],
  "warnings": []
}

Key instructions:
1. Extract ALL text, including: product names, quantities, prices, customer names, dates, notes, doodles
2. Classify regions by type: product_name for items, quantity for numbers near products, unit_price for price values, date for dates, customer_name for names
3. Detect structural markers: circles drawn around text (has_circle=true), arrows pointing to text, strikethrough lines
4. For Indonesian text: handle common transliterations like "Kamis", "OKT", "sore", "bon"
5. Numbers: distinguish quantity (small: 1-99) from price (large: thousands+)
6. Be honest about confidence — flag low confidence items in warnings
7. If text is completely unreadable, return confidence < 0.3
8. bbox coordinates should be in original image pixel coordinates
"""

    def _fallback_text_extraction(self, image_path: str, vision_text: str) -> ScanResult:
        """Fallback if Claude response isn't valid JSON."""
        return ScanResult(
            status="uncertain",
            model=self.name(),
            confidence=0.3,
            raw_text=vision_text,
            warnings=["Claude Vision returned non-JSON response — using raw text only"],
        )
