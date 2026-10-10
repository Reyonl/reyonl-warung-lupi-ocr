"""
Configuration management for Warung Lupi OCR engine.

Handles:
- Engine selection (tesseract, easyocr, claude-vision)
- API endpoint configuration (Warung Lupi backend)
- Processing thresholds and parameters
- Audit log configuration

Configuration priority (highest first):
1. Environment variables
2. Config file (~/.warung-lupi-ocr.json)
3. Default values

Usage:
    config = OCRConfig()
    config.engine           # "easyocr" | "tesseract" | "auto"
    config.api.base_url     # Warung Lupi backend URL
    config.interpreter.confidence_threshold  # Minimum confidence to auto-select
    config.draft.auto_fill_defaults  # Use product defaults when price missing
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """OCR engine configuration."""
    # Which engine to use
    # "auto" = Claude Vision if key set, else EasyOCR if available, else Tesseract
    # "local" = force local engine (Tesseract/EasyOCR) — for cost optimization
    preferred_engine: str = "auto"
    # Google Gemini Vision (cloud, optional — production quality)
    google_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.5-flash"
    # Anthropic Claude Vision (cloud, optional — alternative production option)
    anthropic_api_key: Optional[str] = None
    claude_model: str = "claude-3-5-sonnet-20241022"
    # Tesseract
    tesseract_cmd: Optional[str] = None
    tesseract_lang: str = "ind"
    tessdata_dir: str = ""
    # EasyOCR
    easyocr_languages: list = field(default_factory=lambda: ["en", "id"])
    # Cloud fallback confidence threshold — if local OCR below this, use cloud
    cloud_fallback_threshold: float = 0.45


@dataclass
class APIConfig:
    """Warung Lupi backend API configuration."""
    base_url: str = "http://localhost:8000"
    api_token: Optional[str] = None
    timeout: int = 30


@dataclass
class InterpreterConfig:
    """Interpretation thresholds and parameters."""
    # Minimum OCR confidence for a region to be considered
    region_confidence_threshold: float = 0.35
    # Minimum confidence for product auto-selection
    product_match_threshold: float = 0.85
    # Minimum confidence for customer auto-selection  
    customer_match_threshold: float = 0.80
    # Confidence below which items need user review
    low_confidence_threshold: float = 0.50
    # Date confidence minimum
    date_confidence_threshold: float = 0.60
    # Y-coordinate tolerance for line clustering (pixels)
    line_y_tolerance: int = 60
    # Whether to auto-fill product defaults for draft items
    auto_fill_defaults: bool = True
    # Products to exclude from matching (deprecated items, etc.)
    excluded_product_ids: list = field(default_factory=list)


@dataclass
class AuditConfig:
    """Audit log configuration (Phase 7)."""
    # Enable audit logging
    enabled: bool = True
    # Directory for audit logs
    log_dir: str = "audit_logs"
    # Keep logs for N days (0 = keep forever)
    retention_days: int = 30
    # Write detailed regional-level data
    detailed: bool = True


@dataclass
class OCRConfig:
    """Root configuration for Warung Lupi OCR engine."""
    engine: EngineConfig = field(default_factory=EngineConfig)
    api: APIConfig = field(default_factory=APIConfig)
    interpreter: InterpreterConfig = field(default_factory=InterpreterConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    # Processing directory for temporary files
    temp_dir: str = ""

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "OCRConfig":
        """
        Load configuration from file + environment variables.

        Environment variable mapping:
            WLOCR_ENGINE → engine.preferred_engine
            WLOCR_CLAUDE_KEY → engine.anthropic_api_key
            WLOCR_API_URL → api.base_url
            WLOCR_API_TOKEN → api.api_token
            WLOCR_THRESHOLD → interpreter.low_confidence_threshold
        """
        config = cls()

        # Load from file
        if config_path is None:
            config_path = cls._find_config_file() if hasattr(cls, '_find_config_file') else None

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    file_config = json.load(f)
                config._merge_dict(file_config)
                logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        # Override with environment variables (highest priority)
        env_map = {
            "WLOCR_ENGINE": ("engine", "preferred_engine"),
            "GOOGLE_API_KEY": ("engine", "google_api_key"),
            "GEMINI_VISION_MODEL": ("engine", "gemini_model"),
            "ANTHROPIC_API_KEY": ("engine", "anthropic_api_key"),
            "CLAUDE_VISION_MODEL": ("engine", "claude_model"),
            "WLOCR_TESSERACT_CMD": ("engine", "tesseract_cmd"),
            "WLOCR_API_URL": ("api", "base_url"),
            "WLOCR_API_TOKEN": ("api", "api_token"),
            "WLOCR_CONFIDENCE_THRESHOLD": ("interpreter", "low_confidence_threshold"),
            "WLOCR_PRODUCT_THRESHOLD": ("interpreter", "product_match_threshold"),
            "WLOCR_AUDIT_ENABLED": ("audit", "enabled"),
            "WLOCR_AUDIT_DIR": ("audit", "log_dir"),
        }

        for env_key, (section, attr) in env_map.items():
            env_val = os.environ.get(env_key)
            if env_val:
                # Try to convert to appropriate type
                current_val = getattr(getattr(config, section), attr)
                if isinstance(current_val, bool):
                    val = env_val.lower() in ("true", "1", "yes", "on")
                elif isinstance(current_val, int):
                    val = int(env_val)
                elif isinstance(current_val, float):
                    val = float(env_val)
                elif isinstance(current_val, list):
                    val = env_val.split(",")
                else:
                    val = env_val
                setattr(getattr(config, section), attr, val)

        # Set default temp dir
        if not config.temp_dir:
            config.temp_dir = os.environ.get("TMPDIR", "/tmp")

        # Set tessdata dir if not set
        if not config.engine.tessdata_dir:
            # Look in project directory
            project_tessdata = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "tessdata"
            )
            if os.path.isdir(project_tessdata):
                config.engine.tessdata_dir = project_tessdata

        return config

    def _merge_dict(self, d: Dict[str, Any]) -> None:
        """Merge a dictionary into this config."""
        for section, values in d.items():
            if hasattr(self, section) and isinstance(values, dict):
                target = getattr(self, section)
                for key, val in values.items():
                    if hasattr(target, key):
                        setattr(target, key, val)

    @staticmethod
    def _find_config_file() -> Optional[str]:
        """Find config file in common locations."""
        candidates = [
            os.environ.get("WLOCR_CONFIG"),
            os.path.join(os.getcwd(), ".warung-lupi-ocr.json"),
            os.path.join(os.path.expanduser("~"), ".warung-lupi-ocr.json"),
            "/etc/warung-lupi-ocr.json",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    def validate(self) -> list:
        """
        Validate configuration and return list of issues.
        Returns empty list if all OK.
        """
        issues = []

        # Check engine availability
        if self.engine.preferred_engine == "tesseract":
            try:
                import pytesseract
                if self.engine.tesseract_cmd:
                    pytesseract.pytesseract.tesseract_cmd = self.engine.tesseract_cmd
                pytesseract.get_tesseract_version()
            except Exception:
                issues.append("Tesseract configured but not installed or not in PATH")

        elif self.engine.preferred_engine == "easyocr":
            try:
                import easyocr  # type: ignore
            except ImportError:
                issues.append("EasyOCR configured but 'easyocr' package not installed")

        elif self.engine.preferred_engine in ("gemini-vision", "gemini"):
            if not self.engine.google_api_key:
                issues.append("Gemini Vision selected but GOOGLE_API_KEY not set")
            else:
                try:
                    import google.genai  # type: ignore
                except ImportError:
                    issues.append("Gemini Vision selected but 'google-genai' package not installed")

        elif self.engine.preferred_engine == "claude-vision":
            if not self.engine.anthropic_api_key:
                issues.append("Claude Vision selected but ANTHROPIC_API_KEY not set")
            else:
                try:
                    import anthropic  # type: ignore
                except ImportError:
                    issues.append("Claude Vision selected but 'anthropic' package not installed")

        # Check API config
        if not self.api.base_url:
            issues.append("API base_url not configured")

        return issues
