"""
Audit Trail Logger (Phase 7).
Persists interpretation results for full traceability.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class AuditTrail:
    """Persistent audit trail for the OCR pipeline."""

    def __init__(self, log_dir: str = "audit_logs", enabled: bool = True):
        self.log_dir = log_dir
        self.enabled = enabled
        os.makedirs(log_dir, exist_ok=True)

    def new_trace(self) -> str:
        """Create a new trace ID for a pipeline run."""
        return str(uuid4())

    def log_phase1_ocr(self, trace_id, engine, confidence, region_count, raw_text, regions):
        """Log Phase 1: OCR result."""
        self._write(trace_id, "phase1_ocr", {
            "engine": engine,
            "confidence": round(confidence, 4),
            "region_count": region_count,
            "raw_text": raw_text[:500],
            "regions": [{"text": r.get("text", "")[:50], "confidence": r.get("confidence")} for r in regions[:20]],
        })

    def log_phase2_interpretation(self, trace_id, date, date_confidence, customer, items, uncertain):
        """Log Phase 2: Interpretation result."""
        self._write(trace_id, "phase2_interpretation", {
            "date": date,
            "date_confidence": round(date_confidence, 4),
            "customer": customer,
            "items_count": len(items),
            "items_summary": [{"text": i.get("detected_text", "")[:50], "qty": i.get("quantity"), "price": i.get("price"), "confidence": round(i.get("confidence", 0), 4)} for i in items],
            "uncertain": uncertain,
        })

    def log_phase3_product_matching(self, trace_id, matches):
        self._write(trace_id, "phase3_product_matching", {"matches": matches})

    def log_phase4_customer_matching(self, trace_id, detected_name, candidates, matched_id):
        self._write(trace_id, "phase4_customer_matching", {"detected_name": detected_name, "matched_id": matched_id, "candidates": candidates})

    def log_phase5_draft(self, trace_id, draft):
        self._write(trace_id, "phase5_draft", draft)

    def log_phase6_user_action(self, trace_id, action, details):
        self._write(trace_id, "phase6_user_action", {"action": action, "details": details})

    def log_phase9_api(self, trace_id, payload, response=None, error=None):
        self._write(trace_id, "phase9_api", {"payload": payload, "response": response, "error": error})

    def get_trace(self, trace_id) -> Dict[str, Any]:
        """Retrieve full audit trail for a trace ID."""
        log_file = os.path.join(self.log_dir, f"{trace_id}.jsonl")
        entries = []
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                for line in f:
                    entries.append(json.loads(line.strip()))
        return {"trace_id": trace_id, "entries": entries}

    def _write(self, trace_id, phase, data):
        """Write an audit entry."""
        if not self.enabled:
            return
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "trace_id": trace_id, "phase": phase, "data": data}
        log_file = os.path.join(self.log_dir, f"{trace_id}.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
        logger.debug(f"Audit: {trace_id} - {phase}")
