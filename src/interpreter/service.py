"""
Phase 2: Handwriting Interpretation Service.

Takes a Phase 1 ScanResult (raw OCR output) and interprets it
in the context of Indonesian warung (warung) bookkeeping.

What it does:
- Detects and classifies date format
- Identifies customer name (first named entity, near "pelanggan"/"plg")
- Parses item lines: product_name, quantity, unit_price
- Flags struck-through items
- Identifies structural symbols (@, +, =, Rp, x)
- Marks uncertain fields for user review

Output model: InterpretedResult (consumed by Phase 3 matching + Phase 5 draft)
"""

import re
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
import numpy as np

from src.models.scan_result import ScanResult, Region, RegionType, Annotation

logger = logging.getLogger(__name__)


@dataclass
class InterpretedItem:
    """A single interpreted item from the book note."""
    detected_text: str
    quantity: Optional[int] = None
    price: Optional[float] = None
    unit: str = "pcs"
    confidence: float = 0.0
    region_ids: List[str] = field(default_factory=list)
    struck_through: bool = False
    uncertain_fields: List[str] = field(default_factory=list)
    # Spatial info for UI layout
    y_position: int = 0
    x_position: int = 0
    # For product matching (Phase 3)
    product_candidates: List[dict] = field(default_factory=list)
    matched_product_id: Optional[int] = None
    matched_product_name: str = ""


@dataclass
class InterpretedCustomer:
    """Interpreted customer information."""
    detected_name: str
    confidence: float = 0.0
    customer_candidates: List[dict] = field(default_factory=list)
    matched_customer_id: Optional[int] = None


@dataclass
class InterpretedResult:
    """
    Phase 2 output — structured interpretation of raw OCR.

    This is what gets passed to:
    - Phase 3 (Product Matching): add product_candidates to items
    - Phase 4 (Customer Matching): add customer_candidates
    - Phase 5 (Draft Transaction): used to create draft transaction
    """
    date: Optional[str] = None               # ISO format: "2026-09-28"
    date_confidence: float = 0.0
    customer: Optional[InterpretedCustomer] = None
    items: List[InterpretedItem] = field(default_factory=list)
    uncertain: List[str] = field(default_factory=list)
    notes: str = ""
    total_confidence: float = 0.0
    # Preserve Phase 1 source for audit
    source_scan: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "date_confidence": round(self.date_confidence, 4),
            "customer": {
                "detected_name": self.customer.detected_name if self.customer else None,
                "confidence": round(self.customer.confidence, 4) if self.customer else 0,
                "matched_id": self.customer.matched_customer_id if self.customer else None,
                "candidates": self.customer.customer_candidates if self.customer else [],
            } if self.customer else None,
            "items": [
                {
                    "detected_text": item.detected_text,
                    "quantity": item.quantity,
                    "price": item.price,
                    "unit": item.unit,
                    "confidence": round(item.confidence, 4),
                    "struck_through": item.struck_through,
                    "uncertain_fields": item.uncertain_fields,
                    "product_candidates": item.product_candidates,
                    "matched_product_id": item.matched_product_id,
                    "matched_product_name": item.matched_product_name,
                }
                for item in self.items
            ],
            "uncertain": self.uncertain,
            "notes": self.notes,
            "total_confidence": round(self.total_confidence, 4),
        }


class InterpretationService:
    """
    Main interpreter for handwritten book notes.

    Uses rule-based parsing with spatial analysis and heuristics.
    Designed to be conservative — uncertain results are flagged, not guessed.
    """

    # Known Indonesian day names
    DAY_NAMES = {
        "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu"
    }

    # Indonesian month abbreviations (used in handwritten dates)
    MONTH_ABBR = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "may": 5, "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "okt": 10, "nov": 11, "des": 12,
        "januari": 1, "februari": 2, "maret": 3, "april": 4,
        "mei": 5, "juni": 6, "juli": 7, "agustus": 8,
        "september": 9, "oktober": 10, "november": 11, "desember": 12,
    }

    # Structural symbols that affect interpretation
    STRUCTURAL_CHARS = {"@", "x", "X", "+", "=", "→", "★", "×", "Rp"}

    # Known product keywords (for context)
    PRODUCT_KEYWORDS = {
        "kopi", "teh", "susu", "jeruk", "es", "air", "aqua", "galon",
        "makan", "nasi", "telur", "mie", "goreng", "donat", "ketan",
        "nagasari", "gorengan", "sosis", "roti", "rokok", "cigarette",
        "madu", "karamel", "kecap", "saus", "garam", "gula",
        "minyak", "terigu", "beras", "kentang", "wortel", "kangkung",
        "bayam", "kacang", "tempe", "tahu", "ayam", "ikan", "sapi",
        "telur", "cemilan", "snack", "permen", "mint", "klem",
        "bungkus", "batang", "pcs", "kaleng", "botol", "galon",
        "porsi", "gelas", "liter", "kg", "kg",
    }

    # Common Indonesian abbreviations in warung notes
    ABBREVIATIONS = {
        "bon": "bon",
        "bp": "belanja",
        "bkp": "belanja",
        "plg": "pelanggan",
        "pgn": "pengen",
        "msh": "masih",
        "sdh": "sudah",
        "blm": "belum",
        "utk": "untuk",
        "dgn": "dengan",
        "dll": "dan lain-lain",
        "dsb": "dan seterusnya",
        "hutang": "hutang",
        "lunas": "lunas",
        "tunai": "tunai",
        "kredit": "kredit",
        "diskon": "diskon",
        "pot": "potongan",
        "tot": "total",
    }

    def __init__(self):
        pass

    def interpret(self, scan: ScanResult) -> InterpretedResult:
        """
        Parse a Phase 1 ScanResult into structured interpretation.

        Strategy:
        1. Sort regions by position (top-to-bottom, left-to-right)
        2. Cluster into lines (Y-coordinate similarity)
        3. Classify each line as: date, customer, item, note, or unknown
        4. Parse items: extract product name, qty, price from line context
        5. Flag uncertainties conservatively
        """
        result = InterpretedResult()

        if not scan.regions:
            result.uncertain.append("No regions to interpret")
            return result

        # Preserve source for audit
        result.source_scan = {
            "model": scan.model,
            "confidence": scan.confidence,
            "region_count": len(scan.regions),
        }

        # Sort regions by position
        sorted_regions = sorted(scan.regions, key=lambda r: (
            # Y center for line grouping
            float(np.mean([r.bbox[0][1], r.bbox[2][1]])),
            # X position for left-to-right
            r.bbox[0][0]
        ))

        # Phase 1.5: Noise filtering — remove OCR noise regions
        # before sending to interpretation
        # Remove: low confidence, pure symbol lines, single-char noise
        NOISE_PATTERNS = re.compile(r'^[—–−|]+$|^[\.,;:!?]+$|^[+\*★※]+$')
        filtered_regions = []
        for r in sorted_regions:
            text = r.text.strip()
            # Keep high-confidence regions (>0.35) regardless of content
            # Keep numeric regions (they might be qty/price)
            # Keep known product keywords
            # Remove pure symbol lines and single non-alpha chars
            is_numeric = bool(re.match(r'^\d+$', text))
            is_keyword = any(kw in text.lower() for kw in self.PRODUCT_KEYWORDS)
            is_date_part = any(d in text.lower() for d in
                              list(self.DAY_NAMES) + list(self.MONTH_ABBR.keys()))
            is_noise = bool(NOISE_PATTERNS.match(text))
            is_single_non_alpha = len(text) == 1 and not text.isalpha()

            if (r.confidence > 0.35 or is_numeric or is_keyword or
                is_date_part or text in self.STRUCTURAL_CHARS):
                if not (is_noise and not is_keyword):
                    if not (is_single_non_alpha and r.confidence < 0.5):
                        filtered_regions.append(r)
            elif r.confidence > 0.5 and len(text) > 1:
                filtered_regions.append(r)

        sorted_regions = filtered_regions

        # Cluster into lines
        lines = self._cluster_into_lines(sorted_regions)

        # Process each line
        current_line_type = "unknown"

        for line_idx, line_regions in enumerate(lines):
            if not line_regions:
                continue

            line_text = " ".join(r.text for r in line_regions)
            line_y = int(np.mean([
                np.mean([r.bbox[0][1], r.bbox[2][1]]) for r in line_regions
            ]))
            line_x = min(r.bbox[0][0] for r in line_regions)

            # Detect struck-through lines
            struck_through = any(r.annotation.has_strikethrough for r in line_regions)

            # Classification
            classification = self._classify_line(line_text, line_regions, line_idx, len(lines))

            # Parse based on classification
            if classification == "date":
                result.date, result.date_confidence = self._parse_date(line_text, line_regions)

            elif classification == "customer":
                result.customer = self._parse_customer(line_text, line_regions)

            elif classification == "item" or classification == "unknown":
                # Only parse as item if there are numeric values or product keywords
                item = self._parse_item_line(line_text, line_regions, struck_through)
                if item:
                    # Only add items that have some product-like text
                    if item.confidence > 0.1 or any(
                        kw in item.detected_text.lower() for kw in self.PRODUCT_KEYWORDS
                    ):
                        result.items.append(item)
                    else:
                        # Low-confidence line with no product keywords — flag as uncertain
                        result.uncertain.append(
                            f"Line {line_idx}: '{line_text[:30]}...' — unrecognized content"
                        )

        # Post-processing: handle quantity-only annotations (circled numbers)
        self._reconcile_quantities(result, lines)

        # Calculate overall confidence
        all_confs = [result.date_confidence]
        if result.customer:
            all_confs.append(result.customer.confidence)
        all_confs.extend(item.confidence for item in result.items)
        if all_confs:
            result.total_confidence = float(np.mean(all_confs))

        # Check for high uncertainty
        if len(result.uncertain) > (len(result.items) * 0.4 + 1):
            logger.warning(
                f"High uncertainty: {len(result.uncertain)} uncertain out of "
                f"{len(result.items)} items"
            )

        return result

    def _cluster_into_lines(self, regions: List[Region], y_tolerance: int = 60) -> List[List[Region]]:
        """
        Group regions into lines based on Y-coordinate proximity.
        Uses a simple clustering algorithm.
        """
        if not regions:
            return []

        lines = []
        current_line = [regions[0]]

        for region in regions[1:]:
            # Get Y ranges
            curr_y_center = np.mean([region.bbox[0][1], region.bbox[2][1]])
            line_y_centers = [
                np.mean([r.bbox[0][1], r.bbox[2][1]]) for r in current_line
            ]
            line_avg_y = np.mean(line_y_centers)
            curr_region_y = np.mean([region.bbox[0][1], region.bbox[2][1]])

            # If within tolerance, same line
            if abs(curr_region_y - line_avg_y) < y_tolerance:
                current_line.append(region)
            else:
                # Sort current line by X, then start new line
                current_line.sort(key=lambda r: r.bbox[0][0])
                lines.append(current_line)
                current_line = [region]

        current_line.sort(key=lambda r: r.bbox[0][0])
        lines.append(current_line)

        return lines

    def _classify_line(
        self, text: str, regions: List[Region], line_idx: int, total_lines: int
    ) -> str:
        """
        Classify a line as 'date', 'customer', 'item', or 'unknown'.

        Rules:
        - First 2 lines likely date/customer header
        - Lines containing day/month/year patterns -> date
        - Lines with single word + known customer pattern -> customer
        - Lines with product keywords + numbers -> item
        """
        lower_text = text.lower().strip()

        # --- Date detection ---
        # Pattern 1: DD MON YYYY ("02 Sep 2026", "28 Okt Kamis")
        # Pattern 2: DD/MM/YYYY or DD-MM-YYYY
        date_patterns = [
            r'\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|okt|nov|des|januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\s+\d{4}',
            r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',
            r'\b(kamis|senin|selasa|rabu|jumat|sabtu|minggu)\s+',
        ]
        for pattern in date_patterns:
            if re.search(pattern, lower_text, re.IGNORECASE):
                return "date"

        # --- Customer detection ---
        # Usually first/last in header section, short text (1-3 words)
        if line_idx <= 2:
            # Check if it looks like a name (no numbers, no product keywords)
            has_numbers = any(c.isdigit() for c in text)
            has_product = any(kw in lower_text for kw in self.PRODUCT_KEYWORDS)
            has_date_kw = any(d in lower_text for d in ["tanggal", "date", "tgl"])

            if not has_numbers and not has_product and not has_date_kw and len(lower_text) < 30:
                # Likely customer name
                words = lower_text.split()
                if 1 <= len(words) <= 4:
                    return "customer"

        # --- Item detection ---
        has_product = any(kw in lower_text for kw in self.PRODUCT_KEYWORDS)
        has_number = any(c.isdigit() for c in text)
        if has_product or (has_number and line_idx > 0):
            return "item"

        return "unknown"

    def _parse_date(self, text: str, regions: List[Region]) -> Tuple[Optional[str], float]:
        """Parse date from text line."""
        import re

        lower = text.lower().strip()

        # Pattern 1: DD MON YYYY or DD MM YYYY with day name
        # e.g., "08 Okt Kamis", "02 Sep 2026"
        m = re.search(r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|okt|nov|dec|januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)', lower)
        if m:
            day = int(m.group(1))
            month_abbr = m.group(2)
            month = self.MONTH_ABBR.get(month_abbr)
            if month:
                # Try to extract year (default to current year if not found)
                year_match = re.search(r'(\d{4})', text)
                year = int(year_match.group(1)) if year_match else None
                if year:
                    try:
                        return f"{year:04d}-{month:02d}-{day:02d}", 0.85
                    except ValueError:
                        pass

                # No year found — use current year with lower confidence
                from datetime import datetime
                year = datetime.now().year
                try:
                    return f"{year:04d}-{month:02d}-{day:02d}", 0.70
                except ValueError:
                    pass

        # Pattern 2: DD/MM/YYYY
        m = re.search(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})', text)
        if m:
            day = int(m.group(1))
            month = int(m.group(2))
            year = int(m.group(3))
            if year < 100:
                year += 2000
            try:
                return f"{year:04d}-{month:02d}-{day:02d}", 0.80
            except ValueError:
                pass

        # Pattern 3: Just day + month name (no year)
        # e.g., "08 September"
        m = re.search(r'(\d{1,2})\s+(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)', lower)
        if m:
            day = int(m.group(1))
            month = self.MONTH_ABBR.get(m.group(2))
            if month:
                from datetime import datetime
                year = datetime.now().year
                try:
                    return f"{year:04d}-{month:02d}-{day:02d}", 0.65
                except ValueError:
                    pass

        return None, 0.0

    def _parse_customer(self, text: str, regions: List[Region]) -> InterpretedCustomer:
        """Parse customer name from a line."""
        # Clean up text — only keep alphabetic part (remove OCR noise)
        words = [w for w in text.split() if w.isalpha() and len(w) > 1]
        clean_name = " ".join(words).strip()

        confidence = 0.7  # Medium — could be wrong (might be a note)

        # If text contains "plg" or "pelanggan", higher confidence
        lower = text.lower()
        if any(kw in lower for kw in ["plg", "pelanggan", "nama", "customer"]):
            confidence = 0.85
            # Remove the keyword from name
            for kw in ["plg", "pelanggan", "nama", "customer"]:
                clean_name = clean_name.replace(kw, "").strip()

        return InterpretedCustomer(
            detected_name=clean_name if clean_name else text.strip(),
            confidence=confidence,
            customer_candidates=[],  # Filled by Phase 4
        )

    def _parse_item_line(
        self, text: str, regions: List[Region], struck_through: bool
    ) -> Optional[InterpretedItem]:
        """
        Parse an item line and extract:
        - product name (first word/non-numeric part)
        - quantity (if circled or first small number)
        - price (if preceded by @, Rp, or is a large number)
        - structural symbols

        Rules for numbers:
        - Small number (1-99) near product name → likely quantity
        - Large number (1000+) preceded by @ or Rp → likely price
        - Standalone number → uncertain
        """
        if not regions:
            return None

        # Extract structural info from regions
        has_circled_number = any(r.annotation.has_circle for r in regions)
        has_at_symbol = False
        if regions and hasattr(regions[0].annotation, "symbols"):
            for r in regions:
                if r.annotation.symbols and any(
                    s.text == "@" for s in r.annotation.symbols
                ):
                    has_at_symbol = True
                    break

        # Parse numbers from the line
        numbers = []
        for r in regions:
            num_match = re.match(r'^(\d+)$', r.text.strip())
            if num_match:
                numbers.append((int(num_match.group(1)), r))

        # Classify numbers
        quantities = []
        prices = []

        for num_val, region in numbers:
            if has_circled_number and num_val <= 99:
                # Circled small number → quantity
                quantities.append(num_val)
            elif num_val >= 1000:
                # Large number → likely price
                prices.append(float(num_val))
            elif num_val <= 99 and has_circled_number:
                # Without circled marker, small number might be qty
                quantities.append(num_val)
            elif num_val <= 99 and len(numbers) == 1 and any(
                kw in text.lower() for kw in self.PRODUCT_KEYWORDS
            ):
                # Single number with product keyword → likely quantity
                quantities.append(num_val)
            # Large standalone number without context → price
            elif num_val >= 100:
                prices.append(float(num_val))

        # Extract product name: text regions that aren't numbers, structural chars, or noise
        name_parts = []
        uncertain_fields = []

        for r in regions:
            text_val = r.text.strip()
            if not text_val:
                continue
            # Skip pure numbers
            if re.match(r'^\d+$', text_val):
                continue
            # Skip structural symbols
            if text_val in self.STRUCTURAL_CHARS:
                continue
            # Skip single chars that might be OCR noise
            if len(text_val) == 1 and not text_val.isalpha():
                continue
            # Skip known date/day names
            if text_val.lower() in self.DAY_NAMES or text_val.lower() in self.MONTH_ABBR:
                continue

            name_parts.append(text_val)

        product_name = " ".join(name_parts).strip()

        # Determine quantity
        quantity = None
        if quantities:
            quantity = quantities[0]
        elif numbers and len(numbers) <= 1 and numbers[0][0] <= 99:
            # Small single number without context → assume quantity, flag uncertain
            quantity = numbers[0][0]
            uncertain_fields.append("quantity_assumed_no_explicit_marker")
        elif not numbers:
            uncertain_fields.append("no_quantity_detected")
        else:
            uncertain_fields.append("quantity_missing")

        # Determine price
        price = None
        if prices:
            price = prices[0]

        # Calculate item confidence
        text_conf = float(np.mean([r.confidence for r in regions])) if regions else 0
        has_product_keyword = any(
            kw in product_name.lower() for kw in self.PRODUCT_KEYWORDS
        )
        keyword_conf_bonus = 0.3 if has_product_keyword else 0
        struck_penalty = -0.3 if struck_through else 0
        item_confidence = min(1.0, text_conf + keyword_conf_bonus + struck_penalty)

        if struck_through:
            uncertain_fields.append("struck_through_excluded")

        return InterpretedItem(
            detected_text=product_name,
            quantity=quantity,
            price=price,
            unit="pcs",
            confidence=item_confidence,
            region_ids=[id(r) for r in regions],
            struck_through=struck_through,
            uncertain_fields=uncertain_fields,
            y_position=int(np.mean([
                np.mean([r.bbox[0][1], r.bbox[2][1]]) for r in regions
            ])) if regions else 0,
            x_position=min(r.bbox[0][0] for r in regions) if regions else 0,
        )

    def _reconcile_quantities(self, result: InterpretedResult, lines: List[List[Region]]):
        """
        Post-processing: reconcile quantities with potential duplicates
        or circled annotations that might be on previous lines.
        """
        pass  # Placeholder — implement based on real-world patterns

    def _is_noise_region(self, text: str, confidence: float) -> bool:
        """Check if a region is OCR noise (lines, symbols, artifacts)."""
        stripped = text.strip()
        if not stripped:
            return True
        # Pure drawing symbols
        if re.match(r'^[—–−|]+$', stripped):
            return True
        # Pure punctuation
        if re.match(r'^[\.,;:!?]+$', stripped):
            return True
        # Single non-alpha char with low confidence
        if len(stripped) == 1 and not stripped.isalpha() and confidence < 0.5:
            return True
        return False

    def _should_keep_region(self, region: Region) -> bool:
        """Determine if a region should be kept for interpretation."""
        text = region.text.strip()
        if self._is_noise_region(text, region.confidence):
            # But keep if it's a known product keyword
            if any(kw in text.lower() for kw in self.PRODUCT_KEYWORDS):
                return True
            return False

        # Keep numbers (may be qty or price)
        if re.match(r'^\d+$', text):
            return True

        # Keep product keywords
        if any(kw in text.lower() for kw in self.PRODUCT_KEYWORDS):
            return True

        # Keep date parts
        text_lower = text.lower()
        if any(d in text_lower for d in list(self.DAY_NAMES) + list(self.MONTH_ABBR.keys())):
            return True

        # Keep structural symbols
        if text in self.STRUCTURAL_CHARS:
            return True

        # Keep high-confidence regions
        if region.confidence > 0.4:
            return True

        # Keep alphabetic words > 1 char
        if len(text) > 1 and text.isalpha():
            return True

        return False

    def enhance_with_products(self, result: InterpretedResult, products: List[dict]) -> InterpretedResult:
        """
        Phase 3 integration: add product candidates to each item.
        Call this after interpretation if product DB is available.
        """
        from difflib import SequenceMatcher

        for item in result.items:
            item.product_candidates = self._match_products(item.detected_text, products)
            if item.product_candidates:
                top = item.product_candidates[0]
                if top["score"] > 0.85:
                    item.matched_product_id = top["product_id"]
                    item.matched_product_name = top["name"]
                else:
                    item.uncertain_fields.append("low_confidence_product_match")

        return result

    def enhance_with_customers(self, result: InterpretedResult, customers: List[dict]) -> InterpretedResult:
        """Phase 4 integration: add customer candidates."""
        from difflib import SequenceMatcher

        if result.customer and result.customer.detected_name:
            matches = []
            detected = result.customer.detected_name.lower()

            for cust in customers:
                cust_name = cust.get("name", "").lower()
                if not cust_name:
                    continue

                # Fuzzy match
                ratio = SequenceMatcher(None, detected, cust_name).ratio()

                # Substring match (e.g., "tami" matches "Mbak Tami")
                if detected in cust_name or cust_name in detected:
                    ratio = max(ratio, 0.9)

                if ratio > 0.4:  # Threshold
                    matches.append({
                        "customer_id": cust.get("id"),
                        "name": cust.get("name"),
                        "score": round(float(ratio), 4),
                    })

            matches.sort(key=lambda x: x["score"], reverse=True)
            result.customer.customer_candidates = matches[:3]  # Top 3

            if matches and matches[0]["score"] > 0.8:
                result.customer.matched_customer_id = matches[0]["customer_id"]
            elif matches:
                result.uncertain.append(
                    f"Customer '{result.customer.detected_name}' — low match ({matches[0]['score']:.2f})"
                )

        return result

    def _match_products(self, detected_text: str, products: List[dict]) -> List[dict]:
        """
        Phase 3 core: fuzzy match detected text against product names.
        Returns ranked list of candidates with scores.
        """
        from difflib import SequenceMatcher

        detected_lower = detected_text.lower().strip()
        matches = []

        for prod in products:
            prod_name = prod.get("name", "").lower()
            if not prod_name:
                continue

            # Try multiple matching strategies:
            # 1. Subsequence match (handles shorthand: "kopi" matches "Kopi")
            # 2. Levenshtein ratio
            # 3. Token set ratio (handles "kopi susu" vs "kopi")

            # Strategy 1: Subsequence
            subseq_score = self._subsequence_score(detected_lower, prod_name)

            # Strategy 2: Sequence matcher
            seq_score = SequenceMatcher(None, detected_lower, prod_name).ratio()

            # Strategy 3: Token-based
            det_tokens = set(detected_lower.split())
            prod_tokens = set(prod_name.split())
            if prod_tokens:
                token_score = len(det_tokens & prod_tokens) / len(prod_tokens)
            else:
                token_score = 0

            # Combine scores (weighted)
            final_score = max(subseq_score, seq_score * 0.8, token_score * 0.7)

            if final_score > 0.35:  # Threshold for candidates
                matches.append({
                    "product_id": prod.get("id"),
                    "name": prod.get("name"),
                    "score": round(float(final_score), 4),
                })

        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:5]  # Top 5 candidates

    def _subsequence_score(self, detected: str, target: str) -> float:
        """
        Check if detected text is a subsequence of target.
        Handles abbreviations: "k" matches "kopi", "kk" matches "kopi kopi".
        """
        # Normalize — remove spaces, lowercase
        d = detected.replace(" ", "").lower()
        t = target.replace(" ", "").lower()

        if not d:
            return 0.0

        # Check if d is a subsequence of t
        it = iter(t)
        if all(c in it for c in d):
            # Score based on length match — shorter = higher penalty
            # Perfect match = 1.0, length ratio matters
            ratio = len(d) / len(t) if len(t) > 0 else 0
            # Boost exact prefix matches
            if t.startswith(d):
                ratio = min(1.0, ratio + 0.2)
            return min(1.0, ratio)

        return 0.0
