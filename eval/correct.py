"""
Post-OCR Correction Module — Langkah 5.

Applies domain-aware corrections AFTER OCR raw text is extracted:
1. Character confusion map (handwriting-specific substitutions)
2. Lexicon matching (products + customers from catalog)
3. Date parsing (Indonesian date formats → YYYY-MM-DD)
4. Price/qty notation parser (3rb, 3.500, x2, @3500, etc.)
5. Noise filtering (remove garbage lines)

Usage:
    from eval.correct import PostOCRCorrector

    corrector = PostOCRCorrector(product_catalog, customer_catalog)
    corrected_text = corrector.correct(raw_text, scan_result)
    corrections = corrector.get_corrections_log()  # for audit trail
"""
import re
import logging
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)

# --- Character confusion map for Indonesian handwriting ---
# Maps confused OCR chars → possible corrections based on handwriting similarity
CHAR_CONFUSION = {
    # Zero / O / D
    '0': ['O', 'D'],
    'O': ['0', 'D'],
    'D': ['0', 'O'],
    # One / l / I / /
    '1': ['l', 'I', '/'],
    'l': ['1', 'I', '/'],
    'I': ['1', 'l', '/'],
    # Five / S
    '5': ['S'],
    'S': ['5'],
    # Two / Z
    '2': ['Z'],
    'Z': ['2'],
    # Six / G / b
    '6': ['G', 'b', 'g'],
    'G': ['6'],
    # Nine / g / q
    '9': ['g', 'q'],
    'g': ['9', 'q'],
    'q': ['9', 'g'],
    # Eight / B
    '8': ['B'],
    'B': ['8'],
    # Three / E
    '3': ['E'],
    'E': ['3'],
    # Four / A
    '4': ['A'],
    'A': ['4'],
    # Six / b
    'b': ['6'],
    'd': ['0'],  # lowercase d mistaken for 0
    # Seven / T
    '7': ['T'],
    'T': ['7'],
    # U / V
    'U': ['V'],
    'V': ['U'],
}


def levenshtein(a: str, b: str) -> int:
    """Fast Levenshtein distance for short strings."""
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        curr = [i + 1]
        for j, c2 in enumerate(b):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (0 if c1 == c2 else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def fuzzy_match(text: str, candidates: list[str], threshold: float = 0.6) -> tuple[str, float]:
    """
    Find best fuzzy match for text in candidate list.
    Returns (best_match, similarity_ratio) or ("", 0.0) if below threshold.
    """
    text_norm = text.lower().strip()
    best = ""
    best_ratio = 0.0
    for cand in candidates:
        cand_norm = cand.lower().strip()
        ratio = SequenceMatcher(None, text_norm, cand_norm).ratio()
        # Also try substring match (OCR text might have extra chars)
        if cand_norm in text_norm:
            ratio = max(ratio, 0.85)
        if ratio > best_ratio:
            best_ratio = ratio
            best = cand
    if best_ratio >= threshold:
        return best, best_ratio
    return "", 0.0


def correct_character_confusion(text: str, context: str = "mixed") -> str:
    """
    Apply character confusion corrections based on context.
    context: "product" (favor letters), "price" (favor digits), "mixed" (balanced)
    """
    result = text
    for conf, alternates in CHAR_CONFUSION.items():
        for alt in alternates:
            # Skip digit corrections in product context, skip letter corrections in price context
            if context == "price" and not conf.isdigit():
                continue
            if context == "product" and conf.isdigit() and alt.isalpha():
                continue
            result = result.replace(alt, conf)
    return result


def parse_indonesian_date(text: str) -> Optional[str]:
    """
    Parse Indonesian date formats into YYYY-MM-DD.

    Supported formats:
    - 08/10, 08-10, 08.10
    - 08/10/2026, 08-10-26
    - 8 Okt, 8 Oktober, 8 Okt 2026
    - Kamis 8 Oktober 2026
    """
    # Day/month patterns: 8/10, 08.10, 08-10
    m = re.search(r'(\d{1,2})[/.-](\d{1,2})(?:[/.-](\d{2,4}))?', text)
    if m:
        day = int(m.group(1))
        month = int(m.group(2))
        year_str = m.group(3)
        if year_str:
            year = int(year_str)
            if year < 100:
                year += 2000
        else:
            year = datetime.now().year
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"

    # Indonesian month names
    month_map = {
        'jan': 1, 'januari': 1,
        'feb': 2, 'februari': 2,
        'mar': 3, 'maret': 3,
        'apr': 4, 'april': 4,
        'mei': 5,
        'jun': 6, 'juni': 6,
        'jul': 7, 'juli': 7,
        'ags': 8, 'agustus': 8, 'ag': 8,
        'sep': 9, 'september': 9,
        'okt': 10, 'oktober': 10, 'ok': 10,
        'nov': 11, 'november': 11,
        'des': 12, 'desember': 12,
    }
    # Format: 8 Okt, 8 Oktober, 8 Okt 2026
    m = re.search(r'(\d{1,2})[,\s]+(jan|feb|mar|apr|mei|jun|jul|ags|sep|okt|nov|des|januari|februari|maret|april|agustus|september|oktober|november|desember)', text, re.IGNORECASE)
    if m:
        day = int(m.group(1))
        month = month_map.get(m.group(2).lower(), 0)
        if 1 <= month <= 12 and 1 <= day <= 31:
            year = datetime.now().year
            return f"{year:04d}-{month:02d}-{day:02d}"

    # Try to extract just the date numbers
    m = re.search(r'(\d{1,2})', text)
    return None


def parse_price(text: str) -> Optional[int]:
    """
    Parse Indonesian price notation.

    Supported:
    - 3.500, 3,500 → 3500
    - 3500 → 3500
    - 3rb → 3000
    - 3,5rb → 3500
    - 3k → 3000
    - Rp3500, 3500,-
    - 35 (if context suggests thousand)
    """
    text = text.strip()

    # Handle . as thousand separator: 3.500, 3,500
    m = re.search(r'(\d{1,3})(?:[.,])(\d{3})(?:\s*,?-)?', text)
    if m:
        return int(m.group(1) + m.group(2))

    # Handle decimal with rb unit: "3,5rb" -> 3500
    m = re.search(r'(\d+)[.,](\d+)\s*(?:rb|ribu)', text, re.IGNORECASE)
    if m:
        return int(float(m.group(1) + '.' + m.group(2)) * 1000)

    # Handle integer with rb unit: "3rb", "3ribu", "3k" -> 3000
    m = re.search(r'(\d+)\s*(?:rb|ribu|k)\b', text, re.IGNORECASE)
    if m:
        return int(m.group(1)) * 1000

    # Handle single number
    m = re.search(r'(\d+)', text)
    if m:
        return int(m.group(1))

    return None


def parse_quantity(text: str) -> Optional[int]:
    """
    Parse quantity notation.

    Supported:
    - 2x, x2 → 2
    - (2) → 2
    - 2 → 2 (if small number in product context)
    """
    text = text.strip()
    # x2, 2x
    m = re.search(r'(\d+)\s*[xX]|\s*[xX]\s*(\d+)', text)
    if m:
        qty = int(m.group(1) or m.group(2))
        if qty < 100:  # reasonable quantity
            return qty

    # (2) notation
    m = re.search(r'\((\d+)\)', text)
    if m:
        return int(m.group(1))

    # Single small number
    m = re.search(r'(\d+)', text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 50:
            return val

    return None


class PostOCRCorrector:
    """
    Corrects OCR output using domain knowledge from the Warung Lupi context.
    """

    def __init__(self, product_catalog: list[dict], customer_catalog: list[dict]):
        """
        Args:
            product_catalog: list of {"id": int, "name": str, ...}
            customer_catalog: list of {"id": int, "name": str, ...}
        """
        self.product_catalog = product_catalog
        self.customer_catalog = customer_catalog
        self.product_names = [p["name"] for p in product_catalog]
        self.customer_names = [c["name"] for c in customer_catalog]
        self.corrections = []

    def reset_log(self):
        self.corrections = []

    def get_corrections_log(self) -> list[dict]:
        return list(self.corrections)

    def correct_product_name(self, ocr_text: str) -> tuple[str, float]:
        """
        Match OCR text to product name. Returns (matched_name, confidence).
        Tries: exact, substring, fuzzy (with confusion map pre-correction).
        """
        text = ocr_text.strip()
        if not text:
            return "", 0.0

        # Try exact match first (case-insensitive)
        for name in self.product_names:
            if text.lower() == name.lower():
                return name, 1.0
            if name.lower() in text.lower():
                return name, 0.95

        # Try with character confusion correction
        corrected = correct_character_confusion(text, context="product")
        if corrected != text:
            for name in self.product_names:
                if corrected.lower() == name.lower():
                    self.corrections.append({
                        "type": "product_correction",
                        "original": text,
                        "corrected": name,
                        "method": "character_confusion",
                    })
                    return name, 0.9

        # Fuzzy match
        match, ratio = fuzzy_match(text, self.product_names, threshold=0.55)
        if match:
            self.corrections.append({
                "type": "product_correction",
                "original": text,
                "corrected": match,
                "method": "fuzzy_match",
                "confidence": ratio,
            })
            return match, ratio

        return "", 0.0

    def correct_customer_name(self, ocr_text: str) -> tuple[str, float]:
        """Match OCR text to customer name."""
        text = ocr_text.strip()

        for name in self.customer_names:
            if text.lower() == name.lower():
                return name, 1.0
            if name.lower() in text.lower():
                return name, 0.95

        corrected = correct_character_confusion(text, context="product")
        match, ratio = fuzzy_match(text, self.customer_names, threshold=0.55)
        if match:
            self.corrections.append({
                "type": "customer_correction",
                "original": text,
                "corrected": match,
                "method": "fuzzy_match",
                "confidence": ratio,
            })
            return match, ratio

        return "", 0.0

    def extract_prices(self, text: str) -> list[int]:
        """Extract all plausible prices from text."""
        prices = []
        # Match price patterns: Rp3500, 3.500, 3500, 35rb, 3k
        patterns = [
            r'Rp\s*(\d{1,3}(?:[.,]\d{3})+)',  # Rp3.500
            r'(\d{1,3}(?:[.,]\d{3})+)',  # 3.500
            r'(\d+)(?:rb|ribu)',  # 3rb
            r'(\d+)k\b',  # 3k
            r'(\d{3,6})(?=\s*(?:,-|$)|\s*(?=\d{3}(?:\D|$)))',  # 3500
        ]
        for pat in patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches:
                price = parse_price(str(m))
                if price and 500 <= price <= 500000:
                    prices.append(price)
        return list(set(prices))

    def extract_quantities(self, text: str) -> list[int]:
        """Extract all plausible quantities from text."""
        qtys = []
        # x2, 2x patterns
        matches = re.findall(r'(\d+)x\b|\bx(\d+)', text)
        for m in matches:
            q = int(m[0] or m[1])
            if 1 <= q <= 99:
                qtys.append(q)
        # (2) notation
        for m in re.findall(r'\((\d+)\)', text):
            q = int(m)
            if 1 <= q <= 99:
                qtys.append(q)
        return list(set(qtys))

    def filter_noise_lines(self, text: str) -> str:
        """Remove lines that are just noise/symbols."""
        lines = text.split('\n')
        filtered = []
        for line in lines:
            strip = line.strip()
            if not strip:
                continue
            # Remove lines that are just symbols
            if re.match(r'^[\s—\-_|+*~`.]+$', strip):
                continue
            # Remove lines with no alphanumeric content
            if not re.search(r'[a-zA-Z0-9]', strip):
                continue
            # Remove very short lines that are likely noise
            if len(strip) < 2:
                continue
            filtered.append(line)
        return '\n'.join(filtered)

    def correct_text(self, raw_text: str) -> dict:
        """
        Full correction pipeline on raw OCR text.
        Returns dict with corrected text and corrections list.
        """
        self.reset_log()

        # Step 1: Filter noise lines
        filtered = self.filter_noise_lines(raw_text)
        if len(filtered) != len(raw_text):
            self.corrections.append({
                "type": "noise_filter",
                "original_lines": raw_text.count('\n') + 1,
                "filtered_lines": filtered.count('\n') + 1,
            })

        # Step 2: Extract date
        date = parse_indonesian_date(filtered)
        if date:
            self.corrections.append({
                "type": "date_extraction",
                "value": date,
            })

        result = {
            "corrected_text": filtered,
            "date": date,
            "corrections": self.get_corrections_log(),
        }

        return result
