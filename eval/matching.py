"""
Improved Product & Customer Matching — LANGKAH 6.

Extends eval/correct.py with:
1. Trigram similarity (character n-grams with Jaccard)
2. Phonetic matching for Indonesian (consonant skeleton + vowel normalization)
3. Alias dictionary (indomi→indomie, gls→gelas, etc.)
4. Price-based scoring boost

Usage:
    from eval.matching import ImprovedMatcher, INDONESIAN_ALIAS, build_matcher

    matcher = build_matcher(products, customers, price_map)
    match = matcher.match_product("kopi", price=15000)
    # -> {'name': 'Kopi', 'confidence': 0.95, 'method': 'exact'}
"""
import re
import logging
from difflib import SequenceMatcher
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


# --- Indonesian alias dictionary ---
# Maps OCR text variants → canonical product name
INDONESIAN_ALIAS = {
    # Noodles
    "indomi": "indomie",
    "mie instan": "indomie",
    "mie": "indomie",
    # Drinks
    "aqua": "aqua",
    "aqua glas": "aqua gelas",
    "aqua botol": "aqua botol",
    "gls": "aqua gelas",
    "teh": "teh",
    "teh manis": "teh manis",
    "kopi": "kopi",
    "kopi hitam": "kopi",
    "kopi susu": "kopi susu",
    # Snacks/Food
    "madu": "madu",
    "rokok": "rokok",
    "garam": "garam",
    "gula": "gula",
    "beras": "beras",
    "minyak": "minyak",
    "kecap": "kecap",
    # Warung-specific
    "makan": "makan",
    "donat": "donat",
    "gorengan": "gorengan",
    "es": "es",
    "es cream": "es cream",
}


def normalize_indonesian(text: str) -> str:
    """
    Normalize Indonesian text for matching.
    - Lowercase
    - Remove punctuation
    - Normalize spaces
    - Apply alias mapping
    """
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Apply alias mapping
    for alias, canonical in INDONESIAN_ALIAS.items():
        if alias in text:
            text = text.replace(alias, canonical)

    return text


def trigram_similarity(a: str, b: str) -> float:
    """
    Compute trigram (3-character n-gram) Jaccard similarity.
    Better for short strings with OCR noise than SequenceMatcher.
    """
    a_norm = a.lower().strip()
    b_norm = b.lower().strip()

    if a_norm == b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0

    def get_trigrams(s: str) -> Counter:
        if len(s) < 3:
            # Pad short strings
            s = s + "  "[:3 - len(s)]
        grams = [s[i:i+3] for i in range(len(s) - 2)]
        return Counter(grams)

    tg_a = get_trigrams(a_norm)
    tg_b = get_trigrams(b_norm)

    intersection = sum((tg_a & tg_b).values())
    union = sum((tg_a | tg_b).values())

    if union == 0:
        return 0.0

    return intersection / union


def phonetic_match(a: str, b: str) -> float:
    """
    Phonetic similarity for Indonesian.
    Normalizes:
    - Vowel variations (a/e/i/o/u all treated as same)
    - Consonant confusions (kh→k, sy→s, ch→c, ng→n)
    Returns similarity score [0, 1].
    Requires consonant skeleton match AND similar length.
    """
    def normalize_phonetic(s: str) -> str:
        s = s.lower().strip()
        # Vowel normalization: treat all vowels as 'a'
        s = re.sub(r'[aeiou]', 'a', s)
        # Normalize common Indonesian phonetic confusions
        replacements = [
            ('kh', 'k'), ('sy', 's'), ('ch', 'c'),
            ('ng', 'n'), ('ny', 'n'), ('gh', 'g'),
            ('ts', 'c'), ('dj', 'j'),
        ]
        for old, new in replacements:
            s = s.replace(old, new)
        return re.sub(r'[^a-z]', '', s)  # keep only alphabet

    a_norm = normalize_phonetic(a)
    b_norm = normalize_phonetic(b)

    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0

    # Require consonant skeleton to match (high SequenceMatcher)
    seq_ratio = SequenceMatcher(None, a_norm, b_norm).ratio()

    # Length check: if words are very different length, discount
    len_ratio = min(len(a_norm), len(b_norm)) / max(len(a_norm), len(b_norm))

    # Combined score
    return seq_ratio * len_ratio


def enhanced_fuzzy_match(text: str, candidates: list[str],
                         price: int = None, price_map: dict = None,
                         threshold: float = 0.5) -> tuple[str, float, str]:
    """
    Enhanced fuzzy matching with:
    - Trigram similarity
    - Phonetic matching
    - Alias dictionary lookup
    - Price-based score boost
    - Substring matching (partial)

    Args:
        text: OCR-detected text to match
        candidates: list of canonical names
        price: detected price (if any)
        price_map: dict of {product_name: price} for price-based boosting
        threshold: minimum confidence to return a match

    Returns:
        (matched_name, confidence, method_used) or ("", 0.0, "no_match")
    """
    if not text or not candidates:
        return "", 0.0, "no_match"

    text = text.strip()
    text_norm = normalize_indonesian(text)

    best_match = ""
    best_score = 0.0
    best_method = "no_match"

    for cand in candidates:
        cand_norm = normalize_indonesian(cand)

        # Method 1: Exact (after normalization + alias)
        if text_norm == cand_norm:
            score = 1.0
            if score > best_score:
                return cand, score, "exact"

        # Method 2: Substring (cand in text or vice versa)
        if len(cand_norm) >= 2:
            if cand_norm in text_norm:
                score = 0.9
                if score > best_score:
                    best_score = score
                    best_match = cand
                    best_method = "substring"
            elif text_norm in cand_norm:
                score = 0.85
                if score > best_score:
                    best_score = score
                    best_match = cand
                    best_method = "substring"

        # Method 3: Trigram similarity
        tri_score = trigram_similarity(text_norm, cand_norm)
        if tri_score > best_score:
            # Boost short words (less noise impact)
            if len(cand_norm) <= 5 and tri_score > 0.4:
                tri_score = min(1.0, tri_score + 0.2)
            best_score = tri_score
            best_match = cand
            best_method = "trigram"

        # Method 4: Phonetic matching (strict — consonant skeleton must match)
        ph_score = phonetic_match(text_norm, cand_norm)
        if ph_score > best_score:
            # Phonetic: require high confidence (consonant skeleton nearly identical)
            if ph_score > 0.90:
                best_score = ph_score * 0.9  # small discount
                best_match = cand
                best_method = "phonetic"

        # Method 5: SequenceMatcher fallback
        if not best_match or best_score < 0.5:
            seq_score = SequenceMatcher(None, text_norm, cand_norm).ratio()
            if seq_score > best_score:
                best_score = seq_score
                best_match = cand
                best_method = "levenshtein"

    # Method 6: Price-based boost
    if price is not None and price_map is not None and best_match:
        if best_match in price_map:
            cand_price = price_map[best_match]
            if cand_price > 0:
                price_diff = abs(price - cand_price) / cand_price
                if price_diff < 0.15:  # within 15% of expected price
                    score_boost = 0.15
                    best_score = min(1.0, best_score + score_boost)
                    best_method = f"{best_method}+price"
                    logger.info(f"Price boost: {price} vs {cand_price} (diff={price_diff:.1%})")

    # Threshold check
    if best_score >= threshold:
        return best_match, min(1.0, best_score), best_method
    else:
        return "", 0.0, "no_match"


class ImprovedMatcher:
    """
    Improved matcher with trigram + phonetic + alias + price scoring.
    """

    def __init__(self, product_catalog: list[dict], customer_catalog: list[dict]):
        self.product_catalog = product_catalog
        self.customer_catalog = customer_catalog
        self.product_names = [p["name"] for p in product_catalog]
        self.customer_names = [c["name"] for c in customer_catalog]

        # Build price map (product_name -> price)
        self.price_map = {}
        for p in product_catalog:
            if "default_price" in p and p["default_price"]:
                self.price_map[p["name"].lower()] = p["default_price"]

        # Build alias-expanded product list
        self.alias_dict = {}
        for alias, canonical in INDONESIAN_ALIAS.items():
            if canonical in [p["name"].lower() for p in product_catalog]:
                self.alias_dict[alias] = canonical
            elif canonical in self.alias_dict:
                self.alias_dict[alias] = self.alias_dict[canonical]

        self.corrections = []

    def match_product(self, ocr_text: str, price: int = None) -> tuple[str, float, str]:
        """
        Match OCR text to a product with price context.
        Returns (matched_name, confidence, method).
        """
        match, score, method = enhanced_fuzzy_match(
            ocr_text, self.product_names,
            price=price, price_map=self.price_map,
            threshold=0.55
        )

        if match and method != "exact":
            self.corrections.append({
                "type": "product_match",
                "original": ocr_text,
                "matched": match,
                "confidence": score,
                "method": method,
            })

        return match, score, method

    def match_customer(self, ocr_text: str) -> tuple[str, float, str]:
        """Match OCR text to a customer name."""
        match, score, method = enhanced_fuzzy_match(
            ocr_text, self.customer_names,
            threshold=0.55
        )

        if match and method != "exact":
            self.corrections.append({
                "type": "customer_match",
                "original": ocr_text,
                "matched": match,
                "confidence": score,
                "method": method,
            })

        return match, score, method

    def get_corrections(self) -> list[dict]:
        return list(self.corrections)

    def reset(self):
        self.corrections = []


def build_matcher(product_catalog: list[dict], customer_catalog: list[dict]) -> ImprovedMatcher:
    """Build an ImprovedMatcher from product/customer catalogs."""
    return ImprovedMatcher(product_catalog, customer_catalog)


# --- Quick test ---
if __name__ == "__main__":
    # Test trigram matching
    test_cases = [
        ("kopi", "Kopi", "exact"),
        ("kop", "Kopi", "close match"),
        ("kupi", "Kopi", "vowel swap"),
        ("kapa", "Kapi", "phonetic"),
        ("indomi", "Indomie", "alias"),
        ("madu", "Madu", "exact"),
        ("marco", "Marco", "exact"),
        ("marco", "Rama", "no match"),
    ]

    print("=== Enhanced Matching Test ===")
    products = [{"id": 1, "name": "Kopi", "default_price": 5000},
                {"id": 2, "name": "Teh", "default_price": 4000},
                {"id": 3, "name": "Indomie", "default_price": 8000},
                {"id": 4, "name": "Madu", "default_price": 15000}]
    customers = [{"id": 1, "name": "Marco"}, {"id": 2, "name": "Rama"}]

    matcher = ImprovedMatcher(products, customers)

    for ocr_text, expected, desc in test_cases:
        # Try product first
        match, score, method = matcher.match_product(ocr_text)
        if not match:
            match, score, method = matcher.match_customer(ocr_text)
        # "marco" should match customer "Marco" not product
        if "no match" in desc:
            # Verify it does NOT match something wrong
            if desc == "no match":
                actual_match = match if match else "none"
                status = "✓" if match == "" else "✗"
                print(f"  {status} '{ocr_text}' → '{actual_match}' [{desc}]")
            else:
                status = "✓" if match.lower() == expected.lower() else "✗"
                print(f"  {status} '{ocr_text}' → '{match}' (conf={score:.2f}, method={method}) [{desc}]")
        else:
            status = "✓" if match.lower() == expected.lower() else "✗"
            print(f"  {status} '{ocr_text}' → '{match}' (conf={score:.2f}, method={method}) [{desc}]")
