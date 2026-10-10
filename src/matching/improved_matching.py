"""
Phase 3 & Phase 4: Improved Product & Customer Matching Engine.

Implements multi-strategy matching for Indonesian warung receipts:
1. Alias normalization (indomi → indomie, gls → aqua gelas, etc.)
2. Trigram similarity (character n-grams with Jaccard)
3. Phonetic matching (consonant skeleton + vowel normalization for Indonesian)
4. Substring & exact matching
5. Price-based confidence boosting
6. Candidate ranking for UI review

Usage:
    from src.matching.improved_matching import ImprovedMatcher, build_matcher

    matcher = build_matcher(products, customers)
    matched_name, conf, method = matcher.match_product("kopi", price=5000)
    candidates = matcher.rank_products("kop", top_k=5)
"""

import re
import logging
from difflib import SequenceMatcher
from collections import Counter
from typing import Optional, List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

# --- Indonesian alias dictionary ---
# Maps OCR text variants → canonical product names
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
    "t jus": "teh jus",
    "es t": "es teh",
    # Snacks / Food
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
    "es batu": "es batu",
}


def normalize_indonesian(text: str) -> str:
    """
    Normalize Indonesian text for matching:
    - Lowercase
    - Remove punctuation
    - Normalize spaces
    - Apply alias mapping
    """
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Sort aliases by length descending so longer phrases match first
    sorted_aliases = sorted(INDONESIAN_ALIAS.items(), key=lambda x: len(x[0]), reverse=True)
    for alias, canonical in sorted_aliases:
        pattern = r'\b' + re.escape(alias) + r'\b'
        text = re.sub(pattern, canonical, text)

    return re.sub(r'\s+', ' ', text).strip()


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
            s = s + "  "[:3 - len(s)]
        grams = [s[i:i + 3] for i in range(len(s) - 2)]
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
    - Vowels (all treated as 'a')
    - Consonant confusions (kh→k, sy→s, ch→c, ng→n, etc.)
    """
    def normalize_phonetic(s: str) -> str:
        s = s.lower().strip()
        replacements = [
            ('kh', 'k'), ('sy', 's'), ('ch', 'c'),
            ('ng', 'n'), ('ny', 'n'), ('gh', 'g'),
            ('ts', 'c'), ('dj', 'j'),
        ]
        for old, new in replacements:
            s = s.replace(old, new)
        s = re.sub(r'[^a-z]', '', s)
        if not s:
            return ""
        # Consonant skeleton: keep first letter, drop subsequent vowels
        first = s[0]
        rest = re.sub(r'[aeiou]', '', s[1:])
        return first + rest

    a_norm = normalize_phonetic(a)
    b_norm = normalize_phonetic(b)

    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0

    seq_ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    len_ratio = min(len(a_norm), len(b_norm)) / max(len(a_norm), len(b_norm))
    return seq_ratio * len_ratio


def score_candidate(
    text: str,
    cand: str,
    price: Optional[int] = None,
    expected_price: Optional[int] = None,
) -> Tuple[float, str]:
    """Calculate similarity score and method between text and candidate."""
    text_norm = normalize_indonesian(text)
    cand_norm = normalize_indonesian(cand)

    if not text_norm or not cand_norm:
        return 0.0, "no_match"

    # 1. Exact
    if text_norm == cand_norm:
        score = 1.0
        method = "exact"
        return score, method

    # 2. Substring
    best_score = 0.0
    best_method = "no_match"
    if len(cand_norm) >= 2:
        if cand_norm in text_norm:
            best_score = 0.90
            best_method = "substring"
        elif text_norm in cand_norm:
            best_score = 0.85
            best_method = "substring"

    # 3. Trigram
    tri_score = trigram_similarity(text_norm, cand_norm)
    if len(cand_norm) <= 5 and tri_score > 0.4:
        tri_score = min(1.0, tri_score + 0.2)
    if tri_score > best_score:
        best_score = tri_score
        best_method = "trigram"

    # 4. Phonetic
    ph_score = phonetic_match(text_norm, cand_norm)
    if ph_score > 0.90:
        ph_eff = ph_score * 0.90
        if ph_eff > best_score:
            best_score = ph_eff
            best_method = "phonetic"

    # 5. Levenshtein fallback
    if best_score < 0.5:
        seq_score = SequenceMatcher(None, text_norm, cand_norm).ratio()
        if seq_score > best_score:
            best_score = seq_score
            best_method = "levenshtein"

    # 6. Price boost
    if price is not None and expected_price is not None and expected_price > 0:
        diff = abs(price - expected_price) / expected_price
        if diff < 0.15:
            best_score = min(1.0, best_score + 0.15)
            best_method = f"{best_method}+price"

    return min(1.0, best_score), best_method


def enhanced_fuzzy_match(
    text: str,
    candidates: List[str],
    price: Optional[int] = None,
    price_map: Optional[Dict[str, int]] = None,
    threshold: float = 0.55,
) -> Tuple[str, float, str]:
    """
    Enhanced fuzzy matching against candidate names.
    Returns (best_match_name, score, method).
    """
    if not text or not candidates:
        return "", 0.0, "no_match"

    best_match = ""
    best_score = 0.0
    best_method = "no_match"

    for cand in candidates:
        expected_price = price_map.get(cand.lower()) if price_map else None
        score, method = score_candidate(text, cand, price=price, expected_price=expected_price)
        if score > best_score:
            best_score = score
            best_match = cand
            best_method = method
            if score == 1.0 and method == "exact":
                break

    if best_score >= threshold:
        return best_match, round(best_score, 4), best_method
    return "", 0.0, "no_match"


class ImprovedMatcher:
    """
    Core matcher for Product and Customer databases.
    Implements multi-strategy ranking with trigram, phonetic, and price features.
    """

    def __init__(
        self,
        product_catalog: Optional[List[Dict[str, Any]]] = None,
        customer_catalog: Optional[List[Dict[str, Any]]] = None,
    ):
        self.product_catalog = product_catalog or []
        self.customer_catalog = customer_catalog or []
        self.product_names = [p["name"] for p in self.product_catalog if "name" in p]
        self.customer_names = [c["name"] for c in self.customer_catalog if "name" in c]

        self.price_map: Dict[str, int] = {}
        for p in self.product_catalog:
            p_name = p.get("name", "")
            p_price = p.get("default_price") or p.get("price")
            if p_name and p_price:
                try:
                    self.price_map[p_name.lower()] = int(p_price)
                except (ValueError, TypeError):
                    pass

        self.corrections: List[Dict[str, Any]] = []

    def match_product(self, ocr_text: str, price: Optional[int] = None) -> Tuple[str, float, str]:
        """Match OCR text to a single best product candidate."""
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

    def match_customer(self, ocr_text: str) -> Tuple[str, float, str]:
        """Match OCR text to a single best customer candidate."""
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

    def rank_products(
        self,
        ocr_text: str,
        price: Optional[int] = None,
        top_k: int = 5,
        min_threshold: float = 0.30,
    ) -> List[Dict[str, Any]]:
        """Rank all products against ocr_text and return top-k candidates."""
        candidates = []
        for prod in self.product_catalog:
            p_name = prod.get("name", "")
            if not p_name:
                continue
            expected_price = self.price_map.get(p_name.lower())
            score, method = score_candidate(ocr_text, p_name, price=price, expected_price=expected_price)
            if score >= min_threshold:
                candidates.append({
                    "product_id": prod.get("id"),
                    "name": p_name,
                    "score": round(score, 4),
                    "method": method,
                    "default_price": expected_price,
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def rank_customers(
        self,
        ocr_text: str,
        top_k: int = 3,
        min_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """Rank all customers against ocr_text and return top-k candidates."""
        candidates = []
        for cust in self.customer_catalog:
            c_name = cust.get("name", "")
            if not c_name:
                continue
            score, method = score_candidate(ocr_text, c_name)
            if score >= min_threshold:
                candidates.append({
                    "customer_id": cust.get("id"),
                    "name": c_name,
                    "score": round(score, 4),
                    "method": method,
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def get_corrections(self) -> List[Dict[str, Any]]:
        return list(self.corrections)

    def reset(self) -> None:
        self.corrections = []


def build_matcher(
    product_catalog: List[Dict[str, Any]],
    customer_catalog: List[Dict[str, Any]],
) -> ImprovedMatcher:
    """Factory helper to build an ImprovedMatcher."""
    return ImprovedMatcher(product_catalog, customer_catalog)
