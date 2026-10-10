"""
Improved Product & Customer Matching — LANGKAH 6.

Canonical implementation is located in src/matching/improved_matching.py.
This module re-exports components for eval harness and scripts.
"""

from src.matching.improved_matching import (
    INDONESIAN_ALIAS,
    normalize_indonesian,
    trigram_similarity,
    phonetic_match,
    enhanced_fuzzy_match,
    ImprovedMatcher,
    build_matcher,
)

__all__ = [
    "INDONESIAN_ALIAS",
    "normalize_indonesian",
    "trigram_similarity",
    "phonetic_match",
    "enhanced_fuzzy_match",
    "ImprovedMatcher",
    "build_matcher",
]

if __name__ == "__main__":
    products = [
        {"id": 1, "name": "Kopi", "default_price": 5000},
        {"id": 2, "name": "Teh", "default_price": 4000},
        {"id": 3, "name": "Indomie", "default_price": 8000},
        {"id": 4, "name": "Madu", "default_price": 15000},
    ]
    customers = [{"id": 1, "name": "Marco"}, {"id": 2, "name": "Rama"}]

    matcher = ImprovedMatcher(products, customers)
    test_cases = [
        ("kopi", "Kopi", "exact"),
        ("kop", "Kopi", "close match"),
        ("kupi", "Kopi", "vowel swap"),
        ("indomi", "Indomie", "alias"),
        ("madu", "Madu", "exact"),
        ("marco", "Marco", "exact"),
        ("marco", "Rama", "no match"),
    ]

    print("=== Enhanced Matching Test ===")
    for ocr_text, expected, desc in test_cases:
        match, score, method = matcher.match_product(ocr_text)
        if not match:
            match, score, method = matcher.match_customer(ocr_text)
        if desc == "no match":
            status = "✓" if match == "" else "✗"
            print(f"  {status} '{ocr_text}' → '{match or 'none'}' [{desc}]")
        else:
            status = "✓" if match.lower() == expected.lower() else "✗"
            print(f"  {status} '{ocr_text}' → '{match}' (conf={score:.2f}, method={method}) [{desc}]")
