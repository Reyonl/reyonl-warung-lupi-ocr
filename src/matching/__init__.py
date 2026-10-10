"""
Matching module exports.
"""

from src.matching.improved_matching import (
    ImprovedMatcher,
    build_matcher,
    INDONESIAN_ALIAS,
    normalize_indonesian,
    enhanced_fuzzy_match,
    trigram_similarity,
    phonetic_match,
)

__all__ = [
    "ImprovedMatcher",
    "build_matcher",
    "INDONESIAN_ALIAS",
    "normalize_indonesian",
    "enhanced_fuzzy_match",
    "trigram_similarity",
    "phonetic_match",
]
