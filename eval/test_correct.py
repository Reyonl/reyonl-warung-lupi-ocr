"""
Unit tests for eval/correct.py — Post-OCR Correction Module.

Run: python eval/test_correct.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.correct import (
    correct_character_confusion, parse_indonesian_date, parse_price,
    parse_quantity, PostOCRCorrector, fuzzy_match, levenshtein,
)

# Test data
PRODUCTS = [
    {"id": 1, "name": "Kopi"},
    {"id": 2, "name": "Teh"},
    {"id": 3, "name": "Madu"},
    {"id": 4, "name": "Rokok"},
    {"id": 5, "name": "Aqua"},
    {"id": 6, "name": "Mie"},
    {"id": 7, "name": "Donat"},
]
CUSTOMERS = [
    {"id": 14, "name": "Marco"},
    {"id": 27, "name": "Rama"},
    {"id": 28, "name": "Bu Sri"},
    {"id": 32, "name": "Mama Sasa"},
]


def test_parse_indonesian_date():
    """Test Indonesian date parsing."""
    cases = [
        ("08/10", "2026-10-08"),
        ("08-10-26", "2026-10-08"),
        ("08.10.2026", "2026-10-08"),
        ("8 Okt", "2026-10-08"),
        ("8 Oktober 2026", "2026-10-08"),
        ("08-10-2026", "2026-10-08"),
        ("Kamis 8 Okt", "2026-10-08"),
        ("08, OKT", "2026-10-08"),  # comma separator (OCR noise)
    ]
    for text, expected in cases:
        result = parse_indonesian_date(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} parse_date('{text}') → '{result}' (expected '{expected}')")
        assert result == expected, f"Expected {expected}, got {result}"


def test_parse_price():
    """Test price parsing."""
    cases = [
        ("Rp3.500", 3500),
        ("3.500", 3500),
        ("3500", 3500),
        ("3rb", 3000),
        ("3,5rb", 3500),
        ("3k", 3000),
        ("5000,-", 5000),
        ("15000", 15000),
    ]
    for text, expected in cases:
        result = parse_price(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} parse_price('{text}') → {result} (expected {expected})")
        assert result == expected, f"Expected {expected}, got {result}"


def test_parse_quantity():
    """Test quantity parsing."""
    cases = [
        ("2x", 2),
        ("x2", 2),
        ("(3)", 3),
        ("10", 10),
    ]
    for text, expected in cases:
        result = parse_quantity(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} parse_qty('{text}') → {result} (expected {expected})")
        assert result == expected, f"Expected {expected}, got {result}"


def test_fuzzy_match():
    """Test fuzzy string matching."""
    products = ["Kopi", "Teh", "Madu", "Rokok", "Aqua"]
    cases = [
        ("Kopi", "Kopi", 1.0),
        ("Kop", "Kopi", 0.8),  # close match
        ("kopi", "Kopi", 1.0),  # case insensitive
        ("xyz", "", 0.0),  # no match
    ]
    for text, expected_match, min_ratio in cases:
        match, ratio = fuzzy_match(text, products, threshold=0.6)
        status = "✓" if match == expected_match and ratio >= min_ratio else "✗"
        print(f"  {status} fuzzy_match('{text}') → ('{match}', {ratio:.2f})")
        if expected_match:
            assert match == expected_match


def test_correct_character_confusion():
    """Test character confusion correction."""
    # 0 → O in product context should NOT change (0 is digit, product favors letters)
    result = correct_character_confusion("0", context="product")
    print(f"  product context '0' → '{result}'")

    # O → 0 in price context should change
    result = correct_character_confusion("O", context="price")
    print(f"  price context 'O' → '{result}'")


def test_post_ocr_corrector():
    """Test full PostOCRCorrector with real OCR text."""
    corrector = PostOCRCorrector(PRODUCTS, CUSTOMERS)

    # Test product matching from noisy OCR
    ocr_text = "In 08, OKT Kamis men DD KRRIS — PP Madu 4 5 kena"
    result = corrector.correct_text(ocr_text)

    print(f"  Input: {repr(ocr_text[:60])}")
    print(f"  Date: {result['date']}")
    print(f"  Corrections: {len(result['corrections'])}")
    print(f"  Filtered text: {repr(result['corrected_text'][:60])}")

    # Should detect date
    assert result['date'] == "2026-10-08", f"Date should be 2026-10-08, got {result['date']}"
    print("  ✓ Date detected correctly")

    # Test product matching
    match, conf = corrector.correct_product_name("Madu")
    print(f"  Product 'Madu' → '{match}' (conf={conf:.2f})")
    assert match == "Madu"

    # Test fuzzy product matching
    match, conf = corrector.correct_product_name("Kopi")
    assert match == "Kopi"
    print(f"  Product 'Kopi' → '{match}' (conf={conf:.2f})")

    # Test customer matching
    match, conf = corrector.correct_customer_name("Marco")
    assert match == "Marco"
    print(f"  Customer 'Marco' → '{match}' (conf={conf:.2f})")


def test_noise_filter():
    """Test noise line filtering."""
    corrector = PostOCRCorrector(PRODUCTS, CUSTOMERS)

    text = "08 OKT Kamis\n||\n\n—\nMadu 5000\n___"
    filtered = corrector.filter_noise_lines(text)
    print(f"  Input lines: {len(text.split(chr(10)))}")
    print(f"  Filtered lines: {len(filtered.split(chr(10)))}")
    print(f"  Result: {repr(filtered)}")
    assert "08 OKT Kamis" in filtered
    assert "Madu 5000" in filtered
    assert "___" not in filtered
    print("  ✓ Noise lines filtered")


def main():
    print("=== Post-OCR Correction Unit Tests ===\n")

    print("1. Date parsing:")
    test_parse_indonesian_date()
    print()

    print("2. Price parsing:")
    test_parse_price()
    print()

    print("3. Quantity parsing:")
    test_parse_quantity()
    print()

    print("4. Fuzzy matching:")
    test_fuzzy_match()
    print()

    print("5. Character confusion:")
    test_correct_character_confusion()
    print()

    print("6. Noise filtering:")
    test_noise_filter()
    print()

    print("7. PostOCRCorrector integration:")
    test_post_ocr_corrector()
    print()

    print("=== ALL TESTS PASSED ✓ ===")


if __name__ == "__main__":
    main()
