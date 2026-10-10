"""
Unit and integration tests for Warung Lupi OCR:
- Product and customer matching (ImprovedMatcher)
- Interpretation and enhancement with catalogs
- Draft transaction creation and default pricing
- Laravel API payload generation
- Engine availability
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.matching.improved_matching import (
    ImprovedMatcher,
    build_matcher,
    normalize_indonesian,
    trigram_similarity,
    phonetic_match,
    enhanced_fuzzy_match,
)
from src.interpreter.service import (
    InterpretationService,
    InterpretedResult,
    InterpretedItem,
    InterpretedCustomer,
)
from src.draft.service import DraftTransactionService
from src.engine.engine_service import EngineService
from src.engines.tesseract_engine import TesseractEngine
from src.engines.gemini_vision_engine import GeminiVisionEngine


SAMPLE_PRODUCTS = [
    {"id": 1, "name": "Kopi", "default_price": 5000, "unit": "gelas"},
    {"id": 2, "name": "Es Kopi", "default_price": 7000, "unit": "gelas"},
    {"id": 3, "name": "Teh", "default_price": 4000, "unit": "gelas"},
    {"id": 4, "name": "Aqua Gelas", "default_price": 3000, "unit": "gelas"},
    {"id": 5, "name": "Indomie", "default_price": 8000, "unit": "porsi"},
    {"id": 6, "name": "Madu", "default_price": 15000, "unit": "botol"},
    {"id": 7, "name": "Gorengan", "default_price": 2500, "unit": "pcs"},
]

SAMPLE_CUSTOMERS = [
    {"id": 1, "name": "Marco"},
    {"id": 2, "name": "Rama"},
    {"id": 3, "name": "Mama Sasa"},
    {"id": 4, "name": "Mbak Tami"},
]


def test_normalize_indonesian():
    assert normalize_indonesian("Indomi, Goreng!") == "indomie goreng"
    assert normalize_indonesian("Gls Aqua") == "aqua gelas aqua"
    assert normalize_indonesian("KOPI SUSU") == "kopi susu"


def test_trigram_similarity():
    assert trigram_similarity("kopi", "kopi") == 1.0
    assert trigram_similarity("kopi", "kop") > 0.4
    assert trigram_similarity("abc", "xyz") == 0.0


def test_phonetic_match():
    assert phonetic_match("kopi", "kopi") == 1.0
    # kupi has same consonant skeleton 'kp'
    assert phonetic_match("kupi", "kopi") >= 0.8
    # completely different words
    assert phonetic_match("kopi", "beras") < 0.3


def test_improved_matcher_products():
    matcher = ImprovedMatcher(SAMPLE_PRODUCTS, SAMPLE_CUSTOMERS)

    # 1. Exact match
    name, conf, method = matcher.match_product("Kopi")
    assert name == "Kopi"
    assert conf == 1.0
    assert method == "exact"

    # 2. Alias match (indomi -> indomie)
    name, conf, method = matcher.match_product("indomi")
    assert name == "Indomie"
    assert conf >= 0.85

    # 3. Substring match
    name, conf, method = matcher.match_product("es kopi susu")
    assert "Kopi" in name
    assert conf >= 0.80

    # 4. Price boost
    name, conf, method = matcher.match_product("kopi", price=5000)
    assert name == "Kopi"
    assert conf >= 0.95


def test_improved_matcher_customers():
    matcher = ImprovedMatcher(SAMPLE_PRODUCTS, SAMPLE_CUSTOMERS)

    # Substring / title match ("Tami" -> "Mbak Tami")
    name, conf, method = matcher.match_customer("tami")
    assert name == "Mbak Tami"
    assert conf >= 0.80

    # Exact match
    name, conf, method = matcher.match_customer("Marco")
    assert name == "Marco"
    assert conf == 1.0


def test_rank_products_and_customers():
    matcher = ImprovedMatcher(SAMPLE_PRODUCTS, SAMPLE_CUSTOMERS)

    ranked_prods = matcher.rank_products("kopi", top_k=3)
    assert len(ranked_prods) >= 1
    assert ranked_prods[0]["name"] in ["Kopi", "Es Kopi"]
    assert ranked_prods[0]["product_id"] in [1, 2]

    ranked_custs = matcher.rank_customers("mama sasa", top_k=2)
    assert len(ranked_custs) >= 1
    assert ranked_custs[0]["name"] == "Mama Sasa"


def test_interpretation_service_enhancement():
    service = InterpretationService()
    result = InterpretedResult(
        date="2026-10-08",
        date_confidence=0.9,
        customer=InterpretedCustomer(detected_name="Tami", confidence=0.8),
        items=[
            InterpretedItem(detected_text="indomi", price=8000.0, quantity=1),
            InterpretedItem(detected_text="kopi", price=5000.0, quantity=2),
        ],
    )

    enhanced = service.enhance_with_products(result, SAMPLE_PRODUCTS)
    assert enhanced.items[0].matched_product_name == "Indomie"
    assert enhanced.items[0].matched_product_id == 5
    assert enhanced.items[1].matched_product_name == "Kopi"
    assert enhanced.items[1].matched_product_id == 1

    enhanced = service.enhance_with_customers(enhanced, SAMPLE_CUSTOMERS)
    assert enhanced.customer.matched_customer_id == 4  # Mbak Tami


def test_draft_transaction_and_payload():
    result = InterpretedResult(
        date="2026-10-08",
        date_confidence=0.9,
        customer=InterpretedCustomer(
            detected_name="Marco", confidence=0.95, matched_customer_id=1
        ),
        items=[
            InterpretedItem(
                detected_text="kopi",
                quantity=2,
                price=5000.0,
                matched_product_id=1,
                matched_product_name="Kopi",
            ),
            InterpretedItem(
                detected_text="gorengan",
                quantity=4,
                price=None,  # missing price
                matched_product_id=7,
                matched_product_name="Gorengan",
            ),
        ],
    )

    draft_service = DraftTransactionService()
    draft = draft_service.create_draft_from_interpretation(result)
    assert draft.customer_id == 1
    assert len(draft.items) == 2

    # Apply product defaults for missing price
    draft = draft_service.apply_product_defaults(draft, SAMPLE_PRODUCTS)
    assert draft.items[1].unit_price == 2500

    # Confirm and generate Laravel payload
    payload = draft_service.confirm_draft(draft)
    assert payload["transaction"]["transaction_date"] == "2026-10-08"
    assert payload["transaction"]["customer_id"] == 1
    assert len(payload["items"]) == 2
    # 2*5000 + 4*2500 = 20000
    assert payload["total_amount"] == 20000


def test_engine_service_orchestration():
    service = EngineService()
    service.auto_register_default_engines(None)

    available = service.get_available_engines()
    assert isinstance(available, list)

    tess = TesseractEngine()
    assert tess.name().startswith("tesseract")
    assert tess.is_available() is True

    gemini = GeminiVisionEngine(api_key=None)
    # Available should be false when api_key is None
    assert gemini.is_available() is False


if __name__ == "__main__":
    print("Running tests directly...")
    test_normalize_indonesian()
    print("✓ test_normalize_indonesian")
    test_trigram_similarity()
    print("✓ test_trigram_similarity")
    test_phonetic_match()
    print("✓ test_phonetic_match")
    test_improved_matcher_products()
    print("✓ test_improved_matcher_products")
    test_improved_matcher_customers()
    print("✓ test_improved_matcher_customers")
    test_rank_products_and_customers()
    print("✓ test_rank_products_and_customers")
    test_interpretation_service_enhancement()
    print("✓ test_interpretation_service_enhancement")
    test_draft_transaction_and_payload()
    print("✓ test_draft_transaction_and_payload")
    test_engine_service_orchestration()
    print("✓ test_engine_service_orchestration")
    print("\nALL 9 TESTS PASSED SUCCESSFULLY! ✓")
