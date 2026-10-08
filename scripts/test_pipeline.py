"""
Integration test: Full OCR → Draft pipeline.

Tests the complete flow:
1. Phase 1: OCR (EasyOCR) on sample notebook images
2. Phase 2: Interpretation (date, customer, items, qty, price)
3. Phase 3: Product matching (fuzzy match to product DB)
4. Phase 4: Customer matching (fuzzy match to customer DB)
5. Phase 5: Draft transaction creation
6. Phase 9: API payload generation (for POST /api/transactions)

Usage:
    python scripts/test_pipeline.py [--image PATH] [--engine easyocr|tesseract]

Output:
    Prints results to console + saves JSON to scripts/_test_result.json
"""

import os
import sys
import json
import argparse
import shutil

# Ensure project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

# Clear any cached modules
for k in list(sys.modules.keys()):
    if k.startswith('src'):
        del sys.modules[k]

from src.engine.engine_service import EngineService
from src.engines.tesseract_engine import TesseractEngine
from src.engines.easyocr_engine import EasyOCREngine
from src.interpreter.service import InterpretationService
from src.draft.service import DraftTransactionService
from src.preprocess.processor import ImageProcessor


# Simulated Warung Lupi product DB (from Laravel backend)
PRODUCTS = [
    {"id": 1, "name": "Kopi", "default_price": 5000, "unit": "gelas"},
    {"id": 2, "name": "Es Kopi", "default_price": 7000, "unit": "gelas"},
    {"id": 3, "name": "Teh", "default_price": 4000, "unit": "gelas"},
    {"id": 4, "name": "Es Teh", "default_price": 5000, "unit": "gelas"},
    {"id": 6, "name": "Aqua K", "default_price": 5000, "unit": "botol"},
    {"id": 7, "name": "Aqua Galon", "default_price": 20000, "unit": "galon"},
    {"id": 8, "name": "Teh Poci", "default_price": 6000, "unit": "gelas"},
    {"id": 9, "name": "Sirup", "default_price": 3000, "unit": "botol"},
    {"id": 10, "name": "Kopi Sachet", "default_price": 2000, "unit": "pcs"},
    {"id": 11, "name": "Makan", "default_price": 10000, "unit": "porsi"},
    {"id": 12, "name": "Nasi", "default_price": 5000, "unit": "porsi"},
    {"id": 13, "name": "Mie Goreng", "default_price": 9000, "unit": "porsi"},
    {"id": 14, "name": "Mie", "default_price": 8000, "unit": "porsi"},
    {"id": 15, "name": "Mie Rebus", "default_price": 10000, "unit": "porsi"},
    {"id": 16, "name": "Bakmi", "default_price": 8000, "unit": "porsi"},
    {"id": 17, "name": "Donat", "default_price": 2000, "unit": "pcs"},
    {"id": 18, "name": "Kue", "default_price": 3000, "unit": "pcs"},
    {"id": 19, "name": "Roti", "default_price": 2500, "unit": "pcs"},
    {"id": 20, "name": "Telur", "default_price": 3000, "unit": "butir"},
    {"id": 21, "name": "Ayam", "default_price": 15000, "unit": "potong"},
    {"id": 22, "name": "Rokok", "default_price": 25000, "unit": "bungkus"},
    {"id": 23, "name": "Sigarette", "default_price": 25000, "unit": "bungkus"},
    {"id": 24, "name": "Kretek", "default_price": 22000, "unit": "bungkus"},
    {"id": 25, "name": "gudang garam filter", "default_price": 29000, "unit": "bungkus"},
    {"id": 26, "name": "esse double", "default_price": 47000, "unit": "bungkus"},
]

# Simulated customer DB
CUSTOMERS = [
    {"id": 1, "name": "Umum"},
    {"id": 2, "name": "Pak Udin"},
    {"id": 10, "name": "Bang Lee"},
    {"id": 12, "name": "Uni Chanta"},
    {"id": 14, "name": "Rama"},
    {"id": 15, "name": "Sandi"},
    {"id": 16, "name": "Untung"},
    {"id": 27, "name": "Marco"},
    {"id": 10, "name": "Pak Ade"},
    {"id": 32, "name": "Mama Sasa"},
]


def run_pipeline(image_path: str, engine_name: str = "auto") -> dict:
    """Run the full pipeline on an image."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(image_path)}")
    print(f"Engine: {engine_name}")
    print(f"{'='*60}")

    # Phase 1: OCR
    service = EngineService()
    
    if engine_name == "auto" or engine_name == "both":
        # Try both, pick best
        tess = TesseractEngine()
        easy = EasyOCREngine()
        if tess.is_available():
            service.register_engine(tess)
        if easy.is_available():
            service.register_engine(easy)
        
        tess_result = service.process(image_path, engine="tesseract") if tess.is_available() else None
        easy_result = service.process(image_path, engine="easyocr") if easy.is_available() else None
        
        if tess_result and easy_result:
            # Pick the one with higher confidence
            if easy_result.confidence > tess_result.confidence:
                scan = easy_result
                print(f"  OCR: EasyOCR (conf={scan.confidence:.3f}, regions={len(scan.regions)})")
            else:
                scan = tess_result
                print(f"  OCR: Tesseract (conf={scan.confidence:.3f}, regions={len(scan.regions)})")
        elif easy_result:
            scan = easy_result
            print(f"  OCR: EasyOCR (conf={scan.confidence:.3f}, regions={len(scan.regions)})")
        elif tess_result:
            scan = tess_result
            print(f"  OCR: Tesseract (conf={scan.confidence:.3f}, regions={len(scan.regions)})")
        else:
            print("  ERROR: No OCR engine available")
            return {"status": "error", "error": "No OCR engine available"}
    else:
        if engine_name == "tesseract":
            service.register_engine(TesseractEngine())
        elif engine_name == "easyocr":
            service.register_engine(EasyOCREngine())
        scan = service.process(image_path, engine=engine_name)
        print(f"  OCR: {engine_name} (conf={scan.confidence:.3f}, regions={len(scan.regions)})")

    if scan.status == "error":
        print(f"  ERROR: {scan.warnings}")
        return {"status": "error", "error": scan.warnings}

    # Phase 1.5: Structural symbol detection
    processor = ImageProcessor()
    circled = processor.detect_circled_numbers(image_path)
    symbols = processor.detect_structural_symbols(image_path)
    print(f"  Symbols: {len(circled)} circled numbers, {len(symbols)} structural symbols detected")

    # Phase 2: Interpretation
    interpreter = InterpretationService()
    interpreted = interpreter.interpret(scan)
    print(f"\n  Interpretation:")
    print(f"    Date: {interpreted.date} (conf={interpreted.date_confidence:.3f})")
    if interpreted.customer:
        print(f"    Customer: {interpreted.customer.detected_name} (conf={interpreted.customer.confidence:.3f})")
    print(f"    Items: {len(interpreted.items)}")
    print(f"    Uncertain: {len(interpreted.uncertain)}")

    # Phase 3+4: Matching
    interpreted = interpreter.enhance_with_products(interpreted, PRODUCTS)
    interpreted = interpreter.enhance_with_customers(interpreted, CUSTOMERS)

    # Phase 5: Draft
    draft_service = DraftTransactionService()
    draft = draft_service.create_draft_from_interpretation(interpreted)
    
    # Apply default prices from product DB
    draft = draft_service.apply_product_defaults(draft, PRODUCTS)

    print(f"\n  Draft Transaction:")
    print(f"    ID: {draft.id}")
    print(f"    Date: {draft.date}")
    print(f"    Customer: {draft.customer_name} (id={draft.customer_id})")
    print(f"    Total items: {len(draft.items)}")
    print(f"    Total amount: {draft.total_amount}")
    print(f"    Unresolved warnings: {len(draft.unresolved_items)}")

    # Show items
    print(f"\n  Items Detail:")
    for i, item in enumerate(draft.items):
        status = "✓" if item.product_id else "?"
        qty = item.quantity if item.quantity else "?"
        price = f"Rp{item.unit_price:,.0f}" if item.unit_price else "Rp?"
        print(f"    [{i}] {status} {item.description[:30]:30s} x{qty} @ {price}")
        for w in item.warnings:
            print(f"        ⚠ {w}")

    # Phase 9: API payload
    payload = draft_service.confirm_draft(draft)
    print(f"\n  API Payload (for POST /api/transactions):")
    print(f"    date: {payload['date']}")
    print(f"    customer_id: {payload['customer_id']}")
    print(f"    items count: {len(payload['items'])}")
    print(f"    total_amount: {payload['total_amount']}")

    # Save result
    result = {
        "image": os.path.basename(image_path),
        "engine": scan.model,
        "ocr_confidence": scan.confidence,
        "ocr_regions": len(scan.regions),
        "symbols": {"circled_numbers": len(circled), "structural": len(symbols)},
        "interpretation": {
            "date": interpreted.date,
            "date_confidence": round(interpreted.date_confidence, 4),
            "customer": {
                "detected_name": interpreted.customer.detected_name if interpreted.customer else None,
                "matched_id": interpreted.customer.matched_customer_id if interpreted.customer else None,
                "confidence": round(interpreted.customer.confidence, 4) if interpreted.customer else 0,
            } if interpreted.customer else None,
            "items_count": len(interpreted.items),
            "uncertain_count": len(interpreted.uncertain),
        },
        "draft": draft.to_dict(),
        "api_payload": payload,
        "status": "ok"
    }

    output_path = os.path.join(project_root, "scripts", "_test_result.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  Results saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test Warung Lupi OCR pipeline")
    parser.add_argument(
        "--image", 
        default="D:/dataset/IMG_20261008_115733.jpg",
        help="Path to test image"
    )
    parser.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "tesseract", "easyocr", "both"],
        help="OCR engine to use"
    )
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Run on all images in dataset dir"
    )
    args = parser.parse_args()

    if args.all_images:
        images_dir = "D:/dataset"
        image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and not f.startswith("_")
        ])[:3]

        print(f"Running pipeline on {len(image_files)} images...")
        results = []
        for img in image_files:
            img_path = os.path.join(images_dir, img)
            result = run_pipeline(img_path, args.engine)
            results.append(result)

        print(f"\n{'='*60}")
        print(f"Summary: {len(results)} images processed")
        success = sum(1 for r in results if r.get("status") == "ok")
        print(f"  Success: {success}/{len(results)}")
        failed = [r for r in results if r.get("status") != "ok"]
        if failed:
            for r in failed:
                print(f"  Failed: {r.get('error', 'unknown')}")
        else:
            # Show summary stats
            total_items = sum(len(r.get("draft", {}).get("items", [])) for r in results if r.get("status") == "ok")
            print(f"  Total items parsed: {total_items}")
            print(f"  Avg OCR confidence: {sum(r.get('ocr_confidence', 0) for r in results) / len(results):.3f}")
    else:
        run_pipeline(args.image, args.engine)


if __name__ == "__main__":
    main()
