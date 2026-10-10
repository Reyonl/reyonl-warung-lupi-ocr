"""
Evaluation harness for Warung Lupi OCR pipeline.

Usage:
    python eval/run_eval.py --engine easyocr --limit 5
    python eval/run_eval.py --engine tesseract --limit 40
    python eval/run_eval.py --engine easyocr --image IMG_20261008_115733.jpg

Outputs:
    - Terminal report (table)
    - eval/reports/<timestamp>.json
"""
import os
import sys
import json
import time
import argparse
import shutil
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.engine.engine_service import EngineService
from src.engines.tesseract_engine import TesseractEngine
from src.engines.easyocr_engine import EasyOCREngine

DATASET_DIR = "D:/dataset"
GROUND_TRUTH_DIR = project_root / "eval" / "ground_truth"
REPORTS_DIR = project_root / "eval" / "reports"
DECODE_DIR = project_root / "eval" / "decodes"


def load_ground_truths():
    """Load ground truth JSON files from eval/ground_truth/."""
    gts = {}
    if GROUND_TRUTH_DIR.exists():
        for f in GROUND_TRUTH_DIR.glob("*.json"):
            with open(f, encoding="utf-8") as fh:
                gt = json.load(fh)
                gts[gt["image"]] = gt
    return gts


def character_error_rate(predicted: str, ground_truth: str) -> float:
    """Compute CER using dynamic programming (edit distance / len(gt))."""
    if not ground_truth:
        return 1.0 if predicted else 0.0
    m, n = len(predicted), len(ground_truth)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif predicted[i - 1] == ground_truth[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    edits = dp[m][n]
    return edits / max(n, 1)


def word_error_rate(predicted: str, ground_truth: str) -> float:
    """Compute WER (word-level edit distance / len(words_gt))."""
    pred_words = predicted.lower().split()
    gt_words = ground_truth.lower().split()
    if not gt_words:
        return 1.0 if pred_words else 0.0
    return character_error_rate(" ".join(pred_words), " ".join(gt_words))


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, normalize spaces."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def match_product(ocr_text: str, product_catalog: list) -> str | None:
    """Fuzzy match OCR text to product in catalog."""
    from difflib import SequenceMatcher
    best_match = None
    best_ratio = 0.0
    text_norm = normalize_text(ocr_text)

    for prod in product_catalog:
        prod_name = normalize_text(prod["name"])
        ratio = SequenceMatcher(None, text_norm, prod_name).ratio()
        # Also check if product name is a substring of the OCR text (partial)
        if prod_name in text_norm and ratio > 0.3:
            ratio = max(ratio, 0.85)
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = prod["name"]

    if best_ratio > 0.6:
        return best_match
    return None


def evaluate_image(image_path: str, engine: str, ground_truth: dict | None,
                   product_catalog: list, customer_catalog: list,
                   engine_service: EngineService) -> dict:
    """Run OCR on a single image and compute metrics against ground truth."""
    basename = os.path.basename(image_path)
    t0 = time.time()
    scan = engine_service.process(image_path, engine=engine)
    eval_time = time.time() - t0

    # Extract text lines from regions
    detected_lines = []
    for region in scan.regions:
        text = region.text.strip()
        if text and region.confidence > 0.1:
            detected_lines.append({
                "text": text,
                "confidence": round(region.confidence, 4),
            })

    result = {
        "image": basename,
        "engine": engine,
        "ocr_confidence": round(scan.confidence, 4),
        "ocr_status": scan.status,
        "regions": len(scan.regions),
        "eval_time_seconds": round(eval_time, 3),
        "detected_lines": detected_lines,
    }

    if ground_truth is None:
        result["note"] = "No ground truth available for this image"
        return result

    # Normalize ground truth data structure (supports both sectioned and flat format)
    if "sections" in ground_truth:
        gt_lines = []
        gt_customers = []
        gt_items = []
        for sec in ground_truth.get("sections", []):
            cust = sec.get("customer")
            if cust and cust != "Unknown":
                gt_customers.append(cust)
            for it in sec.get("items", []):
                gt_items.append(it)
                raw = it.get("raw_line")
                if raw:
                    gt_lines.append(raw)
    else:
        gt_lines = ground_truth.get("lines", [])
        gt_customers = [ground_truth.get("customer")] if ground_truth.get("customer") else []
        gt_items = ground_truth.get("items", [])

    # CER/WER: compare full raw text
    gt_lines_joined = " ".join(gt_lines)
    cer = character_error_rate(scan.raw_text, gt_lines_joined)
    wer = word_error_rate(scan.raw_text, gt_lines_joined)
    result["metrics"] = {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
    }

    # Date accuracy
    gt_date = ground_truth.get("date")
    if gt_date:
        detected_dates = []
        for line in detected_lines:
            import re
            text = line["text"]
            if re.search(r'\d{1,2}[/.-]\d{1,2}([/.-]\d{2,4})?', text):
                detected_dates.append(text)
            if re.search(r'\d{1,2}\s+(jan|feb|mar|apr|mei|jun|jul|ags|sep|okt|nov|des|oktober)', text, re.IGNORECASE):
                detected_dates.append(text)

        result["metrics"]["date"] = {
            "ground_truth": gt_date,
            "detected": bool(detected_dates),
            "samples": detected_dates[:5],
        }

    # Customer accuracy
    if gt_customers:
        customer_detected = any(
            any(normalize_text(cust) in normalize_text(line["text"]) for cust in gt_customers)
            for line in detected_lines
        )
        result["metrics"]["customer"] = {
            "ground_truth": ", ".join(gt_customers),
            "detected": customer_detected,
        }

    # Item detection: how many lines contain product names
    detected_product_names = set()
    detected_qtys = set()
    detected_prices = set()

    for line in detected_lines:
        line_text = normalize_text(line["text"])
        for prod in product_catalog:
            prod_norm = normalize_text(prod["name"])
            if prod_norm in line_text or line_text in prod_norm:
                if len(line_text) > 2:
                    detected_product_names.add(prod["name"])
        import re
        prices_in_line = re.findall(r'(?:Rp\s*)?(\d{3,6})(?:\.\d{3})?', line["text"])
        for p in prices_in_line:
            try:
                val = int(p)
                if 500 <= val <= 500000:
                    detected_prices.add(val)
            except ValueError:
                pass
        qty_matches = re.findall(r'\bx(\d+)|\b(\d+)x\b|\b[\(](\d+)[\)]', line["text"])
        for parts in qty_matches:
            for part in parts:
                if part and int(part) < 100:
                    detected_qtys.add(int(part))

    gt_product_names = {item["product"] for item in gt_items if item.get("product")}

    result["metrics"]["items"] = {
        "products": {
            "gt_count": len(gt_product_names),
            "detected_count": len(detected_product_names),
            "precision": round(len(detected_product_names & gt_product_names) / max(len(detected_product_names), 1), 4),
            "recall": round(len(detected_product_names & gt_product_names) / max(len(gt_product_names), 1), 4),
        },
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="OCR Evaluation Harness")
    parser.add_argument("--engine", choices=["easyocr", "tesseract", "gemini", "claude", "auto"],
                        default="auto", help="Engine to evaluate")
    parser.add_argument("--limit", type=int, default=5,
                        help="Max images to evaluate")
    parser.add_argument("--image", type=str, default=None,
                        help="Single image to evaluate")
    parser.add_argument("--dataset", type=str, default=DATASET_DIR,
                        help="Dataset directory")
    args = parser.parse_args()

    # Load ground truths
    gts = load_ground_truths()
    print(f"Loaded {len(gts)} ground truth files")

    # Product catalog (from Laravel or hardcoded for eval)
    product_catalog = [
        {"id": 1, "name": "Kopi"}, {"id": 2, "name": "Teh"},
        {"id": 3, "name": "Aqua"}, {"id": 4, "name": "Indomie"},
        {"id": 5, "name": "Rokok Surya"}, {"id": 6, "name": "Beras"},
        {"id": 7, "name": "Minyak Bimoli"}, {"id": 8, "name": "Gula Pasir"},
        {"id": 9, "name": "Kecap ABC"}, {"id": 10, "name": "Saus Tomat"},
        {"id": 11, "name": "Mie Instan"}, {"id": 12, "name": "Telur"},
        {"id": 13, "name": "Tahu"}, {"id": 14, "name": "Tempe"},
        {"id": 15, "name": "Sayur"}, {"id": 16, "name": "Buah"},
        {"id": 17, "name": "Bumbu"}, {"id": 18, "name": "Garam"},
        {"id": 19, "name": "Penyedap"}, {"id": 20, "name": "Kecap Manis"},
    ]
    customer_catalog = [
        {"id": 14, "name": "Rama"}, {"id": 27, "name": "Marco"},
        {"id": 99, "name": "Bu Sri"}, {"id": 12, "name": "Mama Sasa"},
    ]

    # Discover images
    dataset = args.dataset
    image_files = sorted(
        glob.glob(os.path.join(dataset, "IMG_*.jpg")) +
        glob.glob(os.path.join(dataset, "IMG_*.jpeg"))
    )
    print(f"Found {len(image_files)} images in dataset")

    if args.image:
        image_files = [os.path.join(dataset, args.image)]
    else:
        image_files = image_files[:args.limit]

    # Setup engine service
    service = EngineService()

    # Load config and auto-register engines
    try:
        from src.config import OCRConfig
        config = OCRConfig.load()
        service.auto_register_default_engines(config)
    except Exception as e:
        print(f"Config/Engine registration note: {e}")
        service.auto_register_default_engines(None)

    # Choose engine
    engine = args.engine
    if engine == "auto":
        available = service.get_available_engines()
        if any(name.lower().startswith("gemini") for name in available):
            engine = "gemini"
        elif any(name.lower().startswith("claude") for name in available):
            engine = "claude"
        elif any(name.lower().startswith("easyocr") for name in available):
            engine = "easyocr"
        elif any(name.lower().startswith("tesseract") for name in available):
            engine = "tesseract"
        else:
            engine = "tesseract"
        print(f"Auto-selected engine: {engine} (available: {available})")

    # Run evaluation
    all_results = []
    for img_path in image_files:
        basename = os.path.basename(img_path)
        gt = gts.get(basename)
        result = evaluate_image(img_path, engine, gt, product_catalog,
                                customer_catalog, service)
        all_results.append(result)

        # Print inline summary
        metrics = result.get("metrics", {})
        metric_str = ""
        if "cer" in metrics:
            metric_str = f" CER={metrics['cer']:.4f} WER={metrics['wer']:.4f}"
        if "date" in metrics.get("date", {}):
            d = metrics.get("date", {})
            metric_str += f" date={d.get('ground_truth')}->{'✓' if d.get('detected') else '✗'}"

        if gt:
            item_count = sum(len(s.get("items", [])) for s in gt.get("sections", [])) if "sections" in gt else len(gt.get("items", []))
            gt_line = f" [GT: date={gt.get('date','?')}, items={item_count}]"
        else:
            gt_line = " [No GT]"
        print(f"  {basename}: {engine} conf={result['ocr_confidence']:.3f} regions={result['regions']} t={result['eval_time_seconds']:.1f}s{metric_str}{gt_line}", flush=True)

    # Aggregate metrics
    report = {
        "timestamp": datetime.now().isoformat(),
        "engine": engine,
        "image_count": len(all_results),
        "images_evaluated": [r["image"] for r in all_results],
        "results": all_results,
    }

    # Compute aggregates if we have ground truth
    cer_values = [r["metrics"]["cer"] for r in all_results if "cer" in r.get("metrics", {})]
    wer_values = [r["metrics"]["wer"] for r in all_results if "wer" in r.get("metrics", {})]
    date_values = [r["metrics"]["date"] for r in all_results if "date" in r.get("metrics", {})]

    aggregates = {}
    if cer_values:
        aggregates["avg_cer"] = round(sum(cer_values) / len(cer_values), 4)
        report["aggregates"] = aggregates
    if wer_values:
        aggregates["avg_wer"] = round(sum(wer_values) / len(wer_values), 4)
    if date_values:
        date_hits = sum(1 for d in date_values if d.get("detected"))
        aggregates["date_accuracy"] = round(date_hits / len(date_values), 4)
        report["aggregates"] = aggregates

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"{timestamp_str}_{engine}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Also save as "latest"
    latest_path = REPORTS_DIR / f"latest_{engine}.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"Evaluation Report: {len(all_results)} images, engine={engine}")
    print(f"{'='*70}")

    headers = ["Image", "Conf", "Regions", "Time(s)"]
    if cer_values:
        headers.append("Avg CER")
    if date_values:
        headers.append("Date%")
    print(" | ".join(f"{h:>12}" for h in headers))
    print("-" * 70)

    for r in all_results:
        row = [
            r["image"][:15],
            f"{r['ocr_confidence']:.2f}",
            str(r["regions"]),
            f"{r['eval_time_seconds']:.1f}",
        ]
        if cer_values:
            row.append(f"{r.get('metrics', {}).get('cer', 'N/A'):.4f}" if r.get('metrics', {}).get('cer') else "N/A")
        if date_values and "date" in r.get("metrics", {}):
            d = r["metrics"]["date"]
            row.append(f"{'✓' if d.get('detected') else '✗'}")
        print(" | ".join(f"{c:>12}" for c in row))

    print("-" * 70)
    if aggregates:
        for k, v in aggregates.items():
            print(f"  {k}: {v}")
    print(f"\nReport saved: {report_path}")


import glob  # noqa: E402
import re    # noqa: E402

if __name__ == "__main__":
    main()
