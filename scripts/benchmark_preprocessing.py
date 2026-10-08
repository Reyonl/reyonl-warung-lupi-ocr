"""Benchmark script: Compare OCR preprocessing methods."""

import os
import sys
import json
import time
import shutil
import argparse
import statistics
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Clear Python caches
for root, dirs, files in os.walk(PROJECT_ROOT):
    for d in list(dirs):
        if d == "__pycache__":
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
    for f in files:
        if f.endswith(".pyc"):
            os.remove(os.path.join(root, f))

for k in list(sys.modules.keys()):
    if k.startswith("src"):
        del sys.modules[k]

from src.engine.engine_service import EngineService
from src.engines.easyocr_engine import EasyOCREngine
from src.engines.tesseract_engine import TesseractEngine
from src.preprocess.processor import ImageProcessor


DATASET_DIR = "D:/dataset"
PREPROCESSING_METHODS = [
    "none",
    "gray",
    "gray_norm",
    "gray_sharpen",
    "gray_norm_sharpen",
    "gray_blur_thresh",
    "gray_thresh",
    "gray_clahe",
]


def load_images(max_images=None):
    """Load image paths from dataset directory."""
    images = []
    if os.path.exists(DATASET_DIR):
        for f in sorted(os.listdir(DATASET_DIR)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(DATASET_DIR, f))
    return images[:max_images] if max_images else images


def benchmark_image(image_path, engine_name, preprocessing_methods):
    """Benchmark all preprocessing methods on a single image."""
    results = {"image": os.path.basename(image_path), "methods": {}}

    processor = ImageProcessor()

    for method in preprocessing_methods:
        try:
            start = time.time()
            if method == "none":
                # Original image
                scan = None
                # Process original with engine
                service = EngineService()
                if engine_name == "easyocr":
                    service.register_engine(EasyOCREngine())
                else:
                    service.register_engine(TesseractEngine())

                scan = service.process(image_path, engine=engine_name)
            else:
                # Preprocess then OCR — save to temp
                temp_path = os.path.join(
                    os.path.dirname(image_path),
                    f"_tmp_{method}_{os.path.basename(image_path)}",
                )
                processed = processor.apply(image_path, [method])
                if processed is not None:
                    cv2.imwrite(temp_path, processed)

                    service = EngineService()
                    if engine_name == "easyocr":
                        service.register_engine(EasyOCREngine())
                    else:
                        service.register_engine(TesseractEngine())

                    scan = service.process(temp_path, engine=engine_name)
                    os.remove(temp_path)
                else:
                    scan = None

            elapsed = time.time() - start

            if scan:
                # Count readable words (confidence > 0.35)
                high_conf_words = [r for r in scan.regions if r.confidence > 0.35]
                avg_conf = statistics.mean(
                    [r.confidence for r in scan.regions]
                ) if scan.regions else 0
                total_conf = sum(r.confidence for r in scan.regions)
                total_words = len([r for r in scan.regions if r.text and r.text.strip()])

                results["methods"][method] = {
                    "confidence": round(avg_conf, 4),
                    "regions": len(scan.regions),
                    "high_conf_words": len(high_conf_words),
                    "total_words": total_words,
                    "time_seconds": round(elapsed, 2),
                    "status": scan.status,
                }
            else:
                results["methods"][method] = {
                    "confidence": 0,
                    "regions": 0,
                    "error": "OCR failed",
                    "time_seconds": round(elapsed, 2),
                }

        except Exception as e:
            results["methods"][method] = {"error": str(e)}

    return results


def analyze_results(all_results):
    """Analyze and rank preprocessing methods."""
    print("\n" + "=" * 70)
    print("PREPROCESSING BENCHMARK RESULTS")
    print("=" * 70)

    # Aggregate per method
    method_scores = {m: {"confidence": [], "regions": [], "time": [], "images": 0}
                     for m in PREPROCESSING_METHODS}

    for result in all_results:
        for method, data in result["methods"].items():
            if method in method_scores and "confidence" in data:
                method_scores[method]["confidence"].append(data["confidence"])
                method_scores[method]["regions"].append(data["regions"])
                method_scores[method]["time"].append(data["time_seconds"])
                method_scores[method]["images"] += 1

    print(f"\nTested on {len(all_results)} images\n")
    print(f"{'Method':<25} {'Avg Conf':>10} {'Avg Regions':>12} {'Avg Time':>10} {'Images':>7}")
    print("-" * 70)

    ranking = []
    for method in PREPROCESSING_METHODS:
        scores = method_scores[method]
        if scores["images"] > 0:
            avg_conf = statistics.mean(scores["confidence"])
            avg_reg = statistics.mean(scores["regions"])
            avg_time = statistics.mean(scores["time"])
            ranking.append((method, avg_conf, avg_reg, avg_time, scores["images"]))
            print(f"{method:<25} {avg_conf:>10.4f} {avg_reg:>12.1f} {avg_time:>9.2f}s {scores['images']:>7}")
            status = "✅ BETTER" if avg_conf > 0.35 else "❌ WORSE"
        else:
            print(f"{method:<25} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'0':>7}")

    # Sort ranking by confidence
    ranking.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print("RANKING (best to worst by avg confidence)")
    print("=" * 70)
    for i, (method, conf, reg, t, n) in enumerate(ranking):
        marker = " ← BEST" if i == 0 else ""
        print(f"  {i+1}. {method:<25} conf={conf:.4f} regions={reg:.1f} time={t:.2f}s{marker}")

    print(f"\n⚠️  Key insight: 'none' (no preprocessing) is the baseline.")
    print(f"    Any method WORSE than 'none' should NOT be used.")
    print(f"    Noise amplification from aggressive preprocessing is common.")

    return ranking


def main():
    parser = argparse.ArgumentParser(description="Benchmark OCR preprocessing methods")
    parser.add_argument("--images", type=int, default=3, help="Max images to test")
    parser.add_argument("--engine", choices=["easyocr", "tesseract"], default="easyocr")
    parser.add_argument("--output", default="scripts/_benchmark_result.json")
    args = parser.parse_args()

    print(f"Warung Lupi OCR — Preprocessing Benchmark")
    print(f"Engine: {args.engine}")
    print(f"Dataset: {DATASET_DIR}")

    images = load_images(args.images)
    if not images:
        print(f"ERROR: No images found in {DATASET_DIR}")
        sys.exit(1)

    print(f"\nFound {len(images)} images, benchmarking {len(PREPROCESSING_METHODS)} methods...\n")

    all_results = []
    for img_path in images:
        print(f"Testing {os.path.basename(img_path)}...")
        result = benchmark_image(img_path, args.engine, PREPROCESSING_METHODS)
        all_results.append(result)

        # Show per-image ranking
        methods_with_conf = [
            (m, d["confidence"]) for m, d in result["methods"].items()
            if "confidence" in d
        ]
        methods_with_conf.sort(key=lambda x: x[1], reverse=True)
        print(f"  → Best: {methods_with_conf[0][0]} (conf={methods_with_conf[0][1]:.4f})")
        print(f"  → 'none':  {[m for m,c in methods_with_conf if m=='none'][0] if methods_with_conf else 'N/A'} (conf={[c for m,c in methods_with_conf if m=='none']}")

    # Analyze
    ranking = analyze_results(all_results)

    # Save results
    output = {
        "engine": args.engine,
        "images_tested": len(images),
        "image_files": [os.path.basename(r["image"]) for r in all_results],
        "all_results": all_results,
        "ranking": [{"method": r[0], "confidence": r[1], "regions": r[2], "time": r[3]} for r in ranking],
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output}")

    # Verdict
    none_conf = next((r[1] for r in ranking if r[0] == "none"), 0)
    best_conf = ranking[0][1] if ranking else 0
    best_method = ranking[0][0] if ranking else "none"

    if none_conf >= best_conf:
        print(f"\n✅ VERDICT: 'none' (no preprocessing) is best ({none_conf:.4f})")
        print("   Recommendation: skip preprocessing for this dataset")
    else:
        print(f"\n✅ VERDICT: '{best_method}' is best ({best_conf:.4f})")
        print(f"   Improvement over 'none': +{((best_conf/none_conf - 1)*100):.1f}%" if none_conf > 0 else "")
        print(f"   Recommendation: use '{best_method}' preprocessing")


if __name__ == "__main__":
    main()