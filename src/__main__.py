"""
Warung Lupi OCR Engine — Book Note Scanner
Main entry point for OCR processing.

Usage:
    python -m src --image <path_to_image> --output <json_output_path>
    python -m src --benchmark --images-dir <dir> --engines tesseract easyocr
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Ensure local tessdata is found
this_dir = os.path.dirname(os.path.abspath(__file__))
local_tessdata = os.path.join(os.path.dirname(this_dir), "tessdata")
if os.path.isdir(local_tessdata):
    os.environ["TESSDATA_PREFIX"] = local_tessdata

sys.path.insert(0, os.path.dirname(this_dir))

from src.engine.engine_service import EngineService, EngineConfig
from src.engines.tesseract_engine import TesseractEngine
from src.engines.easyocr_engine import EasyOCREngine
from src.models.scan_result import ScanResult

logger = logging.getLogger("ocr")


def process_image(image_path: str, engine_name: str = "auto") -> dict:
    """Process a single image through OCR pipeline."""
    service = EngineService()

    available = []
    tess = TesseractEngine()
    if tess.is_available():
        available.append(("tesseract", tess))

    easy = EasyOCREngine()
    if easy.is_available():
        available.append(("easyocr", easy))

    if not available:
        return {"status": "error", "error": "No OCR engine available"}

    if engine_name == "auto":
        for name, engine in available:
            if name == "easyocr":
                service.register_engine(engine)
        if not service.engines:
            for name, engine in available:
                service.register_engine(engine)
        result = service.process(image_path)
    else:
        found = False
        for name, engine in available:
            if name == engine_name:
                service.register_engine(engine)
                result = service.process(image_path, engine=engine_name)
                found = True
                break
        if not found:
            return {"status": "error", "error": f"Engine {engine_name} not available"}

    data = result.to_dict()

    if result.regions:
        data["regions_sample"] = [
            {
                "text": r.text,
                "confidence": round(r.confidence, 4),
                "bbox": r.bbox,
                "engine": r.engine,
            }
            for r in result.regions
        ]

    return data


def run_benchmark(images_dir: str, engines: List[str]) -> dict:
    """Benchmark multiple OCR engines on a directory of images."""
    results = {}
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('_')
    ])[:5]

    logger.info(f"Benchmarking on {len(image_files)} images from {images_dir}")

    service = EngineService()
    tess = TesseractEngine()
    easy = EasyOCREngine()

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue

        logger.info(f"Processing: {img_name}")
        img_results = {"image": img_name}

        if "tesseract" in engines and tess.is_available():
            start = time.time()
            result = service.process(img_path, engine="tesseract")
            elapsed = time.time() - start
            img_results["tesseract"] = {
                "confidence": result.confidence,
                "regions": len(result.regions),
                "time": round(elapsed, 2),
                "status": result.status,
                "sample_high_conf": [
                    {"text": r.text, "conf": round(r.confidence, 4)}
                    for r in result.regions if r.confidence > 0.5
                ][:15],
            }
            logger.info(f"  Tesseract: conf={result.confidence:.3f}, "
                       f"regions={len(result.regions)}, time={elapsed:.1f}s")

        if "easyocr" in engines and easy.is_available():
            start = time.time()
            result = service.process(img_path, engine="easyocr")
            elapsed = time.time() - start
            img_results["easyocr"] = {
                "confidence": result.confidence,
                "regions": len(result.regions),
                "time": round(elapsed, 2),
                "status": result.status,
                "sample_high_conf": [
                    {"text": r.text, "conf": round(r.confidence, 4)}
                    for r in result.regions if r.confidence > 0.5
                ][:15],
            }
            logger.info(f"  EasyOCR: conf={result.confidence:.3f}, "
                       f"regions={len(result.regions)}, time={elapsed:.1f}s")

        results[img_name] = img_results

    return {"benchmark_results": results, "images_dir": images_dir}


def main():
    parser = argparse.ArgumentParser(description="Warung Lupi OCR Engine")
    parser.add_argument("--image", help="Path to single image to process")
    parser.add_argument("--engine", default="auto",
                       choices=["tesseract", "easyocr", "auto"],
                       help="OCR engine to use (default: auto)")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark on images directory")
    parser.add_argument("--images-dir", default="D:/dataset",
                       help="Directory of images for benchmark")
    parser.add_argument("--engines", nargs="+", default=["tesseract", "easyocr"],
                       help="Engines to benchmark")
    parser.add_argument("--output", help="Output JSON file path")

    args = parser.parse_args()

    if args.benchmark:
        result = run_benchmark(args.images_dir, args.engines)
        output_json = json.dumps(result, indent=2, default=str)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_json)
            print(f"Benchmark results saved to {args.output}")
        else:
            print(output_json)
    elif args.image:
        result = process_image(args.image, args.engine)
        output_json = json.dumps(result, indent=2, default=str)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_json)
            print(f"Results saved to {args.output}")
        else:
            print(output_json)
        sys.exit(0 if result.get("status") == "ok" else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
