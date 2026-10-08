"""
Warung Lupi OCR Engine — Book Note Scanner
Main entry point for OCR processing.

Usage:
    python -m src --image <path_to_image> --output <json_output_path>
"""
import argparse
import json
import sys
from src.models.scan_result import ScanResult
from src.engine.engine_service import EngineService

def main():
    parser = argparse.ArgumentParser(description="Warung Lupi OCR Engine")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to output JSON (default: stdout)")
    parser.add_argument("--engine", default="auto", choices=["tesseract", "easyocr", "claude", "auto"],
                       help="OCR engine to use")
    parser.add_argument("--products", help="Path to products JSON for matching")
    parser.add_argument("--customers", help="Path to customers JSON for matching")
    parser.add_argument("--save-audit", action="store_true", help="Save audit image with overlays")
    
    args = parser.parse_args()
    
    service = EngineService()
    result = service.process(args.image, args.engine)
    
    # Phase 2-4 if matching data provided
    if args.products or args.customers:
        result = service.enhance(result, args.products, args.customers)
    
    # Output
    output = result.to_dict()
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(output, indent=2, default=str))
    
    # Return exit code based on confidence
    if result.status == "ok":
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
