# Warung Lupi OCR — Book Note Scanner

OCR engine for scanning handwritten book notes into digital receipts.

## Overview

This system scans photos of handwritten notebook pages from Warung Lupi's 
physical bookkeeping, interprets the handwriting, matches against product 
and customer databases, and creates draft transactions that the user 
confirms before saving to the Warung Lupi system.

## Architecture — 7-Phase Pipeline

```
Foto Buku → [Phase 1] → [Phase 2] → [Phase 3] → [Phase 4] → [Phase 5] → [Phase 6] → [Phase 9]
```

### Phase 1 — Image Understanding (OCR Engine)
- Multiple engine support: Tesseract (local/free), EasyOCR (local/free), Claude Vision (cloud/paid)
- Preprocessing: grayscale, CLAHE, adaptive threshold, denoising
- Structural analysis: strike-through detection (Hough Lines), circled numbers, layout analysis
- Output: `ScanResult` with regions, bbox, confidence

### Phase 2 — Handwriting Interpretation
- Line clustering (Y-coordinate based grouping)
- Date parsing (DD MON YYYY, DD/MM/YYYY, DD Month patterns)
- Customer detection (short text in header region)
- Item line parsing: product name, quantity, price extraction
- Noise filtering (removes OCR artifacts: lines, symbols, single chars)
- Conservative — uncertain fields are flagged, never silently guessed

### Phase 3 — Product Matching
- Fuzzy product matching against Warung Lupi database
- Three strategies: subsequence, Levenshtein ratio, token overlap
- Returns top-5 candidates with confidence scores
- Auto-select only if confidence > 0.85

### Phase 4 — Customer Matching
- Fuzzy customer name matching
- Substring matching (handles "Tami" → "Mbak Tami")
- Top-3 candidates with confidence scores

### Phase 5 — Draft Transaction
- `DraftTransaction` model (NOT a finalized transaction)
- Editable: quantity, price, product selection, add/remove items
- Audit trail: all interpretation steps preserved in `interpretation_log`

### Phase 6 — User Confirmation
- User reviews and edits the draft before finalizing
- Uncertain items flagged for review
- User can: change customer, edit items, add/remove items

### Phase 9 — API Integration
- Generates payload for `POST /api/transactions` (existing Laravel endpoint)
- Does NOT add new endpoints — uses existing API contract
- Bearer token auth (Laravel Sanctum)

## Engines

| Engine | Type | Accuracy | Speed | Cost | Notes |
|--------|------|----------|-------|------|-------|
| Tesseract 5.5 | Local | ~0.30 | Fast (~3s) | Free | Good for development baseline |
| EasyOCR 1.7 | Local | ~0.38-0.44 | Slow (~30-45s) | Free | Better for handwriting |
| **Gemini Vision** | Cloud | ~0.90+ | Fast (~5-15s) | ~Rp 2.000/img | **Production — Google AI Pro** |
| Claude Vision | Cloud | ~0.90+ | Fast (~5-10s) | ~$0.05/img | Alternative cloud option |

For production use, **Gemini Vision** (gemini-2.0-flash) is recommended:
- Set `GOOGLE_API_KEY` environment variable (from [Google AI Studio](https://aistudio.google.com/))
- Works with Google AI Pro plan (no additional cost beyond Pro subscription)

Claude Vision is an alternative (set `ANTHROPIC_API_KEY`).

## Quick Start

```bash
# Install dependencies
cd C:\laragon\www\warung-lupi-ocr
pip install -r requirements.txt

# Ensure Tesseract is installed at C:\Program Files\Tesseract-OCR\
# (or in PATH)

# Run on a single image
python -m src --image D:/dataset/IMG_20261008_115733.jpg --output result.json

# Run benchmark comparing both engines
python -m src --benchmark --images-dir D:/dataset

# Run full pipeline test
python scripts/test_pipeline.py --image D:/dataset/IMG_20261008_115733.jpg

```

## Project Structure

```
warung-lupi-ocr/
├── src/
│   ├── __main__.py              # CLI entry point
│   ├── engine/
│   │   ├── engine_service.py    # Engine orchestrator + EngineInterface
│   │   └── __init__.py
│   ├── engines/
│   │   ├── tesseract_engine.py  # Local Tesseract wrapper
│   │   ├── easyocr_engine.py    # Local EasyOCR wrapper
│   │   ├── claude_vision_engine.py  # Cloud Claude Vision (optional)
│   │   ├── gemini_vision_engine.py  # Cloud Gemini Vision (production)
│   │   └── __init__.py
│   ├── interpreter/
│   │   ├── service.py           # Phase 2: Interpretation
│   │   └── __init__.py
│   ├── draft/
│   │   ├── service.py           # Phase 5: Draft Transaction
│   │   └── __init__.py
│   ├── api/
│   │   ├── client.py            # Laravel API client
│   │   └── __init__.py
│   ├── models/
│   │   ├── scan_result.py       # Phase 1 data models
│   │   └── __init__.py
│   ├── preprocess/
│   │   └── processor.py         # Image preprocessing + symbol detection
│   ├── config.py                # OCRConfig, env vars, auto-detection
│   └── audit/
│       └── trail.py             # Phase 7: Audit trail (JSONL)
├── tessdata/
│   └── ind.traineddata          # Indonesian language for Tesseract
├── scripts/
│   ├── test_pipeline.py         # Integration test
│   ├── benchmark.py             # Engine comparison
│   └── _test_ocr_output.json    # Generated test output
├── eval/                        # Evaluation & improvement toolkit
│   ├── run_eval.py              # Evaluation harness (CER/WER/metrics)
│   ├── preprocess.py            # 8-step preprocessing module
│   ├── correct.py               # Post-OCR correction (date/price/qty parser)
│   ├── test_correct.py          # 27 unit tests for correction module
│   ├── ground_truth/            # Ground truth JSON files
│   ├── debug/                   # Debug images per preprocessing step
│   └── reports/                 # Evaluation reports (JSON)
├── config.example.json          # Config file template
├── requirements.txt
└── README.md
```

## Phase 2 Interpretation Heuristics

### Date Detection
- Pattern: `DD Month Year` → "02 September 2026"
- Pattern: `DD/Mon/YYYY` → "02/Sep/2026"  
- Pattern: `DD MMYYYY` with day name → "08 Okt Kamis"

### Quantity Detection
1. **Circled numbers** → high confidence quantity (detected via contour analysis)
2. **Small numbers** (1-99) near product keywords → likely quantity
3. **Large numbers** (1000+) → likely price
4. **Numbers near @ or Rp** → price

### Item Classification
Lines are classified as: `date`, `customer`, `item`, or `unknown`
- First 2 lines are checked for date/customer headers
- Items must have product keywords OR numbers with low line index

## API Integration

### Warung Lupi API Connection

The `WarungLupiAPIClient` connects to the Laravel backend:

```python
from src.api.client import WarungLupiAPIClient

client = WarungLupiAPIClient(
    base_url="http://localhost:8000",
    api_token="your_sanctum_token"
)

# Fetch product/customer database for matching
products = client.get_products()
customers = client.get_customers()

# Finalize draft transaction
payload = draft_service.confirm_draft(draft)
response = client.create_transaction(payload)
```

### API Endpoints Used (existing Laravel endpoints)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/products` | Fetch product database for Phase 3 matching |
| GET | `/api/customers` | Fetch customer database for Phase 4 matching |
| POST | `/api/transactions` | Create finalized transaction (Phase 9) |
| POST | `/api/transactions/{id}/items` | Add items to transaction |
| PUT | `/api/transactions/{id}` | Update transaction |

## Fine-Tuning Plan

The local engines (Tesseract + EasyOCR) achieve ~38-44% average confidence
on real notebook images. For production accuracy, use **Claude Vision**.

For fine-tuning local engines:
1. Use the benchmark script to evaluate on your dataset
2. Adjust preprocessing parameters per engine
3. Create training data from corrected results
4. Fine-tune EasyOCR model with custom handwriting data

## Testing

```bash
# Full pipeline test
python scripts/test_pipeline.py --engine easyocr

# Test all images in dataset
python scripts/test_pipeline.py --all-images --engine auto

# Process a single image
python -m src --image D:/dataset/IMG_20261008_115733.jpg --output result.json
```

## Development

```bash
# Run linter
python -m py_compile src/

# Run a specific phase test
python -c "
from src.interpreter.service import InterpretationService
from src.draft.service import DraftTransactionService
# ... test code
"
```
