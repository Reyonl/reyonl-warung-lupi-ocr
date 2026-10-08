# SKILL: Warung Lupi OCR Pipeline

Use when building an OCR pipeline that scans handwritten notebook images 
into structured digital transactions, with fuzzy matching against a product 
and customer database, draft editing, and audit trail.

## Architecture (9 phases)

```
Foto Buku → Phase 1 (OCR) → Phase 2 (Interpretation) → Phase 3 (Product Match) 
          → Phase 4 (Customer Match) → Phase 5 (Draft) → Phase 6 (UI Confirmation)
          → Phase 9 (API Finalize)
```

Phase 7 (Audit Trail) wraps all phases with a shared trace_id.

## Engine Selection

| Engine | Type | When to use |
|--------|------|-------------|
| Tesseract | Local/free | Development baseline, fast |
| EasyOCR | Local/free | Handwriting, moderate accuracy |
| Claude Vision | Cloud/paid | Production quality (set `ANTHROPIC_API_KEY`) |

Auto-selects: if `ANTHROPIC_API_KEY` env is set, use Claude Vision; else use EasyOCR.

## Quick Start

```python
from src import OCRConfig, EngineService, InterpretationService, DraftTransactionService

config = OCRConfig.load()  # auto-detects from env/file
service = EngineService()
interpreter = InterpretationService()
draft_service = DraftTransactionService()

# Phase 1: OCR
scan = service.process("IMG_20261008_115733.jpg", engine=config.engine.preferred_engine)

# Phase 2-4: Interpret + Match
interpreted = interpreter.interpret(scan)
interpreted = interpreter.enhance_with_products(interpreted, products)   # list of dicts
interpreted = interpreter.enhance_with_customers(interpreted, customers)

# Phase 5: Draft
draft = draft_service.create_draft_from_interpretation(interpreted)
draft = draft_service.apply_product_defaults(draft, products)

# Phase 9: API payload (for Laravel POST /api/transactions)
payload = draft_service.confirm_draft(draft)
```

## Pitfalls & Gotchas

- **Symbol objects vs dicts**: `Annotation.symbols` contains `Symbol` dataclass objects
  (has `.text` attribute), NOT dicts with `.get("text")`. Access directly: `s.text`.

- **f-string escape**: Never put `{}` or `[]` chars directly in f-strings for 
  Tesseract config. Assign to a variable first: `bl = "|<>[]{}"` then use `f"...{bl}..."`.

- **TESSDATA_PREFIX**: Must be set via `os.environ['TESSDATA_PREFIX']`, not CLI flag
  `--tessdata-prefix`. The CLI flag syntax is `--tessdata-prefix=PATH` but it 
  sometimes doesn't work; env var is more reliable.

- **numpy mean on generator**: `np.mean(genexpr)` fails — must use `np.mean([...])` 
  (list, not generator).

- **Python module caching**: After editing `.py` files, clear `__pycache__` dirs 
  and remove from `sys.modules` before re-importing. Python may load stale bytecode.

- **Working directory**: Shell sessions can get stuck in old dirs. Always use 
  `os.chdir()` or `workdir` parameter.

- **Laravel API format**: 
  - Transaction: `transaction_date` field (NOT `date`), `customer_id` required.
  - Items: `product_name` is **required** by Laravel validation, even if 
    `product_id` is provided. `unit_price` must be integer.

- **Noise filtering is critical**: OCR on notebook handwriting produces 40-70% noise.
  Always filter regions before interpretation: remove single-char noise, pure 
  symbol lines (—, |, +), and confidence < 0.35 regions.

- **EasyOCR is slow**: ~30-45s per image (model load + OCR). Cache the EasyOCR 
  reader across calls rather than recreating it each time.

- **Date parsing**: Indonesian dates may not include year. Default to current year
  with lower confidence (0.65-0.70).

- **Number ambiguity**: Small numbers (1-99) could be quantity OR price. Use 
  context: circled numbers → quantity, numbers near `@` or `Rp` → price.

## References
- Dataset: `D:\dataset` (40 JPG images of Warung Lupi notebook pages)
- Laravel backend: `C:\laragon\www\Rekapan_Warung`
- Laravel routes: `routes/api.php`
- Flutter app: `C:\laragon\www\rekapan_warung_flutter`
- GitHub: `reyonl/reyonl-warung-lupi-ocr`
