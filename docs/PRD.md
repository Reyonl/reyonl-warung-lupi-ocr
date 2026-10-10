# PRD: Warung Lupi OCR — Book Note Scanner

> **Status:** Draft ✅ (Review)
> **Dokumen ini adalah sumber otropati untuk tim engineering. Revisi dilacat pada GitHub commit/PR.)
> **Repo:** `github.com/Reyonl/warung-lupi-ocr` (private, kontak Reyon untuk akses)
> **Dibuat:** 2026-10-10  
> **Product Owner:** Reyon Lau Jiemin  
> **Authoring Method:** Iterative verification (ground-truth first → benchmark → correct)

---

## 1. Ringkasan Eksekif (Executive Summary)

**Warung Lupi OCR** adalah sebuah sistem *optical character recognition* (OCR) khusus yang mendeteksi foto buku catatan tulisan tangan (handwritten notebooks) di sebuah warung makan di Indonesia, lalu mentransformasikannya menjadi transaksi digital terstruktur. Sistem ini memindai foto buku catatan fisik, menginterpretasi tulisan tangan, mencocokkannya dengan database produk dan pelanggan, dan membuat draft transaksi yang dapat dikonfirmasi pengguna sebelum disimpan ke sistem Warung Lupi.

- **Target akurasi:** ≥85% akurasi produk & pelanggan (dengan Gemini Vision)
- **Waktu proses gambar:** ≤15s (cloud) / ≤5s (lokal Tesseract)
- **Platform:** Python (lokal + cloud engine), API Laravel backend
- **Pengguna akhir:** Pemilik/pegawai warung Lupi

## 2. Latar Belakang & Masalah

### Konteks Bisnis
Warung Lupi mencatat transaksi harian di buku catatan fisik dengan tulisan tangan. Proses ini:
- Rentan salah baca / lupa dicatat
- Sulit dianalisis untuk laporan penjualan
- Perlu diketik ulang ke sistem digital

### Masalah Teknis
1. **Tulisan tangan tangan** → tinggi variasi font, bebas, bercoret
2. **Layout buku** → multi-pelanggan per halaman, harga dalam format "X ribu"
3. **Ketersediaan mesin lokal** → Tesseract tidak akurat (CER ~1.0), EasyOCR lambat (30-45s/img)
4. **Konsumsi API cloud** → butuh koneksi internet + Google AI Pro / Claude Vision berlangganan

### Asumsi
- Dataset gambar (≈31 gambar) sudah tersedia di `D:/dataset/`
- Laravel backend sudah memiliki endpoint `/api/products`, `/api/customers`, `/api/transactions`
- Pemilik punya akun Google AI Pro (GOOGLE_API_KEY) untuk produksi

## 3. User Stories

| # | Sebagai | Saya ingin | Agar bisa |
|---|---------|------------|-----------|
| US-01 | Pegawai warung | Memindai satu foto buku | Membuat draft transaksi otomatis |
| US-02 | Pegawai warung | Melihat semua item yang terdeteksi | Memastikan tidak ada yang missing |
| US-03 | Pegawai warung | Mengoreksi item yang salah (harga, qty, produk) | Draft akurat sebelum disave |
| US-04 | Owner | Membandingkan akurasi engine (Tesseract vs EasyOCR vs Gemini) | Memilih engine yang paling baik |
| US-05 | Owner | Mengekspor draft ke Laravel | Transaksi masuk ke sistem |
| US-06 | Developer | Men-view CER/WER tiap engine | Mengetahui engine terbaik |

## 4. Ruang Lingkup (Scope)

### In Scope ✅
- Pipeline OCR 7-fase (lihat sebab 1.1)
- Preprocessing gambar (8 langkah toggleable)
- Ground truth builder (HTML canvas untuk labeling bbox)
- Eval harness (CER/WER/acuracy per engine)
- Post-OCR correction (date/price/quantity parser)
- Product & customer fuzzy matching (trigram + phonetic + alias)
- Draft transaction service (editable)
- Audit trail (JSONL log tiap langkah)
- Integrasi API Laravel (payload generation)
- Engine fallback sequence: `gemini → claude → easyocr → tesseract`

### Out of Scope ❌
- Fine-tuning EasyOCR model (butuh training data + GPU)
- Endpoint API baru di Laravel (hanya gunakan existing contract)
- UI frontend desktop (CLI only)
- Batch processing UI (gunakan terminal)
- Fine-tuning Tesseract (hanya kirim `--psm`/`--oem`)

## 5. Arsitektur & Flow

### 5.1 Pipeline 7 Fase

```
Foto Buku → [F1 OCR] → [F2 Interpretasi] → [F3 Product Match] → [F4 Customer Match] → [F5 Draft] → [F6 Konfirmasi] → [F9 API Integrate]
```

| Fase | Nama | Komponen | Output |
|------|------|----------|--------|
| 1 | Image Understanding | `engine/engine_service.py`, `engines/*`, `preprocess/processor.py` | `ScanResult` (regions, confidence, bbox) |
| 2 | Interpretation | `interpreter/service.py` | Parsed lines, date, items, qty, price |
| 3 | Product Matching | `matching/improved_matching.py` | Matched product + confidence |
| 4 | Customer Matching | `matching/improved_matching.py` | Matched customer + confidence |
| 5 | Draft Transaction | `draft/service.py` | `DraftTransaction` (editable) |
| 6 | User Confirmation | CLI review | Confirmed draft |
| 7 | Audit Trail | `audit/trail.py` | JSONL log tiap aksi |

### 5.2 Engine Selection Strategy

Prioritas saat `engine=auto`:
```
gemini (GOOGLE_API_KEY) > claude (ANTHROPIC_API_KEY) > easyocr (local) > tesseract (local)
```

#### Engine Detail

| Engine | Type | Akurasi* | Kecepatan | Cost | Catatan |
|--------|------|----------|-----------|------|---------|
| Tesseract 5.5 | Local | ~0.30 | ~3s | Free | Baseline dev |
| EasyOCR 1.7 | Local | ~0.38-0.44 | ~30-45s | Free | Lebih baik handwriting |
| **Gemini Vision** | Cloud | ~0.85+ | ~5-15s | ~Rp 2.000/img | **Produksi** (Google AI Pro) |
| Claude Vision | Cloud | ~0.85+ | ~5-10s | ~$0.05/img | Alternative |

> \* Akurasi diukur dengan product-date-customer accuracy (bukan CER mentah) pada dataset 2 gambar ground truth.

#### Gemini Model Selection (live-tested)
1. `gemini-3.5-flash` — paling handal, jarang 503
2. `gemini-3.7-flash` — reliable, tier sama
3. `gemini-3.8-flash` — kualitas lebih tinggi tapi *sering* 503 UNAVAILABLE
4. `gemini-flash-latest` — availability tak pasti

**Pitfall:** `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-pro` → 404 (retired/wrong tier). Jangan hardcode. Probe `client.models.list()` di startup.

### 5.3 Cloud Fallback (Hybrid Pipeline)
- Jalankan engine lokal dulu (Tesseract/EasyOCR)
- Jika confidence < `cloud_fallback_threshold` (default **0.45**) OR ada field `needs_review` → panggil engine cloud
- Merge: gunakan cloud result untuk region low-confidence saja, pertahankan local result untuk high-confidence
- Validasi JSON schema sebelum mempercayai respons

## 6. Struktur Data

### 6.1 ScanResult (Phase 1 Output)
```python
ScanResult(
  raw_text: str,
  confidence: float,
  regions: [Region(text, bbox, confidence, type, annotations)],
  layout: LayoutInfo,
  preprocessing: PreprocessingInfo,
  warnings: [str]
)
```

### 6.2 Ground Truth Format
```json
{
  "image": "IMG_20261008_115733.jpg",
  "date": "08 OKT",
  "currency": "IDR",
  "sections": [
    {
      "customer": "Kris",
      "items": [
        {
          "product": "Es Cincau",
          "quantity": 1,
          "unit_price": 3000,
          "raw_line": "es jasjus cincau 3 ribu",
          "bbox": [2049.5, 413.03, 513.7, 67.9]
        }
      ]
    }
  ]
}
```

### 6.3 Laravel API Payload
```python
# POST /api/transactions
{
  "customer_id": 123,       # dari matching, nullable
  "transaction_date": "2026-10-08",  # ISO format parsed dari "08 OKT"
  "notes": "OCR scan via Gemini Vision"
}
# POST /api/transactions/{id}/items
{
  "product_name": "Indomie",   # required string
  "quantity": 1,
  "unit_price": 7000,          # INTEGER (float → validation error!)
  "product_id": 45             # nullable
}
```

## 7. Preprocessing (Langkah 8, Toggleable)

| Step | Nama | Effect |
|------|------|--------|
| 1 | EXIF Auto-Rotate | Rotate otomatis dari metadata |
| 2 | Page Crop | Potong tepi putih |
| 3 | Deskew | Koreksi miring halaman |
| 4 | Lighting (CLAHE+morph) | Normalisasi pencahayaan |
| 5 | Denoise | `fastNlMeansDenoisingColored` |
| 6 | Ruled-line Removal | Hapus garis bimbingan kertas |
| 7 | Binarize | Threshold adaptif |
| 8 | Upscale | Resize ke ~1600px lebar |

> **Pitfall:** `normalize_lighting` mengembalikan grayscale tapi `fastNlMeansDenoisingColored` butuh 3-channel BGR. Cek `len(img.shape)`.

## 8. Post-OCR Correction + Matching

### 8.1 PostOCRCorrector (`eval/correct.py`)
- **Date parser**: `DD MON YYYY`, `DD/Mon/YYYY`, `DD Month Year`
- **Price parser**: `"X ribu"` → `X×1000`, `"X rb"` → `X×1000`, `"Xk"` → `X×1000` (misal `"5k"` → `5000`)
- **Quantity parser**: regex `\d+x`, `\d+ bks`, `\d+ pcs`, circled numbers via contour
- **Char confusion fix**: OCR noise substitution
- **Noise filter**: hapus simbol garis/bintang/structural

### 8.2 ImprovedMatcher (`eval/matching.py`)
| Strategi | Deskripsi |
|----------|-----------|
| Trigram | 3-char n-gram Jaccard (lebih baik dari SequenceMatcher untuk OCR noise) |
| Phonetic | Vowel-normalized consonant skeleton + length ratio |
| Alias | `indomi` → `indomie`, `gls` → `aqua gelas` |
| Price boost | Jika harga dalam 15% dari catalog default → confidence +0.15 |

> **Threshold:** `>0.55` overall, `>0.95` untuk phonetic. Filter lines < 2 karakter.

## 9. Evaluation Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| CER (Character Error Rate) | edit_dist(pred, gt) / len(gt) | Tesseract ≤ 1.0, Gemini ≤ 0.15 |
| WER (Word Error Rate) | edit_dist_words / len(gt_words) | Gemini ≤ 0.20 |
| Date Accuracy | % gambar dengan date parse benar | ≥95% (Gemini) |
| Product Match Accuracy | % item product yang match ke catalog | ≥90% (Gemini) |
| Customer Match Accuracy | % customer yang match | ≥90% (Gemini) |
| Processing Time/img | rata-rata waktu | ≤15s (Gemini), ≤5s (Tesseract) |

## 10. Environment & Setup

### Dependencies (`requirements.txt`)
```txt
easyocr>=1.7.0
pytesseract>=0.3.10
opencv-python>=4.9.0.80
pillow>=10.0.0
numpy>=1.26.0
scikit-learn>=1.3.0
# Cloud (opsional):
google-genai>=0.8.0   # Gemini Vision
anthropic>=0.34.0      # Claude Vision
```

### Environment Variables
| Var | Wajib? | Deskripsi |
|-----|--------|-----------|
| `GOOGLE_API_KEY` | Opsional | Untuk Gemini Vision (produksi) |
| `ANTHROPIC_API_KEY` | Opsional | Untuk Claude Vision |
| `LARAVEL_API_TOKEN` | Opsional | Bearer token Laravel Sanctum |
| `WLOCR_ENGINE` | Opsional | `gemini`/`tesseract`/`easyocr`/`auto` |
| `TESSDATA_PREFIX` | Ya (lokal) | Path ke `ind.traineddata` |

### Instalasi
```bash
cd C:\laragon\www\warung-lupi-ocr
pip install -r requirements.txt
# Tesseract harus ada di C:\Program Files\Tesseract-OCR\
# google-genai + anthropic diperlukan untuk cloud engines
```

### Quick Start (penggunaan)
```bash
# Single image (lokal — Tesseract)
python -m src --image D:/dataset/IMG_20261008_115733.jpg --output result.json

# Benchmark engine
python scripts/test_pipeline.py --engine easyocr
python scripts/test_pipeline.py --all-images --engine auto

# Gemini Vision batch (perlu GOOGLE_API_KEY)
python scripts/run_gemini_pipeline.py --limit 5 --dry-run

# Kirim ke Laravel
python scripts/send_to_laravel.py --limit 40 --dry-run
```

## 11. Ground Truth & Dataset

### Dataset
- **Lokasi:** `D:/dataset/`
- **Jumlah:** 31 gambar (IMG_20261008_115733.jpg ~ 120052.jpg)
- **Format:** JPEG, ~3MB per image

### Ground Truth yang sudah dilabel
| Gambar | Date | Customers | Items | Status |
|---------|------|-----------|-------|--------|
| 115733 | 08 OKT | 4 | 14 | ✅ Verified |
| 115737 | 8 Oktober | 11 | 46 | ✅ Verified |
| 115740 | 2026-10-08 | ? | 1 (Kopi) | ⚠️ OCR-derive (needs verification) |
| 115742 | ? | ? | ? | ⚠️ OCR-derive (needs verification) |
| 115746 | ? | ? | ? | ⚠️ OCR-derive (needs verification) |

> **Ground truth builder:** `eval/ground_truth_builder.html` — canvas overlay untuk interactive bbox. Shortcut `Ctrl+E` untuk export ke JSON.

## 12. Rencana Pengembangan (Roadmap)

Angka ini mencerminkan commit git log hingga 2026-10-09. Progress sebenarnya:
- [x] LANGKAH 1 — Preprocessing module + Ablation testing + Baseline eval
- [x] LANGKAH 5 — Post-OCR Correction + Unit Tests + Evaluation Report
- [x] LANGKAH 6 — Trigram + Phonetic + alias-based matching
- [x] LANGKAH 7 — Audit trail + Config management + E2E test
- [x] LANGKAH 8 — Gemini Vision Engine + retry/fallback + unit tests
- [x] LANGKAH 9 — 2 ground truth images + EasyOCR benchmark
- [x] LANGKAH 10 — Gemini auto-registration to eval harness

### Rencana Selanjutnya
- [ ] **LANGKAH 11** — Jalankan Gemini Vision batch pada 5 gambar (tergantung API key) + bandingkan dengan ground truth
- [ ] **LANGKAH 12** — Perbaiki post-correction rule jika perlu (lihat hasil Gemini)
- [ ] **LANGKAH 13** — Finalisasi 3 ground truth tambahan (115740/42/46)
- [ ] **LANGKAH 14** — Laravel integration end-to-end (dry-run → live)

## 13. Pengujian (Testing)

| Test | Script | Deskripsi |
|------|--------|-----------|
| Unit Correction | `python -m pytest eval/test_correct.py` | 27 unit tests untuk date/price/qty parser |
| Pipeline | `python scripts/test_pipeline.py --single` | Full E2E pada 1 image |
| Benchmark | `python scripts/test_pipeline.py --all-images` | Banding engine semua image |
| Dry-run Laravel | `python scripts/send_to_laravel.py --dry-run` | Validasi payload format |

> **Pitfall:** `run_eval.py` HARUS panggil `OCRConfig.load()` + `service.auto_register_default_engines(config)`. EngineService biasa akan fallback ke EasyOCR meski `GOOGLE_API_KEY` sudah set.

## 14. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
|--------|--------|----------|
| Gemini 503 UNAVAILABLE | Pipeline gagal | Fallback ke model 3.5/3.7, retry 3x + backoff |
| Local engine akurasi rendah | OCR noise tinggi | Cloud engine sebagai fallback (threshold 0.45) |
| Tesseract ind.traineddata missing | Date parse salah | Set `TESSDATA_PREFIX` env var |
| Laravel API format | Validation error | `--dry-run` dulu; `unit_price` INTEGER not float |
| API key leak | Security breach | Env var saja, committed ke .gitignore |

## 15. Dokumen Referensi
1. `SKILL.md` — Skill OCR pipeline (engine quirks, SDK tips)
2. `eval/run_eval.py` — Eval harness (CER/WER metrics)
3. `scripts/benchmark_preprocessing.py` — Preprocessing ablation
4. `eval/architecture.html` — Diagram arsitektur visual
5. `references/tesseract_psm_oem.md` — PSM/OEM cheat sheet
6. `references/matching_techniques.md` — Trigram, phonetic, alias