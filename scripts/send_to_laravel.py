#!/usr/bin/env python
"""
Gemini Vision -> Laravel API Integration.
Translates handwritten notebook OCR sections into individual customer transactions/bon in Laravel POS.
"""
import argparse
import glob
import json
import logging
import os
import re
import sys
import time
from datetime import datetime

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from src.matching.improved_matching import ImprovedMatcher

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("send_to_laravel")

DEFAULT_PRODUCTS = [
    {"id": 1, "name": "Kopi", "default_price": 5000},
    {"id": 2, "name": "Es Kopi", "default_price": 7000},
    {"id": 3, "name": "Teh", "default_price": 4000},
    {"id": 5, "name": "Es Teh", "default_price": 5000},
    {"id": 6, "name": "Aqua K", "default_price": 3000},
    {"id": 7, "name": "Aqua Galon", "default_price": 6000},
    {"id": 8, "name": "Susu", "default_price": 5000},
    {"id": 14, "name": "Mie", "default_price": 8000},
    {"id": 17, "name": "Donat", "default_price": 2000},
    {"id": 21, "name": "Gorengan", "default_price": 2500},
    {"id": 22, "name": "Rokok", "default_price": 25000},
    {"id": 24, "name": "Item Lainnya", "default_price": 5000},
]

DEFAULT_CUSTOMERS = [
    {"id": 9, "name": "Kris"},
    {"id": 10, "name": "Bang Lee"},
    {"id": 14, "name": "Rama"},
    {"id": 32, "name": "Marco"},
    {"id": 20, "name": "Mama Sasa"},
]

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "mei": 5, "may": 5, "jun": 6,
    "jul": 7, "ags": 8, "aug": 8, "sep": 9, "okt": 10, "nov": 11, "des": 12,
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11, "desember": 12,
}


def parse_date(s: str) -> str:
    """Parse various Indonesian date strings into ISO format (YYYY-MM-DD)."""
    if not s:
        return datetime.now().strftime("%Y-%m-%d")
    s = s.strip()
    m = re.match(r"(\d{1,2})\s+([A-Za-z]+)", s)
    if m:
        day = int(m.group(1))
        month_name = m.group(2).lower()[:3]
        month = MONTH_MAP.get(month_name)
        if month:
            ym = re.search(r"(\d{4})", s)
            year = int(ym.group(1)) if ym else datetime.now().year
            return f"{year:04d}-{month:02d}-{day:02d}"
    m = re.match(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})", s)
    if m:
        yr = int(m.group(3))
        if yr < 100:
            yr += 2000
        return f"{yr:04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return datetime.now().strftime("%Y-%m-%d")


def fetch_live_catalogs(api_url: str):
    """Fetch real products and customers directly from Laravel REST API."""
    import requests
    headers = {"Accept": "application/json"}
    customers = DEFAULT_CUSTOMERS
    products = DEFAULT_PRODUCTS

    try:
        r = requests.get(f"{api_url}/api/customers", headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                customers = data
                logger.info(f"Loaded {len(customers)} customers from Laravel API")
    except Exception as e:
        logger.warning(f"Using default customers (could not fetch from {api_url}/api/customers: {e})")

    try:
        r = requests.get(f"{api_url}/api/products", headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                products = data
                logger.info(f"Loaded {len(products)} products from Laravel API")
    except Exception as e:
        logger.warning(f"Using default products (could not fetch from {api_url}/api/products: {e})")

    return products, customers


def get_or_create_customer(name: str, matcher: ImprovedMatcher, api_url: str, dry_run: bool = False) -> int:
    """Find customer ID via ImprovedMatcher or auto-create in Laravel."""
    import requests
    matched_name, conf, _ = matcher.match_customer(name)
    if matched_name and conf >= 0.70:
        for c in matcher.customer_catalog:
            if c["name"].lower() == matched_name.lower():
                return c["id"]

    # Fallback search
    for c in matcher.customer_catalog:
        if c["name"].lower() == name.lower() or name.lower() in c["name"].lower():
            return c["id"]

    if dry_run:
        return 999  # Mock ID for dry-run

    # Auto-create new customer in Laravel
    try:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        res = requests.post(
            f"{api_url}/api/customers",
            json={"name": name.strip(), "notes": "Auto-created from OCR scan"},
            headers=headers,
            timeout=10,
        )
        if res.status_code in [200, 201]:
            new_cust = res.json()
            new_id = new_cust.get("id")
            matcher.customer_catalog.append({"id": new_id, "name": name.strip()})
            logger.info(f"  + Created new customer in Laravel: '{name}' (ID: {new_id})")
            return new_id
    except Exception as e:
        logger.warning(f"Could not auto-create customer '{name}': {e}")

    # Fallback to first customer in list
    return matcher.customer_catalog[0]["id"] if matcher.customer_catalog else 1


def convert_section_to_transaction(
    section: dict,
    date_str: str,
    img_name: str,
    matcher: ImprovedMatcher,
    api_url: str,
    dry_run: bool = False,
):
    """Convert a single customer section into a Laravel transaction payload and its items."""
    raw_customer = section.get("customer", "Umum").strip()
    customer_id = get_or_create_customer(raw_customer, matcher, api_url, dry_run=dry_run)

    # Standard default prices for common warung items if missing
    FALLBACK_PRICES = {
        "kopi": 5000,
        "es kopi": 7000,
        "teh": 4000,
        "es teh": 5000,
        "aqua k": 3000,
        "aqua": 3000,
        "aqua galon": 6000,
        "galon": 6000,
        "susu": 5000,
        "gorengan": 2500,
        "donat": 2000,
        "mie": 8000,
        "pop mie": 6000,
        "rokok": 25000,
        "es batu": 1000,
        "cimory": 9000,
        "roti": 3000,
    }

    items = []
    total_amount = 0

    for item in section.get("items", []):
        raw_product = (item.get("product") or item.get("raw_line") or "Item Lainnya").strip()
        qty = max(1, int(item.get("quantity") or 1))

        # Determine price
        price = item.get("unit_price")
        if price is not None and int(price) > 0:
            unit_price = int(round(float(price)))
        else:
            # Fallback lookup
            clean_name = raw_product.lower()
            matched_price = None
            for key, val in FALLBACK_PRICES.items():
                if key in clean_name:
                    matched_price = val
                    break
            unit_price = matched_price if matched_price else 5000

        # Match product in catalog
        matched_prod, conf, _ = matcher.match_product(raw_product, price=unit_price)
        prod_id = None
        final_name = raw_product

        if matched_prod and conf >= 0.60:
            final_name = matched_prod
            for p in matcher.product_catalog:
                if p["name"].lower() == matched_prod.lower():
                    prod_id = p["id"]
                    break

        subtotal = qty * unit_price
        total_amount += subtotal

        items.append({
            "product_id": prod_id,
            "product_name": final_name,
            "description": f"OCR text: '{item.get('raw_line', raw_product)}'",
            "quantity": qty,
            "unit": "pcs",
            "unit_price": unit_price,
            "subtotal": subtotal,
        })

    transaction_payload = {
        "customer_id": customer_id,
        "transaction_date": date_str,
        "notes": f"Bon OCR: {raw_customer} | Foto: {img_name}",
    }

    return transaction_payload, items, total_amount, raw_customer


def send_transaction_to_laravel(payload: dict, items: list, api_url: str, token: str = "", dry_run: bool = False):
    """Send transaction and its items to Laravel REST API."""
    import requests
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if dry_run:
        logger.info(f"    (DRY-RUN) POST /api/transactions: {json.dumps(payload, ensure_ascii=False)}")
        for it in items:
            logger.info(f"      + (DRY-RUN) item: {it['product_name']} x{it['quantity']} @Rp{it['unit_price']:,}")
        return {"dry_run": True, "transaction_id": 999, "items_saved": len(items)}

    # 1. Create Transaction
    resp = requests.post(f"{api_url}/api/transactions", json=payload, headers=headers, timeout=15)
    if resp.status_code not in [200, 201]:
        logger.error(f"    FAILED /api/transactions [{resp.status_code}]: {resp.text[:300]}")
        return {"error": resp.text[:300]}

    txn_data = resp.json()
    txn_id = (
        txn_data.get("id")
        or txn_data.get("data", {}).get("id")
        or txn_data.get("data", {}).get("transaction", {}).get("id")
    )
    if not txn_id:
        logger.error(f"    Could not extract transaction ID from response: {txn_data}")
        return {"error": "Missing transaction id in response", "response": txn_data}

    logger.info(f"    ✓ Transaksi Created ID: {txn_id} (No: {txn_data.get('transaction_number', txn_id)})")

    # 2. Add Items
    items_saved = 0
    for it in items:
        r = requests.post(f"{api_url}/api/transactions/{txn_id}/items", json=it, headers=headers, timeout=10)
        if r.status_code in [200, 201]:
            items_saved += 1
            logger.info(f"      + {it['product_name']} x{it['quantity']} @Rp{it['unit_price']:,} = Rp{it['subtotal']:,}")
        else:
            logger.warning(f"      ✗ Item fail [{r.status_code}]: {it['product_name']} -> {r.text[:150]}")

    return {"transaction_id": txn_id, "items_saved": items_saved, "items_total": len(items)}


def main():
    parser = argparse.ArgumentParser(description="Send OCR detection results to Laravel POS.")
    parser.add_argument("--from-json", type=str, default=None, help="Path to existing Gemini result JSON")
    parser.add_argument("--latest", action="store_true", help="Use the latest result in eval/results/")
    parser.add_argument("--limit", type=int, default=1, help="Max images to process if calling Gemini")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without writing to DB")
    parser.add_argument("--api-url", type=str, default=None, help="Laravel base URL")
    args = parser.parse_args()

    api_url = args.api_url or os.environ.get("LARAVEL_API_URL", "http://127.0.0.1:8000")
    token = os.environ.get("LARAVEL_API_TOKEN", "")

    print(f"\n=======================================================")
    print(f"       WARUNG LUPI OCR -> LARAVEL POS SYNC")
    print(f"=======================================================")
    print(f"  Target API : {api_url}")
    print(f"  Mode       : {'DRY-RUN (Simulasi)' if args.dry_run else 'LIVE (Masuk ke Database)'}")

    # Fetch live catalogs
    products, customers = fetch_live_catalogs(api_url)
    matcher = ImprovedMatcher(products, customers)

    # Determine input data
    results = []
    if args.from_json and os.path.isfile(args.from_json):
        print(f"  Source     : {args.from_json}")
        with open(args.from_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            results = data if isinstance(data, list) else [data]
    elif args.latest:
        json_files = sorted(glob.glob("eval/results/gemini_*.json"))
        if json_files:
            latest_file = json_files[-1]
            print(f"  Source     : {latest_file} (Latest Gemini output)")
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                results = data if isinstance(data, list) else [data]
        else:
            print("  [ERROR] No result JSON found in eval/results/")
            return
    else:
        # Check latest json by default
        json_files = sorted(glob.glob("eval/results/gemini_*.json"))
        if json_files:
            latest_file = json_files[-1]
            print(f"  Source     : {latest_file} (Auto-detected latest result)")
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                results = data if isinstance(data, list) else [data]
        else:
            print("  [INFO] Calling Gemini Vision live on dataset...")
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                print("  [ERROR] GOOGLE_API_KEY is not set.")
                return
            from scripts.run_gemini_pipeline import process_images
            images = sorted(glob.glob("D:/dataset/IMG_*.jpg"))[:args.limit]
            results = process_images(images, api_key)

    if not results:
        print("  [ERROR] No results to process.")
        return

    print(f"\n[Processing {len(results)} images/results...]")
    total_txns_created = 0
    total_items_created = 0

    for idx, r in enumerate(results, 1):
        if r.get("error"):
            print(f"  SKIP {r.get('_image', idx)}: error {r.get('error')}")
            continue

        img_name = r.get("_image", f"Image_{idx}")
        date_iso = parse_date(r.get("date", ""))
        sections = r.get("sections", [])
        print(f"\n-------------------------------------------------------")
        print(f" Foto : {img_name} (Tanggal: {date_iso})")
        print(f" Ditemukan: {len(sections)} pelanggan / bon terpisah")
        print(f"-------------------------------------------------------")

        for s_idx, sec in enumerate(sections, 1):
            payload, items, total_amount, raw_cust = convert_section_to_transaction(
                sec, date_iso, img_name, matcher, api_url, dry_run=args.dry_run
            )

            print(f"\n [{s_idx}/{len(sections)}] BON PELANGGAN: {raw_cust.upper()} (Customer ID: {payload['customer_id']})")
            print(f"    Jumlah Item : {len(items)} | Total: Rp {total_amount:,}")

            res = send_transaction_to_laravel(payload, items, api_url, token, dry_run=args.dry_run)
            if res.get("transaction_id"):
                total_txns_created += 1
                total_items_created += res.get("items_saved", len(items))

    print(f"\n=======================================================")
    print(f"                 RINGKASAN SELESAI")
    print(f"=======================================================")
    print(f"  Total Transaksi/Bon Dibuat : {total_txns_created}")
    print(f"  Total Item Masuk           : {total_items_created}")
    print(f"  Mode                       : {'DRY-RUN' if args.dry_run else 'LIVE DATABASE BERHASIL'}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
