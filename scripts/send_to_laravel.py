#!/usr/bin/env python
"""Gemini Vision -> Laravel API Integration."""
import argparse, json, os, sys, glob, re, time
from datetime import datetime

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)
os.environ["TESSDATA_PREFIX"] = os.path.abspath("tessdata")

from eval.matching import ImprovedMatcher

DEFAULT_PRODUCTS = [
    {"id": 1, "name": "Kopi", "default_price": 5000},
    {"id": 2, "name": "Teh", "default_price": 4000},
    {"id": 3, "name": "Madu", "default_price": 15000},
    {"id": 4, "name": "Rokok", "default_price": 15000},
    {"id": 5, "name": "Aqua", "default_price": 3000},
    {"id": 6, "name": "Indomie", "default_price": 8000},
    {"id": 7, "name": "Donat", "default_price": 2000},
    {"id": 8, "name": "Makan", "default_price": 12000},
    {"id": 9, "name": "Gorengan", "default_price": 2500},
    {"id": 10, "name": "Es", "default_price": 3000},
    {"id": 11, "name": "Es Cream", "default_price": 3000},
    {"id": 12, "name": "Cimory", "default_price": 5000},
    {"id": 13, "name": "Roti", "default_price": 3000},
]
DEFAULT_CUSTOMERS = [
    {"id": 14, "name": "Rama"}, {"id": 27, "name": "Marco"},
    {"id": 28, "name": "Bu Sri"}, {"id": 32, "name": "Mama Sasa"},
    {"id": 1, "name": "KRIS"},
]
MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "okt": 10, "nov": 11, "des": 12,
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5,
    "juni": 6, "juli": 7, "agustus": 8, "september": 9, "oktober": 10,
}


def parse_date(s):
    if not s:
        return datetime.now().strftime("%Y-%m-%d")
    m = re.match(r"(\d{1,2})\s+([A-Za-z]+)", s.strip())
    if m:
        day = int(m.group(1))
        month = MONTH_MAP.get(m.group(2).lower()[:3])
        if month:
            ym = re.search(r"(\d{4})", s)
            year = int(ym.group(1)) if ym else datetime.now().year
            return f"{year:04d}-{month:02d}-{day:02d}"
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return datetime.now().strftime("%Y-%m-%d")


def find_cust(name, customers):
    for c in customers:
        if c["name"].lower() == name.lower() or name.lower() in c["name"].lower():
            return c["id"]
    return None


def find_prod(name, catalog):
    for p in catalog:
        if p["name"].lower() == name.lower():
            return p["id"]
    return None


def convert(gemini_result, matcher):
    date = parse_date(gemini_result.get("date", ""))
    sections = gemini_result.get("sections", [])
    cust_name = sections[0].get("customer", "") if sections else ""
    cust_id = find_cust(cust_name, matcher.customer_catalog)

    items = []
    total = 0
    for section in sections:
        for item in section.get("items", []):
            qty = max(1, item.get("quantity", 1))
            price = item.get("unit_price") or 0
            if price and price > 1000:
                price = int(price)
            else:
                price = 0
            matched, conf, method = matcher.match_product(item.get("product", ""), price=price if price else None)
            pname = matched if matched else item.get("product", "").strip()
            pid = find_prod(matched, matcher.product_catalog) if matched else None
            items.append({"product_name": pname, "quantity": qty, "unit_price": price, "product_id": pid})
            total += price * qty

    payload = {"customer_id": cust_id, "transaction_date": date,
               "notes": f"Auto-scanned. Image: {gemini_result.get('_image', '?')}. {len(items)} items."}
    return payload, items, total


def send(payload, items, api_url, token, dry_run=False):
    import requests
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json", "Content-Type": "application/json"}
    print(f"  -> POST /api/transactions")
    if dry_run:
        print(f"  (dry-run) {json.dumps(payload, ensure_ascii=False)[:200]}")
        return {"dry_run": True}

    resp = requests.post(f"{api_url}/api/transactions", json=payload, headers=headers, timeout=15)
    print(f"  <- {resp.status_code}")
    if resp.status_code not in [200, 201]:
        return {"error": resp.text[:200]}

    txn = resp.json()
    txn_id = txn.get("id") or txn.get("data", {}).get("id") or txn.get("data", {}).get("transaction", {}).get("id")
    if not txn_id:
        return {"response": txn}
    print(f"  Tx ID: {txn_id}")
    ok = 0
    for item in items:
        r = requests.post(f"{api_url}/api/transactions/{txn_id}/items", json=item, headers=headers, timeout=10)
        if r.status_code in [200, 201]:
            ok += 1
            print(f"    + {item['product_name']} x{item['quantity']} @Rp{item['unit_price']:,}")
    return {"txn_id": txn_id, "items_saved": ok, "items_total": len(items)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-url", type=str, default=None)
    args = parser.parse_args()

    api_url = args.api_url or os.environ.get("LARAVEL_API_URL", "http://127.0.0.1:8000")
    token = os.environ.get("LARAVEL_API_TOKEN", "")
    api_key = os.environ.get("GOOGLE_API_KEY")

    print(f"=== Gemini Vision -> Laravel ===")
    print(f"  API: {api_url}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE' if token else 'TOKEN MISSING'}")

    if not api_key:
        print("Set GOOGLE_API_KEY")
        return

    from scripts.run_gemini_pipeline import process_images
    images = sorted(glob.glob("D:/dataset/IMG_*.jpg"))[:args.limit]
    print("[1/2] Gemini Vision on", len(images), "images...")
    t0 = time.time()
    results = process_images(images, api_key)
    print(f"  Done in {time.time()-t0:.1f}s")

    matcher = ImprovedMatcher(DEFAULT_PRODUCTS, DEFAULT_CUSTOMERS)
    print("[2/2] Converting + sending...")
    for r in results:
        if r.get("error"):
            print(f"  SKIP {r.get('_image')}: error")
            continue
        name = r.get("_image", "unknown")
        payload, items, total = convert(r, matcher)
        print(f" --- {name} ---")
        print(f"  Date: {payload['transaction_date']}")
        print(f"  Customer ID: {payload['customer_id']}")
        print(f"  Items: {len(items)} (total: Rp{total:,})")
        res = send(payload, items, api_url, token, dry_run=args.dry_run)
        print(f"  Result: {res}")

    print(f" === DONE ===")

    sd = os.path.expanduser("~") + "/AppData/Local/hermes/state/reyon-watch"
    if os.path.exists(sd):
        st = {"status": "done", "task": f"Gemini->Laravel: {len(results)} imgs",
              "updated_at": int(time.time()),
              "note": "Integration ready. Use --dry-run to test."}
        with open(os.path.join(sd, "current.json"), "w") as f:
            json.dump(st, f, indent=2)


if __name__ == "__main__":
    main()
