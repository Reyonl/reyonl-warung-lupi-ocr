#!/usr/bin/env python
"""Gemini Vision OCR Pipeline - Production runner."""
import argparse, json, os, sys, time, glob
from datetime import datetime

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from google.genai import Client, types

OCR_PROMPT = """Analyze this Indonesian warung notebook receipt photo. Multiple customers/sections.

CRITICAL: Output ONLY valid JSON. No markdown, no commentary.

{
  "date": "08 OKT",
  "sections": [
    {"customer": "name", "items": [{"product": "name", "quantity": 1, "unit_price": 15000, "raw_line": "text"}]}
  ],
  "all_text": "complete readable text"
}"""


def parse_json_response(text):
    text = text.strip()
    if "```json" in text:
        text = text.replace("```json", "").replace("```", "")
    elif text.startswith("```"):
        text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
    if "```" in text:
        text = text.split("```")[0]
    return json.loads(text.strip())


def process_single_image(image_path, client, model="gemini-3.5-flash"):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    img_size_kb = len(image_bytes) / 1024

    t0 = time.time()
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=OCR_PROMPT),
        ],
        config=types.GenerateContentConfig(max_output_tokens=8192, temperature=0.1)
    )
    dt = time.time() - t0

    result = parse_json_response(resp.text)
    result["_processing_time"] = round(dt, 1)
    result["_image_size_kb"] = round(img_size_kb)
    result["_model"] = model
    result["_image"] = os.path.basename(image_path)
    return result


def deduplicate_items(sections):
    seen = set()
    for section in sections:
        unique_items = []
        for item in section.get("items", []):
            key = (item["product"].lower().strip(), item["raw_line"].lower().strip()[:50])
            if key not in seen:
                seen.add(key)
                unique_items.append(item)
        section["items"] = unique_items
    return sections


def process_images(image_paths, api_key, model="gemini-3.5-flash"):
    client = Client(api_key=api_key)
    results = []
    model_sequence = [model, "gemini-3.7-flash", "gemini-3.8-flash", "gemini-flash-latest"]

    for i, img_path in enumerate(image_paths):
        basename = os.path.basename(img_path)
        success = False

        for idx, m in enumerate(model_sequence):
            try:
                result = process_single_image(img_path, client, model=m)
                sections = result.get("sections", [])
                total_items = sum(len(s.get("items", [])) for s in sections)
                if total_items == 0:
                    raise ValueError("Zero items")

                sections = deduplicate_items(sections)
                result["sections"] = sections
                result["_total_items"] = sum(len(s["items"]) for s in sections)

                print(f"  OK {basename}: {result['_total_items']} items, {result.get('date', 'N/A')}, {result['_processing_time']}s [{m}]")
                success = True
                results.append(result)
                break
            except Exception as e:
                err_str = str(e)[:100]
                if idx < len(model_sequence) - 1:
                    print(f"  ERR {basename}: {m} - {err_str}")
                    time.sleep(0.5)

        if not success:
            print(f"  WARN {basename}: all models failed")
            results.append({"image": basename, "error": "all_failed", "_total_items": 0})
        if i < len(image_paths) - 1:
            time.sleep(0.3)
    return results


def main():
    parser = argparse.ArgumentParser(description="Gemini Vision OCR Pipeline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--single", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY env variable first!")
        sys.exit(1)

    if args.single:
        img_path = f"D:/dataset/{args.single}.jpg"
        if not os.path.exists(img_path):
            matches = glob.glob(f"D:/dataset/{args.single}.*")
            img_path = matches[0] if matches else None
        if not img_path or not os.path.exists(img_path):
            print(f"Image not found: {args.single}")
            sys.exit(1)
        images = [img_path]
    else:
        images = sorted(glob.glob("D:/dataset/IMG_*.jpg"))[args.start:]
        if args.limit:
            images = images[:args.limit]

    print(f"Gemini Vision OCR Pipeline - {len(images)} images")

    t0 = time.time()
    results = process_images(images, api_key)
    total_time = time.time() - t0

    total_items = sum(r.get("_total_items", 0) for r in results)
    success_count = sum(1 for r in results if r.get("sections"))

    print(f"\n=== SUMMARY ===")
    print(f"Images: {len(results)} | Success: {success_count}/{len(results)}")
    print(f"Total items: {total_items}")
    print(f"Time: {total_time:.1f}s ({total_time/len(results):.1f}s avg)")

    for r in results:
        basename = r.get("_image", "N/A")
        if r.get("error"):
            print(f"  ERR: {basename}")
        else:
            print(f"  OK: {basename}: {r.get('_total_items', 0)} items, {r.get('_processing_time', 0)}s")

    output_file = args.output or f"eval/results/gemini_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_file}")

    # Telegram watchdog update
    state_dir = os.path.expanduser("~") + "/AppData/Local/hermes/state/reyon-watch"
    if os.path.exists(state_dir):
        state = {
            "status": "done",
            "task": f"Gemini pipeline: {success_count}/{len(results)} OK, {total_items} items",
            "updated_at": int(time.time()),
            "note": f"LANGKAH 8 done: {success_count}/{len(results)} images, {total_items} items. Ready for Laravel integration."
        }
        with open(os.path.join(state_dir, "current.json"), "w") as f:
            json.dump(state, f, indent=2)
        print("Telegram watchdog updated")


if __name__ == "__main__":
    main()
