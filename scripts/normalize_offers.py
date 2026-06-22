"""
Normalizes catalog_a.jsonl and catalog_b.jsonl:
  - Detects and rejects off-domain entities (vehicles, real estate)
  - Cleans price field (strip $, commas → float or None)
  - Removes brand-in-title duplication
  - Constructs a composite text field (brand + title + truncated description)
  - Writes normalized output to catalog_a_normalized.jsonl / catalog_b_normalized.jsonl
  - Routes rejected rows to rejected.jsonl

Same streaming / out-of-core pattern as the rest of the pipeline.
"""

import json
import re

# ── Paths ───────────────────────────────────────────────────────────
INPUT_FILES = {
    "a": "data/catalog_a.jsonl",
    "b": "data/catalog_b.jsonl",
}
OUTPUT_FILES = {
    "a": "data/catalog_a_normalized.jsonl",
    "b": "data/catalog_b_normalized.jsonl",
}
REJECTED_PATH = "data/rejected.jsonl"

# ── Off-domain detection patterns (from scan_offdomain.py) ──────────
VEHICLE_PATTERN = re.compile(r'^\$[\d,]+\s*[·•]\s*\d{4}\s+\w', re.IGNORECASE)
ADDRESS_PATTERN = re.compile(
    r'^\d+\s+[\w\s]+,\s*[\w\s]+,\s*[A-Z]{2}[\s\d]',
    re.IGNORECASE
)


def is_off_domain(title: str) -> str | None:
    """Returns the rejection reason if off-domain, else None."""
    if VEHICLE_PATTERN.match(title):
        return "vehicle_listing"
    if ADDRESS_PATTERN.match(title):
        return "real_estate_listing"
    return None


def clean_price(raw_price) -> float | None:
    """Strip $, commas, cast to float. Returns None if unparseable."""
    if raw_price is None or raw_price == "":
        return None
    if isinstance(raw_price, (int, float)):
        return float(raw_price)
    # String: strip currency symbols and commas
    cleaned = str(raw_price).strip().lstrip("$").replace(",", "")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def remove_brand_from_title(brand: str, title: str) -> str:
    """If title starts with the brand name, strip it to avoid duplication."""
    if not brand or not title:
        return title
    # Case-insensitive prefix check
    if title.lower().startswith(brand.lower()):
        stripped = title[len(brand):].lstrip(" -–—:|/")
        return stripped if stripped else title
    return title


DESC_MAX_CHARS = 200

def build_composite_text(brand: str, title: str, description: str) -> str:
    """
    Constructs: brand + title + truncated description.
    Title already has brand prefix removed, so no duplication.
    """
    parts = []
    if brand:
        parts.append(brand)
    if title:
        parts.append(title)
    if description:
        desc_truncated = description[:DESC_MAX_CHARS]
        if len(description) > DESC_MAX_CHARS:
            desc_truncated += "..."
        parts.append(desc_truncated)
    return " | ".join(parts)


def normalize_record(record: dict) -> dict:
    """Apply all normalization steps to a single record."""
    brand = (record.get("brand") or "").strip()
    title = (record.get("title") or "").strip()
    description = (record.get("description") or "").strip()

    # Clean the title of brand duplication
    clean_title = remove_brand_from_title(brand, title)

    # Build composite text for embedding
    composite = build_composite_text(brand, clean_title, description)

    # Clean price
    price = clean_price(record.get("price"))

    # Guard: no usable text → reject rather than pollute embedding index
    if not composite.strip():
        return None

    normalized = {
        "id": record.get("id"),
        "cluster_id": record.get("cluster_id"),
        "brand": brand or None,
        "title": clean_title,
        "title_original": title if clean_title != title else None,
        "description": description or None,
        "price": price,
        "priceCurrency": record.get("priceCurrency") or None,
        "url": record.get("url") or None,
        "composite_text": composite,
    }
    return normalized


# ── Main processing loop ────────────────────────────────────────────
stats = {
    "a": {"total": 0, "rejected": 0, "normalized": 0},
    "b": {"total": 0, "rejected": 0, "normalized": 0},
}

with open(REJECTED_PATH, "w", encoding="utf-8") as f_rejected:
    for catalog_key in ("a", "b"):
        input_path = INPUT_FILES[catalog_key]
        output_path = OUTPUT_FILES[catalog_key]
        s = stats[catalog_key]

        print(f"Processing catalog_{catalog_key}: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f_in, \
             open(output_path, "w", encoding="utf-8") as f_out:
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                s["total"] += 1

                title = (record.get("title") or "").strip()

                # Off-domain check
                rejection_reason = is_off_domain(title)
                if rejection_reason:
                    record["_rejection_reason"] = rejection_reason
                    record["_source_catalog"] = catalog_key
                    f_rejected.write(json.dumps(record) + "\n")
                    s["rejected"] += 1
                    continue

                # Normalize and write
                normalized = normalize_record(record)
                if normalized is None:
                    record["_rejection_reason"] = "no_composite_text"
                    record["_source_catalog"] = catalog_key
                    f_rejected.write(json.dumps(record) + "\n")
                    s["rejected"] += 1
                    continue

                f_out.write(json.dumps(normalized) + "\n")
                s["normalized"] += 1

                if line_num % 500_000 == 0:
                    print(f"  ...processed {line_num:,} rows")

        print(f"  Done: {s['total']:,} total, "
              f"{s['rejected']:,} rejected, "
              f"{s['normalized']:,} normalized")
        print()

# ── Summary ─────────────────────────────────────────────────────────
total_all = sum(s["total"] for s in stats.values())
rejected_all = sum(s["rejected"] for s in stats.values())
normalized_all = sum(s["normalized"] for s in stats.values())

print("=" * 60)
print("NORMALIZATION SUMMARY")
print("=" * 60)
print(f"  Catalog A:  {stats['a']['normalized']:>10,} normalized  "
      f"{stats['a']['rejected']:>8,} rejected  "
      f"(of {stats['a']['total']:,})")
print(f"  Catalog B:  {stats['b']['normalized']:>10,} normalized  "
      f"{stats['b']['rejected']:>8,} rejected  "
      f"(of {stats['b']['total']:,})")
print(f"  {'─' * 56}")
print(f"  TOTAL:      {normalized_all:>10,} normalized  "
      f"{rejected_all:>8,} rejected  "
      f"(of {total_all:,})")
print()
print("OUTPUT FILES:")
print(f"  {OUTPUT_FILES['a']}")
print(f"  {OUTPUT_FILES['b']}")
print(f"  {REJECTED_PATH}")
print("=" * 60)
