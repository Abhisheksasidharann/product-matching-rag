"""
Scan catalog_a.jsonl for off-domain entities (vehicles, real estate)
to find patterns for reliable detection rules.
"""

import json
import re

vehicle_pattern = re.compile(r'^\$[\d,]+\s*[·•]\s*\d{4}\s+\w', re.IGNORECASE)
address_pattern = re.compile(
    r'^\d+\s+[\w\s]+,\s*[\w\s]+,\s*[A-Z]{2}[\s\d]',
    re.IGNORECASE
)

found = 0
scanned = 0

with open('data/catalog_a.jsonl') as f:
    for line in f:
        scanned += 1
        r = json.loads(line)
        title = r.get('title', '') or ''
        if vehicle_pattern.match(title) or address_pattern.match(title):
            found += 1
            print(f"{found:3d} | {title[:120]}")
            if found >= 50:
                break

print(f"\n--- Found {found} off-domain rows in first {scanned:,} rows scanned ---")
