"""
Splits the WDC product corpus into three files:
  - catalog_a.jsonl  : one representative offer per cluster (the "anchor")
  - catalog_b.jsonl  : the remaining offers from each cluster (to match against A)
  - singletons.jsonl : offers whose cluster has only 1 member (no match possible)

Only clusters with size >= 2 are included (singletons are dropped,
same as filter_clusters.py).

The first offer encountered for a cluster becomes the catalog entry;
all subsequent offers for that cluster become queries.

Same streaming / out-of-core pattern as filter_clusters.py:
  Pass 1 → count cluster sizes (Counter in RAM, rows stream through)
  Pass 2 → route each row to catalog_a, catalog_b, or singletons output file
"""

import gzip
import json
from collections import Counter

DATA_PATH = "data/wdcproducts_corpus_with_url.json.gz"
CATALOG_A_PATH = "data/catalog_a.jsonl"
CATALOG_B_PATH = "data/catalog_b.jsonl"
SINGLETONS_PATH = "data/singletons.jsonl"

# ── Pass 1: count cluster_id frequency ──────────────────────────────
cluster_sizes = Counter()

with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        cluster_id = record.get("cluster_id")
        if cluster_id is not None:
            cluster_sizes[cluster_id] += 1

        if line_num % 500_000 == 0:
            print(f"  [pass 1] processed {line_num:,} rows")

print(f"Pass 1 done. {len(cluster_sizes):,} unique clusters found.")

# Quick bucket summary of kept clusters (size >= 2)
kept_cluster_size_buckets = Counter()
for cid, size in cluster_sizes.items():
    if size >= 2:
        if size == 2:
            kept_cluster_size_buckets["2"] += 1
        elif size <= 5:
            kept_cluster_size_buckets["3-5"] += 1
        elif size <= 20:
            kept_cluster_size_buckets["6-20"] += 1
        else:
            kept_cluster_size_buckets["21+"] += 1

print("KEPT CLUSTERS BY SIZE BUCKET:")
for bucket, count in sorted(kept_cluster_size_buckets.items()):
    print(f"  size {bucket}: {count:,} clusters")
print()

# ── Pass 2: route rows to catalog vs queries ────────────────────────
seen_clusters = set()     # tracks which cluster_ids already have a catalog_a entry
total_rows = 0
catalog_a_rows = 0
catalog_b_rows = 0
singleton_rows = 0

with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f_in, \
     open(CATALOG_A_PATH, "w", encoding="utf-8") as f_catalog_a, \
     open(CATALOG_B_PATH, "w", encoding="utf-8") as f_catalog_b, \
     open(SINGLETONS_PATH, "w", encoding="utf-8") as f_singles:
    for line_num, line in enumerate(f_in, 1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        total_rows += 1
        cluster_id = record.get("cluster_id")

        # Singletons → singletons file (no match possible, negative test set)
        if cluster_id is None or cluster_sizes[cluster_id] < 2:
            f_singles.write(line + "\n")
            singleton_rows += 1
            continue

        # First offer for this cluster → catalog A; rest → catalog B
        if cluster_id not in seen_clusters:
            seen_clusters.add(cluster_id)
            f_catalog_a.write(line + "\n")
            catalog_a_rows += 1
        else:
            f_catalog_b.write(line + "\n")
            catalog_b_rows += 1

        if line_num % 500_000 == 0:
            print(f"  [pass 2] processed {line_num:,} rows")

print()
print("=" * 60)
print(f"TOTAL ROWS:              {total_rows:,}")
print(f"CATALOG A (anchors):     {catalog_a_rows:,}")
print(f"CATALOG B (to match):    {catalog_b_rows:,}")
print(f"SINGLETONS (no match):   {singleton_rows:,}")
print(f"  → A + B + singles:     {catalog_a_rows + catalog_b_rows + singleton_rows:,}")
print(f"OUTPUT FILES:")
print(f"  {CATALOG_A_PATH}")
print(f"  {CATALOG_B_PATH}")
print(f"  {SINGLETONS_PATH}")
print("=" * 60)
