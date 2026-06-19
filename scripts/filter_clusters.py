"""
Filters the WDC product corpus down to offers belonging to
multi-member clusters (size >= 2), excluding singletons.
Writes the result to a new, much smaller JSONL file.
"""

import gzip
import json
from collections import Counter

DATA_PATH = "data/wdcproducts_corpus_with_url.json.gz"
OUTPUT_PATH = "data/filtered_offers.jsonl"

# Pass 1: count cluster_id frequency (same logic as inspect_corpus.py)
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

# Pass 2: keep only rows whose cluster has size >= 2
total_rows = 0
kept_rows = 0

with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f_in, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
    for line_num, line in enumerate(f_in, 1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        total_rows += 1
        cluster_id = record.get("cluster_id")

        if cluster_sizes[cluster_id] >= 2:
            f_out.write(line + "\n")
            kept_rows += 1

        if line_num % 500_000 == 0:
            print(f"  [pass 2] processed {line_num:,} rows")

print()
print("=" * 60)
print(f"TOTAL ROWS: {total_rows:,}")
print(f"KEPT ROWS (cluster size >= 2): {kept_rows:,}")
print(f"DROPPED ROWS (singletons): {total_rows - kept_rows:,}")
print("=" * 60)
