"""
Samples 50,000 rows from catalog_b_normalized.jsonl, keeping only rows
whose cluster_id also appears in catalog_a_normalized.jsonl.

This ensures every sampled query has a matching anchor in catalog A.

Steps:
  1. Stream catalog_a_normalized.jsonl → collect all cluster_ids into a set
  2. Stream catalog_b_normalized.jsonl → keep rows with matching cluster_id
  3. Randomly sample 50,000 from the kept rows
  4. Write to data/catalog_b_sample.jsonl
"""

import json
import random

CATALOG_A_PATH = "data/catalog_a_normalized.jsonl"
CATALOG_B_PATH = "data/catalog_b_normalized.jsonl"
OUTPUT_PATH = "data/catalog_b_sample.jsonl"
SAMPLE_SIZE = 50_000
RANDOM_SEED = 42

# ── Step 1: collect cluster_ids from catalog A ──────────────────────
catalog_a_clusters = set()

with open(CATALOG_A_PATH, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        cluster_id = record.get("cluster_id")
        if cluster_id is not None:
            catalog_a_clusters.add(cluster_id)

        if line_num % 500_000 == 0:
            print(f"  [step 1] processed {line_num:,} rows")

print(f"Step 1 done. {len(catalog_a_clusters):,} cluster_ids from catalog A.")

# ── Step 2: collect matching rows from catalog B ────────────────────
matching_rows = []
total_b = 0
skipped = 0

with open(CATALOG_B_PATH, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        total_b += 1
        cluster_id = record.get("cluster_id")

        if cluster_id in catalog_a_clusters:
            matching_rows.append(line)
        else:
            skipped += 1

        if line_num % 500_000 == 0:
            print(f"  [step 2] processed {line_num:,} rows")

print(f"Step 2 done. {len(matching_rows):,} matching rows "
      f"(skipped {skipped:,} with no catalog A anchor).")

# ── Step 3: random sample ──────────────────────────────────────────
random.seed(RANDOM_SEED)

if len(matching_rows) <= SAMPLE_SIZE:
    sampled = matching_rows
    print(f"Step 3: only {len(matching_rows):,} matching rows, "
          f"keeping all (< {SAMPLE_SIZE:,} target).")
else:
    sampled = random.sample(matching_rows, SAMPLE_SIZE)
    print(f"Step 3 done. Sampled {len(sampled):,} from {len(matching_rows):,}.")

# ── Step 4: write output ───────────────────────────────────────────
with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
    for line in sampled:
        f_out.write(line + "\n")

print()
print("=" * 60)
print(f"TOTAL CATALOG B ROWS:    {total_b:,}")
print(f"MATCHING (has anchor):   {len(matching_rows):,}")
print(f"SKIPPED (no anchor):     {skipped:,}")
print(f"SAMPLED:                 {len(sampled):,}")
print(f"OUTPUT: {OUTPUT_PATH}")
print("=" * 60)
