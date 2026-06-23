"""
Build FAISS index from catalog_b_sample.jsonl embeddings.

Steps:
  1. Load catalog_b_sample.jsonl, extract composite_text for encoding
     and id/cluster_id for the position mapping.
  2. Encode all texts with SentenceTransformer("all-MiniLM-L6-v2")
     in batches of 64.
  3. Normalize vectors and build a FAISS IndexFlatIP (cosine similarity).
  4. Write:
       - data/catalog_b_sample.index    (FAISS binary)
       - data/catalog_b_sample_ids.json (position → id/cluster_id mapping)
"""

import json
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_PATH = "data/catalog_b_sample.jsonl"
INDEX_PATH = "data/catalog_b_sample.index"
ID_MAP_PATH = "data/catalog_b_sample_ids.json"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

# ── Step 1: load texts and metadata ─────────────────────────────────
print("Step 1: Loading catalog_b_sample.jsonl ...")
texts = []
id_map = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        texts.append(record["composite_text"])
        id_map.append({
            "position": i,
            "id": record.get("id"),
            "cluster_id": record.get("cluster_id"),
            "title": record.get("title"),
            "composite_text": record.get("composite_text"),
            "brand": record.get("brand"),
            "price": record.get("price"),
            "priceCurrency": record.get("priceCurrency"),
        })

print(f"  Loaded {len(texts):,} rows.")

# ── Step 2: encode with SentenceTransformer ─────────────────────────
print(f"Step 2: Encoding with {MODEL_NAME} (batch_size={BATCH_SIZE}) ...")
t0 = time.time()

model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,   # L2-normalize so IP == cosine
)

elapsed = time.time() - t0
print(f"  Encoded {len(embeddings):,} vectors in {elapsed:.1f}s "
      f"({len(embeddings) / elapsed:.0f} vec/s).")

# ── Step 3: build FAISS index ───────────────────────────────────────
print("Step 3: Building FAISS IndexFlatIP ...")
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

print(f"  Index contains {index.ntotal:,} vectors (dim={dim}).")

# ── Step 4: write outputs ──────────────────────────────────────────
print("Step 4: Writing outputs ...")
faiss.write_index(index, INDEX_PATH)
print(f"  Wrote {INDEX_PATH}")

with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(id_map, f)
print(f"  Wrote {ID_MAP_PATH} ({len(id_map):,} entries)")

print()
print("=" * 60)
print(f"VECTORS:   {index.ntotal:,}")
print(f"DIMENSION: {dim}")
print(f"INDEX:     {INDEX_PATH}")
print(f"ID MAP:    {ID_MAP_PATH}")
print("=" * 60)
