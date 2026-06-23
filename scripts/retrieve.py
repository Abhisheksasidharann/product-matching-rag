"""
RAG retrieval: embed a query offer from catalog A, search the FAISS index
built from catalog B, and return the top-K candidate products with scores.

This is the core retrieval step — "Candidate products are retrieved."

Steps:
  1. Load FAISS index and ID map (built by build_embeddings.py)
  2. Load the SentenceTransformer model (same one used for indexing)
  3. Stream catalog_a_normalized.jsonl as query offers
  4. For each query: embed composite_text → search index → return top-K
  5. Print results with similarity scores and ground-truth match check
"""

import json
import time
import warnings

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ──────────────────────────────────────────────────────────
INDEX_PATH = "data/catalog_b_sample.index"
ID_MAP_PATH = "data/catalog_b_sample_ids.json"
CATALOG_A_PATH = "data/catalog_a_normalized.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 10
NUM_QUERIES = 500         # how many queries to run (set None for all)

# ── Step 1: load FAISS index ────────────────────────────────────────
print("Loading FAISS index ...")
t0 = time.time()
index = faiss.read_index(INDEX_PATH)
print(f"  Loaded {index.ntotal:,} vectors in {time.time() - t0:.2f}s")

# ── Step 2: load ID map ────────────────────────────────────────────
print("Loading ID map ...")
with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
    id_map = json.load(f)
print(f"  Loaded {len(id_map):,} entries")

# Build set of cluster_ids present in the sample index
indexed_clusters = set(entry["cluster_id"] for entry in id_map)
print(f"  {len(indexed_clusters):,} unique cluster_ids in index")

# ── Step 3: load embedding model ───────────────────────────────────
print(f"Loading model: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
print("  Model ready.")

# ── Step 4: stream queries from catalog A and retrieve ─────────────
print(f"\nRunning retrieval: top-{TOP_K} for {NUM_QUERIES or 'all'} queries")
print("=" * 70)

hits = 0           # query where correct cluster_id appears in top-K
total = 0
recall_at_k = []   # 1 if hit, 0 if miss — for computing recall@K

with open(CATALOG_A_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if NUM_QUERIES is not None and total >= NUM_QUERIES:
            break

        line = line.strip()
        if not line:
            continue
        query = json.loads(line)

        query_cluster = query.get("cluster_id")
        if query_cluster not in indexed_clusters:
            continue

        total += 1

        query_text = query["composite_text"]
        query_title = query.get("title", "")

        # Embed the query (single vector, normalized for cosine)
        q_vec = model.encode(
            [query_text],
            normalize_embeddings=True,
        ).astype(np.float32)

        # Search FAISS index
        scores, indices = index.search(q_vec, TOP_K)

        # Map positions back to offer metadata
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            entry = id_map[idx]
            results.append({
                "rank": rank + 1,
                "score": float(score),
                "id": entry.get("id"),
                "cluster_id": entry.get("cluster_id"),
                "title": entry.get("title"),
                "brand": entry.get("brand"),
                "price": entry.get("price"),
                "priceCurrency": entry.get("priceCurrency"),
            })

        # Check ground truth: does any top-K result share the query's cluster_id?
        matched = any(r["cluster_id"] == query_cluster for r in results)
        if matched:
            hits += 1
        recall_at_k.append(1 if matched else 0)

        # ── Print misses only (for diagnosis) ──────────────────────
        if not matched:
            top_title = results[0]["title"] if results else "N/A"
            top_score = results[0]["score"] if results else 0.0
            print(f"\n[MISS] cluster={query_cluster}")
            print(f"  Query: {query_title[:100]}")
            print(f"  composite: {query_text[:100]}")
            print(f"  Top result: {(top_title or '')[:80]} (sim={top_score:.4f})")

print()
print("=" * 70)
print(f"RETRIEVAL SUMMARY (top-{TOP_K})")
print("=" * 70)
print(f"  Queries run:        {total:,}")
print(f"  Hits (correct in K):{hits:,}")
print(f"  Misses:             {total - hits:,}")
print(f"  Recall@{TOP_K}:          {hits / total:.4f}" if total > 0 else "  Recall: N/A")
print("=" * 70)
