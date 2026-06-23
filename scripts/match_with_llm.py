"""
LLM-powered product matching — Stage 5 of the RAG pipeline.

Takes query offers from catalog_a, retrieves top-K candidates from the
FAISS index (catalog_b), then calls the Anthropic Claude API to decide
which candidate (if any) is a true match.

This is where retrieval becomes reasoning: the LLM sees both the query
and each candidate's full text, and must decide match vs no-match with
cited evidence.

Steps:
  1. Load FAISS index, ID map, and SentenceTransformer model
  2. Stream queries from catalog_a_normalized.jsonl
  3. For each query: embed → retrieve top-K → build LLM prompt
  4. Call Claude API for match/no-match decision + evidence
  5. Write structured results to data/llm_match_results.jsonl
  6. Evaluate against ground truth cluster_ids
"""

import json
import os
import time
import warnings

import anthropic
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ──────────────────────────────────────────────────────────
INDEX_PATH = "data/catalog_b_sample.index"
ID_MAP_PATH = "data/catalog_b_sample_ids.json"
CATALOG_A_PATH = "data/catalog_a_normalized.jsonl"
OUTPUT_PATH = "data/llm_match_results.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "claude-sonnet-4-6"
RETRIEVAL_K = 10          # how many to pull from FAISS
LLM_K = 5                 # how many to send to LLM (top LLM_K of RETRIEVAL_K)
NUM_QUERIES = 50          # start small — each query is an API call
MAX_RETRIES = 3
RETRY_DELAY = 5           # seconds between retries on rate limit

# ── LLM prompt template ────────────────────────────────────────────
SYSTEM_PROMPT = """You are a product matching expert. Your job is to determine
whether any candidate product from a catalog is the SAME product as a query
product offer.

"Same product" means: identical item that a customer would consider
interchangeable. Different colors, sizes, or pack quantities are NOT the same
product. Different sellers or different prices for the identical SKU ARE the
same product.

Respond with valid JSON only — no markdown, no commentary:
{
  "match_found": true or false,
  "matched_candidate": 1-indexed rank of the best match (null if no match),
  "confidence": "high", "medium", or "low",
  "evidence": "one sentence explaining WHY this is or isn't a match, citing
               specific attributes (brand, model number, specs) that confirm
               or rule out the match"
}"""


def build_user_prompt(query: dict, candidates: list[dict]) -> str:
    """Build the user prompt comparing query against candidates."""
    parts = ["## Query Product\n"]

    # Query details
    parts.append(f"**Title:** {query.get('title') or 'N/A'}")
    parts.append(f"**Brand:** {query.get('brand') or 'N/A'}")
    if query.get("price") is not None:
        currency = query.get("priceCurrency") or ""
        parts.append(f"**Price:** {query['price']} {currency}".rstrip())
    parts.append(f"**Full text:** {query.get('composite_text', '')[:300]}")
    parts.append("")

    parts.append("## Candidate Products\n")
    for i, c in enumerate(candidates, 1):
        parts.append(f"### Candidate {i} (similarity={c['score']:.4f})")
        parts.append(f"**Title:** {c.get('title') or 'N/A'}")
        parts.append(f"**Brand:** {c.get('brand') or 'N/A'}")
        if c.get("price") is not None:
            currency = c.get("priceCurrency") or ""
            parts.append(f"**Price:** {c['price']} {currency}".rstrip())
        parts.append(f"**Full text:** {(c.get('composite_text') or '')[:300]}")
        parts.append("")

    parts.append("Which candidate, if any, is the SAME product as the query?")
    return "\n".join(parts)


def call_llm(client: anthropic.Anthropic, query: dict,
             candidates: list[dict]) -> dict:
    """Call Claude API and parse structured match response."""
    user_prompt = build_user_prompt(query, candidates)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=LLM_MODEL,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            # Parse JSON response
            result = json.loads(text)
            result["raw_response"] = text
            result["tokens_in"] = response.usage.input_tokens
            result["tokens_out"] = response.usage.output_tokens
            return result

        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  Rate limited, retrying in {wait}s ...")
                time.sleep(wait)
            else:
                return {"match_found": False, "error": "rate_limit_exhausted"}

        except json.JSONDecodeError:
            return {
                "match_found": False,
                "error": "json_parse_failed",
                "raw_response": text,
            }

        except anthropic.APIError as e:
            return {"match_found": False, "error": f"api_error: {e}"}

    return {"match_found": False, "error": "max_retries_exhausted"}


# ── Step 1: load FAISS index + ID map + model ──────────────────────
print("Loading FAISS index ...")
index = faiss.read_index(INDEX_PATH)
print(f"  {index.ntotal:,} vectors")

print("Loading ID map ...")
with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
    id_map = json.load(f)
indexed_clusters = set(entry["cluster_id"] for entry in id_map)
print(f"  {len(id_map):,} entries, {len(indexed_clusters):,} unique clusters")

print(f"Loading embedding model: {MODEL_NAME} ...")
embed_model = SentenceTransformer(MODEL_NAME)

print(f"Initializing Anthropic client (model: {LLM_MODEL}) ...")
client = anthropic.Anthropic()   # uses ANTHROPIC_API_KEY env var

# ── Step 2: run queries ────────────────────────────────────────────
print(f"\nRunning LLM matching: {NUM_QUERIES} queries, retrieve {RETRIEVAL_K} → send {LLM_K} to LLM")
print("=" * 70)

total = 0
tp = 0    # true positive: LLM says match, ground truth agrees
fp = 0    # false positive: LLM says match, ground truth disagrees
fn = 0    # false negative: LLM says no match, but ground truth had a match in candidates
tn = 0    # true negative: LLM says no match, ground truth has no match in candidates
errors = 0
total_tokens_in = 0
total_tokens_out = 0
t_start = time.time()

with open(CATALOG_A_PATH, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
    for line in f_in:
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

        # ── Retrieve top-K candidates ──────────────────────────────
        q_vec = embed_model.encode(
            [query["composite_text"]],
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = index.search(q_vec, RETRIEVAL_K)

        candidates = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            entry = id_map[idx]
            candidates.append({
                "rank": rank + 1,
                "score": float(score),
                "id": entry.get("id"),
                "cluster_id": entry.get("cluster_id"),
                "title": entry.get("title"),
                "brand": entry.get("brand"),
                "price": entry.get("price"),
                "priceCurrency": entry.get("priceCurrency"),
                "composite_text": entry.get("composite_text"),
            })

        # ── Call LLM (send only top LLM_K candidates) ─────────────
        llm_candidates = candidates[:LLM_K]
        llm_result = call_llm(client, query, llm_candidates)

        # ── Evaluate against ground truth ──────────────────────────
        gt_has_match = any(c["cluster_id"] == query_cluster for c in candidates)
        llm_says_match = llm_result.get("match_found", False)

        # Check if LLM picked the RIGHT candidate
        llm_correct = False
        matched_rank = llm_result.get("matched_candidate")
        if llm_says_match and matched_rank is not None:
            picked = candidates[matched_rank - 1] if 1 <= matched_rank <= len(candidates) else None
            if picked and picked["cluster_id"] == query_cluster:
                llm_correct = True

        if "error" in llm_result:
            errors += 1
        elif llm_says_match and llm_correct:
            tp += 1
        elif llm_says_match and not llm_correct:
            fp += 1
        elif not llm_says_match and gt_has_match:
            fn += 1
        else:
            tn += 1

        total_tokens_in += llm_result.get("tokens_in", 0)
        total_tokens_out += llm_result.get("tokens_out", 0)

        # ── Write result row ───────────────────────────────────────
        result_row = {
            "query_id": query.get("id"),
            "query_cluster_id": query_cluster,
            "query_title": query.get("title"),
            "gt_has_match_in_candidates": gt_has_match,
            "llm_match_found": llm_says_match,
            "llm_matched_candidate_rank": matched_rank,
            "llm_correct": llm_correct if llm_says_match else (not gt_has_match),
            "llm_confidence": llm_result.get("confidence"),
            "llm_evidence": llm_result.get("evidence"),
            "candidates_count": len(candidates),
            "top_candidate_title": candidates[0]["title"] if candidates else None,
            "top_candidate_score": candidates[0]["score"] if candidates else None,
        }
        if "error" in llm_result:
            result_row["error"] = llm_result["error"]

        f_out.write(json.dumps(result_row) + "\n")

        # ── Progress ───────────────────────────────────────────────
        status = "✓" if result_row["llm_correct"] else "✗"
        conf = llm_result.get("confidence", "?")
        print(f"  [{total:>3}] {status} conf={conf:<6} "
              f"{(query.get('title') or '')[:50]}")

elapsed = time.time() - t_start
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
accuracy = (tp + tn) / total if total > 0 else 0.0

print()
print("=" * 70)
print("LLM MATCHING SUMMARY")
print("=" * 70)
print(f"  Queries evaluated:    {total:,}")
print(f"  True positives (TP):  {tp:,}")
print(f"  False positives (FP): {fp:,}")
print(f"  False negatives (FN): {fn:,}")
print(f"  True negatives (TN):  {tn:,}")
print(f"  Errors:               {errors:,}")
print(f"  ──────────────────────────────")
print(f"  Precision:            {precision:.4f}")
print(f"  Recall:               {recall:.4f}")
print(f"  F1 Score:             {f1:.4f}")
print(f"  Accuracy:             {accuracy:.4f}")
print(f"  ──────────────────────────────")
print(f"  Total tokens in:      {total_tokens_in:,}")
print(f"  Total tokens out:     {total_tokens_out:,}")
print(f"  Time elapsed:         {elapsed:.1f}s ({total / elapsed:.1f} queries/s)")
print(f"  Output:               {OUTPUT_PATH}")
print("=" * 70)
