# Product Matching RAG Pipeline

## 1. Problem Statement
E-commerce platforms aggregate millions of products from disparate sellers, resulting in duplicate listings for the identical physical product under vastly different titles, descriptions, and metadata. Identifying and linking identical products across different sources is a notoriously difficult problem due to missing global identifiers (like UPCs), varied naming conventions, and subtle variant differences (like size or color). This project solves that problem by building a scalable Retrieval-Augmented Generation (RAG) pipeline to semantically embed, retrieve, and accurately match products using a high-precision Large Language Model.

## 2. Why this Dataset?
The Web Data Commons (WDC) Product Corpus provides millions of real-world product offers extracted from e-commerce websites via schema.org microdata. It perfectly mirrors the messiness of real-world aggregators: it is incredibly noisy, featuring sparse attributes, missing brands, mismatched pricing currencies, and out-of-domain data (like real estate or vehicles). This dataset tests the limits of normalization, embedding robustness, and LLM reasoning.

## 3. Architecture

```text
[WDC Corpus (1.3M+ Offers)]
       │
       ├─ filter_clusters.py (Removes singletons)
       │
       ├─ build_catalog_split.py
       │    ├── catalog_a.jsonl (Anchors)
       │    └── catalog_b.jsonl (Candidates)
       │
       ├─ normalize_offers.py (Off-domain rejection, composite text)
       │    ├── catalog_a_normalized.jsonl
       │    └── catalog_b_normalized.jsonl
       │
       ├─ sample_catalog.py (50k sample for evaluation)
       │
       ├─ build_embeddings.py
       │    ├── FAISS IndexFlatIP (Vectors)
       │    └── ID Map JSON (Rich Metadata)
       │
       ├─ retrieve.py (Top-K FAISS Search)
       │
       └─ match_with_llm.py (claude-sonnet-4-6 Match & Eval)
            └── llm_match_results.jsonl (TP/FP/FN/TN metrics)
```

## 4. Pipeline Stages & Key Decisions
1. **Data Splitting (`build_catalog_split.py`)**: Streams the corpus out-of-core, splitting into `catalog_a` (anchor queries) and `catalog_b` (candidates). Singletons (clusters of size 1) are aggressively dropped to prevent indexing unmatchable noise.
2. **Normalization (`normalize_offers.py`)**: Cleans prices, rejects out-of-domain junk via regex, and builds a rich `composite_text` field to maximize embedding signal.
3. **Sampling (`sample_catalog.py`)**: Extracts a 50k candidate sample ensuring ground-truth clusters remain intact for rapid iteration.
4. **Embedding & Indexing (`build_embeddings.py`)**: Uses `all-MiniLM-L6-v2` with batch processing (10-50x faster). Vectors are L2-normalized so `IndexFlatIP` performs Cosine Similarity. A rich ID map (~50MB) is constructed alongside the index so the downstream LLM has full product metadata without re-opening JSONL files.
5. **Retrieval (`retrieve.py`)**: Embeds incoming queries and fetches Top-K candidates. Non-matchable queries are skipped to ensure evaluation metrics reflect true retrieval capability.
6. **LLM Matching (`match_with_llm.py`)**: The top 5 candidates are sent to Claude. The LLM evaluates the candidates against the query, enforcing strict definitions of "same product" (ignoring variants, but accepting different sellers), returning a structured JSON response with citations.

## 5. Results
Based on a 50-query end-to-end evaluation harness:

*Note: Evaluated on 50 queries due to API rate limits. Retrieval-only evaluation (retrieve.py) covers 500 queries with Recall@10 = 0.896.*

* **Precision: 1.0 (100%)** — Zero false positives.
* **Recall: 0.87 (87%)**
* **F1 Score: 0.93**

**What these numbers mean:** The LLM reasoning step acts as a perfect high-precision filter. In production, 0 False Positives is vastly more valuable than squeezing out a few extra points of recall at the cost of bad matches. The pipeline retrieves effectively and filters defensively.

## 6. Key Engineering Decisions
* **Singleton Filtering:** Stripping clusters with only 1 member saves massive amounts of compute and storage by ignoring items that can never technically be matched.
* **Year-Pattern False Positive Rejection:** Early EDA showed vehicle and real estate listings heavily skewed results. Regex filters were implemented to cleanly reject them before embedding.
* **Composite Text Construction:** Dense embeddings require high-quality textual signal. Brands were stripped from titles (to avoid duplication), and `Brand + Clean Title + Truncated Description` were merged into a single composite string to give the SentenceTransformer maximum context.
* **Why LLM Reasoning Catches What Embeddings Miss:** Dense vector embeddings treat an "8GB RAM module" and a "64GB RAM module" as highly similar due to immense semantic overlap. FAISS will confidently retrieve both. The LLM acts as the ultimate reasoning engine to spot variant differences that embeddings merge together, effectively rescuing precision.

## 7. Failure Modes & Limitations
* **Variant Caution (False Negatives):** When candidates are extremely close variants (e.g., iPhone 6 vs 5s battery, wrong pack size), the LLM correctly rejects them. If metadata is sketchy or conflicting across listings for the actual exact same product, the LLM will play it safe and reject, hurting recall but preserving precision.
* **Sparse / Generic Titles:** Products titled simply "USB Cable" or "Screen Protector" with no brand or description have virtually zero signal. These are unrecoverable at the retrieval stage without richer source data.
* **Sampling Artifacts:** Evaluated over a 50K sample, some queries simply missed because their true match was in the remaining 1.3M corpus. This resolves when scaling up to index the full catalog.

## 8. How to Run
**Prerequisites:**
- Python 3.10+ and a virtual environment activated.
- An Anthropic API key (`export ANTHROPIC_API_KEY="your_anthropic_key"`) must be set in your environment before running the LLM matching step.

```bash
# 0. Download the WDC Products Corpus
# Download from: http://webdatacommons.org/largescaleproductcorpus/
# Place the file at: data/wdcproducts_corpus_with_url.json.gz

# 1. Install dependencies
pip install sentence-transformers faiss-cpu anthropic numpy

# 2. Extract and split the corpus
python scripts/filter_clusters.py
python scripts/build_catalog_split.py

# 3. Normalize the data
python scripts/normalize_offers.py

# 4. Generate the 50k sample and build FAISS index
python scripts/sample_catalog.py
python scripts/build_embeddings.py

# 5. Evaluate Retrieval (Optional)
python scripts/retrieve.py

# 6. Run End-to-End LLM Matching
python scripts/match_with_llm.py
```
