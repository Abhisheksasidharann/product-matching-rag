import json

print("=== REJECTED.JSONL (first 3 rows) ===")
with open('data/rejected.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 3: break
        r = json.loads(line)
        print(r.get('_rejection_reason'), '|', r.get('title', '')[:60])

print()
print("=== CATALOG_A_NORMALIZED.JSONL (first 3 rows) ===")
with open('data/catalog_a_normalized.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 3: break
        r = json.loads(line)
        print('composite_text:', r.get('composite_text', '')[:100])
        print('price:', r.get('price'))
        print('---')
