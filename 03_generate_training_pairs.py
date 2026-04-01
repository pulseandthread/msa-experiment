"""
Step 3: Generate contrastive training pairs for MSA router.

Each pair: a query + positive documents + negative documents.

Positive strategies:
1. Temporal neighbors (±2 turns from query position)
2. Entity overlap (documents mentioning the same specific entities)

Negative strategy:
- Random documents from distant time periods (>50 docs away)
- ~1:7 positive:negative ratio

NOTE: This is a simplified approach. The MSA paper uses the model's own
attention patterns as the training signal, which is richer but requires
significantly more compute. Our heuristic-based approach is cheaper but
noisier — temporal proximity is a weak proxy for semantic relevance.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(__file__).parent
DOCS_FILE = PROJECT_DIR / "dataset" / "documents.json"
OUTPUT_FILE = PROJECT_DIR / "dataset" / "training_pairs.jsonl"

# === ADAPT THIS ===
# Key entities to track for overlap-based positives.
# Add names, topics, features, etc. that appear in your conversations.
# Remove generic terms that appear in nearly every document.
ENTITIES = [
    # People (adapt to your data)
    'user_name', 'assistant_name',
    # Topics/features (adapt to your data)
    'memory', 'calendar', 'voice', 'image',
]

# Entities that appear too frequently to be useful as positive signals
COMMON_ENTITIES = {'user_name', 'assistant_name'}

random.seed(42)


def extract_entities(text):
    """Find which entities are mentioned in a document"""
    text_lower = text.lower()
    return {e for e in ENTITIES if e in text_lower}


def build_entity_index(documents):
    """Build inverted index: entity -> list of doc_ids"""
    index = defaultdict(list)
    doc_entities = {}
    for doc in documents:
        entities = extract_entities(doc['text'])
        doc_entities[doc['doc_id']] = entities
        for e in entities:
            index[e].append(doc['doc_id'])
    return index, doc_entities


def generate_pairs(documents, entity_index, doc_entities):
    """Generate contrastive training pairs"""
    pairs = []
    num_docs = len(documents)

    for idx, doc in enumerate(documents):
        doc_id = doc['doc_id']

        # === POSITIVES ===
        positives = set()

        # Strategy 1: Temporal neighbors (±2 positions)
        for offset in [-2, -1, 1, 2]:
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < num_docs:
                positives.add(documents[neighbor_idx]['doc_id'])

        # Strategy 2: Entity overlap (shared meaningful entities)
        my_entities = doc_entities.get(doc_id, set())
        if my_entities:
            for other_doc in documents:
                if other_doc['doc_id'] == doc_id:
                    continue
                other_entities = doc_entities.get(other_doc['doc_id'], set())
                meaningful_overlap = (my_entities & other_entities) - COMMON_ENTITIES
                if len(meaningful_overlap) >= 1:
                    positives.add(other_doc['doc_id'])
                    if len(positives) >= 8:
                        break

        positives.discard(doc_id)

        if not positives:
            if idx > 0:
                positives.add(documents[idx - 1]['doc_id'])
            if idx < num_docs - 1:
                positives.add(documents[idx + 1]['doc_id'])

        positives = list(positives)[:8]

        # === NEGATIVES ===
        num_negatives = min(len(positives) * 7, 56)
        negative_candidates = [
            documents[other_idx]['doc_id']
            for other_idx in range(num_docs)
            if abs(other_idx - idx) > 50
            and documents[other_idx]['doc_id'] not in positives
            and documents[other_idx]['doc_id'] != doc_id
        ]

        if len(negative_candidates) < num_negatives:
            negative_candidates.extend([
                documents[other_idx]['doc_id']
                for other_idx in range(num_docs)
                if abs(other_idx - idx) > 10
                and documents[other_idx]['doc_id'] not in positives
                and documents[other_idx]['doc_id'] != doc_id
                and documents[other_idx]['doc_id'] not in negative_candidates
            ])

        negatives = random.sample(negative_candidates, min(num_negatives, len(negative_candidates)))

        # === QUERY ===
        text = doc['text']
        if text.startswith('User:'):
            parts = text.split('\nAssistant:', 1)
            query = parts[0].replace('User: ', '', 1).strip()
        elif text.startswith('Assistant (unprompted):'):
            query = text.replace('Assistant (unprompted): ', '', 1).strip()
        else:
            query = text[:500]

        if len(query) < 10:
            continue

        pairs.append({
            'query': query[:1000],
            'query_doc_id': doc_id,
            'positive_doc_ids': positives,
            'negative_doc_ids': negatives,
            'timestamp': doc.get('timestamp', ''),
        })

    return pairs


def main():
    print("Loading documents...")
    with open(DOCS_FILE, 'r', encoding='utf-8') as f:
        documents = json.load(f)['documents']
    print(f"Total documents: {len(documents)}")

    print("Building entity index...")
    entity_index, doc_entities = build_entity_index(documents)
    for entity, doc_ids in sorted(entity_index.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {entity}: {len(doc_ids)} documents")

    print("\nGenerating training pairs...")
    pairs = generate_pairs(documents, entity_index, doc_entities)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    avg_pos = sum(len(p['positive_doc_ids']) for p in pairs) / len(pairs) if pairs else 0
    avg_neg = sum(len(p['negative_doc_ids']) for p in pairs) / len(pairs) if pairs else 0

    print(f"\n{'='*60}")
    print(f"TRAINING PAIRS GENERATED")
    print(f"{'='*60}")
    print(f"Total pairs: {len(pairs):,}")
    print(f"Avg positives per query: {avg_pos:.1f}")
    print(f"Avg negatives per query: {avg_neg:.1f}")


if __name__ == '__main__':
    main()
