"""
Step 2: Convert clean conversation history into MSA documents.
Each document = one conversation turn (user message + assistant response).
Autonomous/unprompted assistant messages become standalone documents.
"""
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATASET_FILE = PROJECT_DIR / "dataset" / "conversation_complete.json"
OUTPUT_FILE = PROJECT_DIR / "dataset" / "documents.json"


def build_documents():
    print("Loading conversation dataset...")
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    messages = data['conversation']
    print(f"Total messages: {len(messages)}")

    documents = []
    doc_id = 0
    i = 0

    while i < len(messages):
        msg = messages[i]

        if msg['role'] == 'user':
            # Standard turn: user + assistant pair
            user_content = msg['content']
            user_ts = msg.get('timestamp', '')

            if i + 1 < len(messages) and messages[i + 1]['role'] == 'assistant':
                assistant_content = messages[i + 1]['content']
                assistant_ts = messages[i + 1].get('timestamp', '')

                doc = {
                    'doc_id': doc_id,
                    'text': f"User: {user_content}\nAssistant: {assistant_content}",
                    'timestamp': user_ts or assistant_ts,
                    'type': 'conversation',
                    'char_count': len(user_content) + len(assistant_content),
                }
                documents.append(doc)
                doc_id += 1
                i += 2
            else:
                # Orphan user message (no response)
                doc = {
                    'doc_id': doc_id,
                    'text': f"User: {user_content}",
                    'timestamp': user_ts,
                    'type': 'orphan_user',
                    'char_count': len(user_content),
                }
                documents.append(doc)
                doc_id += 1
                i += 1

        elif msg['role'] == 'assistant':
            # Standalone assistant message (autonomous/unprompted)
            assistant_content = msg['content']
            assistant_ts = msg.get('timestamp', '')
            is_autonomous = msg.get('autonomous', False)

            doc = {
                'doc_id': doc_id,
                'text': f"Assistant (unprompted): {assistant_content}",
                'timestamp': assistant_ts,
                'type': 'autonomous' if is_autonomous else 'unprompted',
                'char_count': len(assistant_content),
            }
            documents.append(doc)
            doc_id += 1
            i += 1
        else:
            i += 1

    # Stats
    total_chars = sum(d['char_count'] for d in documents)
    avg_chars = total_chars / len(documents) if documents else 0
    types = {}
    for d in documents:
        t = d['type']
        types[t] = types.get(t, 0) + 1

    timestamps = [d['timestamp'][:10] for d in documents if d.get('timestamp')]
    earliest = min(timestamps) if timestamps else 'unknown'
    latest = max(timestamps) if timestamps else 'unknown'

    output = {
        'metadata': {
            'total_documents': len(documents),
            'total_characters': total_chars,
            'estimated_tokens': total_chars // 4,
            'avg_chars_per_doc': int(avg_chars),
            'avg_tokens_per_doc': int(avg_chars // 4),
            'date_range': f"{earliest} to {latest}",
            'document_types': types,
        },
        'documents': documents,
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"DOCUMENTS PREPARED")
    print(f"{'='*60}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total documents: {len(documents):,}")
    print(f"Document types: {types}")
    print(f"Estimated tokens: {total_chars // 4:,}")
    print(f"Avg tokens/doc: {int(avg_chars // 4)}")
    print(f"Date range: {earliest} to {latest}")


if __name__ == '__main__':
    build_documents()
