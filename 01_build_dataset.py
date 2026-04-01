"""
Step 1: Build clean MSA training dataset from conversation history.
- Strips thinking blocks and memory injections
- Deduplicates by content hash
- Merges consecutive assistant messages
- Outputs clean chronological conversation

Adapt the source paths and entity names to your own setup.
"""
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime

# === ADAPT THESE PATHS ===
PROJECT_DIR = Path(__file__).parent
CONVERSATION_DIR = PROJECT_DIR / "conversations"      # Your conversation JSON files
BACKUP_DIR = PROJECT_DIR / "conversation_backups"      # Optional: daily backups
ARCHIVE_DIR = CONVERSATION_DIR / "archives"            # Optional: archived conversations
OUTPUT_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_messages(filepath):
    """Load messages from a conversation JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        messages = data.get('conversation', [])
        if not messages and isinstance(data, list):
            messages = data
        return messages
    except Exception as e:
        print(f"  SKIP {filepath.name}: {e}")
        return []


def normalize_for_hash(text):
    """Strip formatting characters so pre/post cleanup versions match"""
    # Remove markdown formatting
    text = text.replace('**', '').replace('*', '')
    text = text.replace('"', '').replace('\u201c', '').replace('\u201d', '')
    text = text.replace('(', '').replace(')', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def message_hash(msg):
    """Create a unique hash for a message based on role + normalized content"""
    role = msg.get('role', '')
    content = msg.get('content', '')
    if not content:
        return None
    normalized = normalize_for_hash(content)
    return hashlib.md5(f"{role}:{normalized}".encode()).hexdigest()


def clean_message(msg):
    """Strip thinking blocks, memory injections, and unnecessary fields"""
    role = msg.get('role', '')
    content = msg.get('content', '')
    timestamp = msg.get('timestamp', '')

    if not content or not role:
        return None

    # Strip injected system content from user messages
    if role == 'user':
        # Remove memory injection blocks (adapt pattern to your format)
        content = re.sub(r'<active_memories>.*?</active_memories>', '', content, flags=re.DOTALL)
        content = re.sub(r'Relevant memories:\s*\n(?:\[.*?\].*?\n)*', '', content, flags=re.DOTALL)
        content = re.sub(r'<daily_thread>.*?</daily_thread>', '', content, flags=re.DOTALL)
        # Remove timestamp headers
        content = re.sub(r'^\[(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),.*?\]\s*\n*', '', content)
        content = content.strip()

    if not content:
        return None

    cleaned = {
        'role': role,
        'content': content,
        'timestamp': timestamp,
    }

    # Mark autonomous messages (e.g., heartbeat/pulse messages)
    if msg.get('pulse_message'):
        cleaned['autonomous'] = True

    return cleaned


def build_dataset():
    """Collect, deduplicate, and clean all conversation messages"""
    seen_hashes = set()
    all_messages = []

    print("=" * 60)
    print("BUILDING MSA DATASET")
    print("=" * 60)

    # Collect source files — adapt to your file structure
    sources = []

    # Daily backups (earliest data)
    if BACKUP_DIR.exists():
        sources.extend(sorted(BACKUP_DIR.glob("*_202*.json")))

    # Archives
    if ARCHIVE_DIR.exists():
        for f in sorted(ARCHIVE_DIR.glob("*.json")):
            # Skip pre-cleanup duplicate versions
            if any(x in f.name for x in ['pre_bold', 'pre_paren', 'pre_quote', 'pre_dedup', 'corrupted']):
                print(f"  SKIP (pre-cleanup): {f.name}")
                continue
            sources.append(f)

    # Current active conversation (most recent)
    for f in CONVERSATION_DIR.glob("*.json"):
        if f.name != "archives":
            sources.append(f)

    print(f"\nScanning {len(sources)} source files...")

    for filepath in sources:
        msgs = load_messages(filepath)
        new = 0
        for msg in msgs:
            h = message_hash(msg)
            if h and h not in seen_hashes:
                seen_hashes.add(h)
                cleaned = clean_message(msg)
                if cleaned:
                    all_messages.append(cleaned)
                    new += 1
        if new > 0:
            print(f"  {filepath.name}: +{new} unique messages")

    # Sort chronologically
    print("\nSorting chronologically...")
    all_messages.sort(key=lambda m: m.get('timestamp', ''))

    # Merge consecutive assistant messages (multi-part responses)
    print("Merging consecutive assistant messages...")
    merged_messages = []
    i = 0
    merge_count = 0
    while i < len(all_messages):
        msg = all_messages[i]
        if msg['role'] == 'assistant':
            parts = [msg['content']]
            timestamp = msg.get('timestamp', '')
            autonomous = msg.get('autonomous', False)

            # Autonomous messages stay standalone
            if autonomous:
                merged_messages.append(msg)
                i += 1
                continue

            j = i + 1
            while j < len(all_messages) and all_messages[j]['role'] == 'assistant':
                if all_messages[j].get('autonomous'):
                    break
                # Don't merge if >15 min gap
                next_ts = all_messages[j].get('timestamp', '')
                if timestamp and next_ts:
                    try:
                        t1 = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        t2 = datetime.fromisoformat(next_ts.replace('Z', '+00:00'))
                        if abs((t2 - t1).total_seconds()) / 60 > 15:
                            break
                    except (ValueError, TypeError):
                        pass
                parts.append(all_messages[j]['content'])
                j += 1
            if len(parts) > 1:
                merge_count += 1
            merged_messages.append({
                'role': 'assistant',
                'content': '\n\n'.join(parts),
                'timestamp': timestamp,
            })
            i = j
        else:
            merged_messages.append(msg)
            i += 1

    print(f"  Merged {merge_count} multi-part responses")
    all_messages = merged_messages

    # Remove first of consecutive user messages (typo corrections)
    print("Removing duplicate user sends...")
    cleaned_messages = []
    removed = 0
    for i, msg in enumerate(all_messages):
        if msg['role'] == 'user' and i + 1 < len(all_messages) and all_messages[i + 1]['role'] == 'user':
            removed += 1
            continue
        cleaned_messages.append(msg)
    print(f"  Removed {removed} correction sends")
    all_messages = cleaned_messages

    # Stats
    total_chars = sum(len(m['content']) for m in all_messages)
    user_msgs = sum(1 for m in all_messages if m['role'] == 'user')
    assistant_msgs = sum(1 for m in all_messages if m['role'] == 'assistant')
    dates = [m['timestamp'][:10] for m in all_messages if m.get('timestamp')]
    earliest = min(dates) if dates else "unknown"
    latest = max(dates) if dates else "unknown"

    # Save
    output_file = OUTPUT_DIR / "conversation_complete.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'description': 'Complete conversation history for MSA training',
                'created': datetime.now().isoformat(),
                'date_range': f"{earliest} to {latest}",
                'total_messages': len(all_messages),
                'user_messages': user_msgs,
                'assistant_messages': assistant_msgs,
                'total_characters': total_chars,
                'estimated_tokens': total_chars // 4,
            },
            'conversation': all_messages,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"DATASET BUILT")
    print(f"{'=' * 60}")
    print(f"Output: {output_file}")
    print(f"Total unique messages: {len(all_messages):,}")
    print(f"  User: {user_msgs:,}")
    print(f"  Assistant: {assistant_msgs:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Estimated tokens: {total_chars // 4:,}")
    print(f"Date range: {earliest} to {latest}")


if __name__ == '__main__':
    build_dataset()
