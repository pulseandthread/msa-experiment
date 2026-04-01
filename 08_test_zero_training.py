"""
Zero-training retrieval baseline: use base model hidden states directly.
No projectors, no training. Just the pretrained model's understanding.

Encode each document by running it through the model and mean-pooling
the last hidden layer. Query the same way. Cosine similarity for ranking.

This baseline is surprisingly strong — a pretrained LLM already has
rich semantic understanding of conversational relevance.
"""
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import time

PROJECT_DIR = Path(__file__).parent
DOCS_FILE = PROJECT_DIR / "dataset" / "documents.json"
CACHE_FILE = PROJECT_DIR / "cache_zero" / "doc_embeddings.pt"

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-4B"

# === ADAPT THESE TEST QUERIES ===
TEST_QUERIES = [
    "When did we first discuss the memory system?",
    "What happened last week?",
    "Tell me about the calendar feature",
    "When did we talk about voice chat?",
    "What was happening when the server crashed?",
    "When did we first build the memory system?",
]


def encode_text(model, tokenizer, text, device, dtype):
    """Encode text to a single embedding vector using last hidden state mean"""
    tokens = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
        )
    last_hidden = outputs.hidden_states[-1].squeeze(0)
    mask = tokens.attention_mask.squeeze(0).unsqueeze(-1).float()
    embedding = (last_hidden.float() * mask).sum(dim=0) / mask.sum(dim=0)
    return F.normalize(embedding, dim=-1)


def main():
    # Load documents
    print("Loading documents...")
    with open(DOCS_FILE, 'r', encoding='utf-8') as f:
        documents = json.load(f)['documents']
    print(f"  {len(documents)} documents")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True,
    )
    model.eval()

    # Encode all documents (or load cache)
    cache_dir = Path(CACHE_FILE).parent
    cache_dir.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists():
        print("Loading cached embeddings...")
        cache = torch.load(CACHE_FILE, map_location=DEVICE, weights_only=True)
        doc_embeddings = cache['embeddings']
        doc_ids = cache['doc_ids']
    else:
        print("Encoding all documents...")
        doc_embeddings = []
        doc_ids = []
        for doc in tqdm(documents, desc="Encoding"):
            emb = encode_text(model, tokenizer, doc['text'], DEVICE, DTYPE)
            doc_embeddings.append(emb)
            doc_ids.append(doc['doc_id'])
            if len(doc_embeddings) % 200 == 0:
                torch.cuda.empty_cache()

        doc_embeddings = torch.stack(doc_embeddings)
        print(f"Embeddings shape: {doc_embeddings.shape}")
        torch.save({'embeddings': doc_embeddings, 'doc_ids': doc_ids}, CACHE_FILE)

    print(f"\n{'='*60}")
    print(f"ZERO-TRAINING RETRIEVAL BASELINE")
    print(f"{'='*60}")

    for query in TEST_QUERIES:
        start = time.time()
        q_emb = encode_text(model, tokenizer, query, DEVICE, DTYPE)
        scores = torch.matmul(doc_embeddings, q_emb)
        top_scores, top_indices = torch.topk(scores, 5)
        elapsed = (time.time() - start) * 1000

        print(f"\nQuery: \"{query}\"")
        print(f"Time: {elapsed:.0f}ms")
        for rank, (score, idx) in enumerate(zip(top_scores, top_indices)):
            doc_id = doc_ids[idx.item()]
            doc = documents[doc_id]
            text_preview = doc['text'][:150].replace('\n', ' ')
            ts = doc.get('timestamp', '')[:10]
            print(f"  {rank+1}. [{ts}] (score: {score.item():.4f}) {text_preview}...")

    # Interactive
    print(f"\n{'='*60}")
    print("INTERACTIVE MODE — type a query (or 'quit' to exit)")
    print(f"{'='*60}")

    while True:
        try:
            query = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ('quit', 'exit', 'q'):
            break

        start = time.time()
        q_emb = encode_text(model, tokenizer, query, DEVICE, DTYPE)
        scores = torch.matmul(doc_embeddings, q_emb)
        top_scores, top_indices = torch.topk(scores, 5)
        print(f"Time: {(time.time()-start)*1000:.0f}ms")
        for rank, (score, idx) in enumerate(zip(top_scores, top_indices)):
            doc_id = doc_ids[idx.item()]
            doc = documents[doc_id]
            text_preview = doc['text'][:200].replace('\n', ' ')
            ts = doc.get('timestamp', '')[:10]
            print(f"  {rank+1}. [{ts}] (score: {score.item():.4f}) {text_preview}...")


if __name__ == '__main__':
    main()
