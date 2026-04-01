"""
Test trained MSA router with real queries.
Supports both v1 and v2 checkpoints.

Usage:
    python 07_test_retrieval.py              # v2 by default
    python 07_test_retrieval.py --version v1 # test v1
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import time
import argparse

PROJECT_DIR = Path(__file__).parent

# === ADAPT THESE TEST QUERIES ===
TEST_QUERIES = [
    "When did we first discuss the memory system?",
    "What happened last week?",
    "Tell me about the calendar feature",
    "When did we talk about voice chat?",
    "What was happening when the server crashed?",
]


class QueryProjectorV1(nn.Module):
    def __init__(self, hidden_dim, router_dim=256):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, router_dim, bias=False)
    def forward(self, x):
        return self.proj(x)


class QueryProjectorV2(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.down = nn.Linear(hidden_dim, hidden_dim // 4, bias=False)
        self.up = nn.Linear(hidden_dim // 4, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
    def forward(self, x):
        projected = self.up(F.silu(self.down(x)))
        gate = torch.sigmoid(self.gate(x))
        return x + gate * projected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='v2', choices=['v1', 'v2'])
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    if args.version == 'v1':
        docs_file = PROJECT_DIR / "dataset" / "documents.json"
        cache_file = PROJECT_DIR / "cache_v1" / "routing_keys.pt"
        checkpoint_file = PROJECT_DIR / "checkpoints_v1" / "router_best.pt"
    else:
        docs_file = PROJECT_DIR / "dataset" / "documents.json"
        cache_file = PROJECT_DIR / "cache_v2" / "doc_keys_raw.pt"
        checkpoint_file = PROJECT_DIR / "checkpoints_v2" / "router_v2_best.pt"

    # Load documents
    print("Loading documents...")
    with open(docs_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)['documents']

    # Load cached keys
    print("Loading routing keys...")
    doc_keys = torch.load(cache_file, map_location='cpu', weights_only=True)

    # Load checkpoint
    print(f"Loading {args.version} checkpoint...")
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
    config = checkpoint['config']

    if args.version == 'v2':
        routing_layers = config['routing_layers']
    else:
        routing_layers = list(range(config['msa_start_layer'], config['num_layers']))

    # Load base model
    print(f"Loading base model: {config['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        config['model_name'], torch_dtype=dtype, device_map=device, trust_remote_code=True,
    )
    for param in base_model.parameters():
        param.requires_grad = False

    # Create and load query projectors
    if args.version == 'v1':
        query_projectors = nn.ModuleDict({
            str(l): QueryProjectorV1(config['hidden_dim'], config['router_dim'])
            for l in routing_layers
        }).to(device=device, dtype=dtype)
    else:
        query_projectors = nn.ModuleDict({
            str(l): QueryProjectorV2(config['hidden_dim'])
            for l in routing_layers
        }).to(device=device, dtype=dtype)

    query_projectors.load_state_dict(checkpoint['query_projectors'])
    query_projectors.eval()

    doc_id_list = sorted(doc_keys.keys())

    # For v2: pre-stack keys for fast batch scoring
    if args.version == 'v2':
        max_chunks = max(doc_keys[did][routing_layers[0]].shape[0] for did in doc_id_list)
        stacked_keys = {}
        for layer_idx in routing_layers:
            layer_keys = []
            for did in doc_id_list:
                k = doc_keys[did][layer_idx]
                if k.shape[0] < max_chunks:
                    k = F.pad(k, (0, 0, 0, max_chunks - k.shape[0]))
                layer_keys.append(k)
            stacked_keys[layer_idx] = torch.stack(layer_keys).to(device=device).float()

    print(f"\n{'='*60}")
    print(f"MSA {args.version.upper()} RETRIEVAL TEST")
    print(f"{'='*60}")

    for query in TEST_QUERIES:
        start = time.time()

        tokens = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = base_model(
                input_ids=tokens.input_ids, attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            )

            scores = torch.zeros(len(doc_id_list), device=device)
            for layer_idx in routing_layers:
                q_hs = outputs.hidden_states[layer_idx + 1].squeeze(0).to(dtype)
                q_vec = query_projectors[str(layer_idx)](q_hs)
                q_vec = q_vec.max(dim=0).values
                q_vec = F.normalize(q_vec.float(), dim=-1)

                if args.version == 'v2':
                    all_sims = torch.matmul(stacked_keys[layer_idx], q_vec)
                    scores += all_sims.max(dim=1).values
                else:
                    for doc_i, doc_id in enumerate(doc_id_list):
                        dk = doc_keys[doc_id][layer_idx].to(device=device, dtype=dtype)
                        dk = F.normalize(dk, dim=-1)
                        scores[doc_i] += torch.matmul(dk, q_vec).max()

        top_scores, top_indices = torch.topk(scores, min(5, len(scores)))
        elapsed = (time.time() - start) * 1000

        print(f"\nQuery: \"{query}\"")
        print(f"Time: {elapsed:.0f}ms")
        for rank, (score, idx) in enumerate(zip(top_scores, top_indices)):
            doc_id = doc_id_list[idx.item()]
            doc = documents[doc_id]
            text_preview = doc['text'][:150].replace('\n', ' ')
            ts = doc.get('timestamp', '')[:10]
            print(f"  {rank+1}. [{ts}] (score: {score.item():.3f}) {text_preview}...")

    # Interactive mode
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
        tokens = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = base_model(
                input_ids=tokens.input_ids, attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            )
            scores = torch.zeros(len(doc_id_list), device=device)
            for layer_idx in routing_layers:
                q_hs = outputs.hidden_states[layer_idx + 1].squeeze(0).to(dtype)
                q_vec = query_projectors[str(layer_idx)](q_hs)
                q_vec = F.normalize(q_vec.max(dim=0).values.float(), dim=-1)
                if args.version == 'v2':
                    scores += torch.matmul(stacked_keys[layer_idx], q_vec).max(dim=1).values
                else:
                    for doc_i, doc_id in enumerate(doc_id_list):
                        dk = doc_keys[doc_id][layer_idx].to(device=device, dtype=dtype)
                        scores[doc_i] += torch.matmul(F.normalize(dk, dim=-1), q_vec).max()

        top_scores, top_indices = torch.topk(scores, min(5, len(scores)))
        print(f"Time: {(time.time()-start)*1000:.0f}ms")
        for rank, (score, idx) in enumerate(zip(top_scores, top_indices)):
            doc_id = doc_id_list[idx.item()]
            doc = documents[doc_id]
            text_preview = doc['text'][:200].replace('\n', ' ')
            ts = doc.get('timestamp', '')[:10]
            print(f"  {rank+1}. [{ts}] (score: {score.item():.3f}) {text_preview}...")


if __name__ == '__main__':
    main()
