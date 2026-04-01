"""
v2 Training: Raw hidden state keys + gated query projectors.

Key differences from v1:
1. No key projectors — use base model's raw chunk-pooled hidden states
2. Gated residual query projectors (start near identity)
3. Only 4 routing layers instead of all 18
4. Lower learning rate (1e-4 vs 5e-4)

RESULT: Peaked at ~50% recall@16 at epoch 4, then degraded.
See README.md for analysis.
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import random
import time
import gc

# === ADAPT THESE PATHS ===
PROJECT_DIR = Path(__file__).parent
DOCS_FILE = PROJECT_DIR / "dataset" / "documents.json"
PAIRS_FILE = PROJECT_DIR / "dataset" / "training_pairs.jsonl"
OUTPUT_DIR = PROJECT_DIR / "checkpoints_v2"
CACHE_DIR = PROJECT_DIR / "cache_v2"

# Training config
MODEL_NAME = "Qwen/Qwen3-4B"
CHUNK_SIZE = 64
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-4  # Lower LR — refining, not learning from scratch
TEMPERATURE = 0.07
DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_POSITIVES = 4
MAX_NEGATIVES = 16
NUM_ROUTING_LAYERS = 4  # Evenly-spaced from latter half


class GatedQueryProjector(nn.Module):
    """Gated residual bottleneck — starts near identity."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.down = nn.Linear(hidden_dim, hidden_dim // 4, bias=False)
        self.up = nn.Linear(hidden_dim // 4, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.xavier_normal_(self.down.weight)
        nn.init.xavier_normal_(self.up.weight)
        nn.init.xavier_normal_(self.gate.weight)

    def forward(self, x):
        projected = self.up(F.silu(self.down(x)))
        gate = torch.sigmoid(self.gate(x))
        return x + gate * projected


def chunk_pool(tensor, chunk_size=64):
    seq_len, dim = tensor.shape
    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_len > 0:
        tensor = F.pad(tensor, (0, 0, 0, pad_len))
    num_chunks = (seq_len + pad_len) // chunk_size
    return tensor.view(num_chunks, chunk_size, dim).mean(dim=1)


def get_routing_layers(msa_start_layer, num_layers, num_routing_layers):
    msa_layers = list(range(msa_start_layer, num_layers))
    if len(msa_layers) <= num_routing_layers:
        return msa_layers
    step = len(msa_layers) / num_routing_layers
    return [msa_layers[int(i * step)] for i in range(num_routing_layers)]


def load_data():
    print("Loading documents...")
    with open(DOCS_FILE, 'r', encoding='utf-8') as f:
        documents = json.load(f)['documents']

    print("Loading training pairs...")
    pairs = []
    with open(PAIRS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"  {len(documents)} documents, {len(pairs)} training pairs")
    return documents, pairs


def pre_encode_documents(base_model, tokenizer, documents, routing_layers,
                          cache_dir, device, dtype, chunk_size):
    """Extract chunk-pooled RAW hidden states. No projection — just the
    pretrained model's representations. These are already semantically rich."""
    cache_file = Path(cache_dir) / "doc_keys_raw.pt"

    if cache_file.exists():
        print(f"Loading cached document keys from {cache_file}")
        return torch.load(cache_file, map_location='cpu', weights_only=True)

    print(f"Pre-encoding {len(documents)} documents (raw hidden states)...")
    cached = {}

    for i, doc in enumerate(tqdm(documents, desc="Encoding")):
        tokens = tokenizer(
            doc['text'], return_tensors="pt", truncation=True, max_length=2048
        ).to(device)

        with torch.no_grad():
            outputs = base_model(
                input_ids=tokens.input_ids, attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            )

        doc_keys = {}
        for layer_idx in routing_layers:
            hs = outputs.hidden_states[layer_idx + 1].squeeze(0)
            pooled = chunk_pool(hs, chunk_size)
            pooled = F.normalize(pooled.float(), dim=-1).half()
            doc_keys[layer_idx] = pooled.cpu()

        cached[doc['doc_id']] = doc_keys

        if (i + 1) % 200 == 0:
            torch.cuda.empty_cache()

    torch.save(cached, cache_file)
    print(f"Cache saved: {cache_file.stat().st_size / (1024**2):.0f} MB")
    return cached


def train_epoch(query_projectors, base_model, tokenizer, pairs, doc_keys,
                optimizer, routing_layers, epoch, total_epochs, device, dtype):
    for qp in query_projectors.values():
        qp.train()

    random.shuffle(pairs)
    total_loss = 0.0
    num_steps = 0
    optimizer.zero_grad()

    for i in range(0, len(pairs), BATCH_SIZE):
        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid = 0

        for pair in pairs[i:i + BATCH_SIZE]:
            pos_ids = [pid for pid in pair['positive_doc_ids'][:MAX_POSITIVES] if pid in doc_keys]
            neg_ids = [nid for nid in pair['negative_doc_ids'][:MAX_NEGATIVES] if nid in doc_keys]
            if not pos_ids or not neg_ids:
                continue

            tokens = tokenizer(
                pair['query'], return_tensors="pt", truncation=True, max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = base_model(
                    input_ids=tokens.input_ids, attention_mask=tokens.attention_mask,
                    output_hidden_states=True,
                )

            layer_losses = []
            for layer_idx in routing_layers:
                q_hs = outputs.hidden_states[layer_idx + 1].squeeze(0).to(dtype)
                q_projected = query_projectors[str(layer_idx)](q_hs)
                q_vec = F.normalize(q_projected.max(dim=0).values.float(), dim=-1)

                pos_sims = [torch.matmul(
                    doc_keys[pid][layer_idx].to(device=device).float(), q_vec
                ).max() for pid in pos_ids]

                neg_sims = torch.stack([torch.matmul(
                    doc_keys[nid][layer_idx].to(device=device).float(), q_vec
                ).max() for nid in neg_ids])

                for ps in pos_sims:
                    logits = torch.cat([ps.unsqueeze(0), neg_sims]) / TEMPERATURE
                    labels = torch.zeros(1, dtype=torch.long, device=device)
                    layer_losses.append(F.cross_entropy(logits.unsqueeze(0), labels))

            if layer_losses:
                batch_loss = batch_loss + torch.stack(layer_losses).mean()
                valid += 1

        if valid > 0:
            (batch_loss / valid).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for qp in query_projectors.values() for p in qp.parameters()], 1.0
            )
            optimizer.step()
            optimizer.zero_grad()
            total_loss += (batch_loss / valid).item()
            num_steps += 1

        if (i // BATCH_SIZE + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}/{total_epochs}, {(i+BATCH_SIZE)/len(pairs)*100:.0f}%, loss: {total_loss/max(num_steps,1):.4f}")

        if (i // BATCH_SIZE + 1) % 50 == 0:
            torch.cuda.empty_cache()

    return total_loss / max(num_steps, 1)


def evaluate(query_projectors, base_model, tokenizer, pairs, doc_keys,
             routing_layers, device, dtype, num_eval=100):
    for qp in query_projectors.values():
        qp.eval()

    eval_pairs = random.sample(pairs, min(num_eval, len(pairs)))
    doc_id_list = sorted(doc_keys.keys())
    hits = 0

    for pair in tqdm(eval_pairs, desc="Evaluating", leave=False):
        tokens = tokenizer(
            pair['query'], return_tensors="pt", truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = base_model(
                input_ids=tokens.input_ids, attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            )

        scores = torch.zeros(len(doc_id_list), device=device)
        for layer_idx in routing_layers:
            q_hs = outputs.hidden_states[layer_idx + 1].squeeze(0).to(dtype)
            with torch.no_grad():
                q_vec = query_projectors[str(layer_idx)](q_hs)
            q_vec = F.normalize(q_vec.max(dim=0).values.float(), dim=-1)

            for doc_i, doc_id in enumerate(doc_id_list):
                dk = doc_keys[doc_id][layer_idx].to(device=device).float()
                scores[doc_i] += torch.matmul(dk, q_vec).max()

        retrieved = set(doc_id_list[idx.item()] for idx in torch.topk(scores, min(16, len(scores))).indices)
        if set(pair['positive_doc_ids']) & retrieved:
            hits += 1

    recall = hits / len(eval_pairs)
    print(f"Recall@16: {recall:.3f} ({hits}/{len(eval_pairs)})")
    return recall


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    documents, pairs = load_data()

    print(f"\nLoading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True,
    )
    for param in base_model.parameters():
        param.requires_grad = False

    config = base_model.config
    hidden_dim = config.hidden_size
    num_layers = config.num_hidden_layers
    msa_start_layer = num_layers // 2
    routing_layers = get_routing_layers(msa_start_layer, num_layers, NUM_ROUTING_LAYERS)
    print(f"Routing layers: {routing_layers}")

    doc_keys = pre_encode_documents(
        base_model, tokenizer, documents, routing_layers,
        CACHE_DIR, DEVICE, DTYPE, CHUNK_SIZE
    )

    query_projectors = nn.ModuleDict({
        str(l): GatedQueryProjector(hidden_dim) for l in routing_layers
    }).to(device=DEVICE, dtype=DTYPE)
    print(f"Query projector params: {sum(p.numel() for p in query_projectors.parameters()):,}")

    # Baseline eval (before training)
    print("\nBaseline evaluation (untrained)...")
    baseline = evaluate(query_projectors, base_model, tokenizer, pairs, doc_keys,
                        routing_layers, DEVICE, DTYPE, num_eval=50)

    optimizer = AdamW(query_projectors.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_recall = baseline

    for epoch in range(EPOCHS):
        avg_loss = train_epoch(
            query_projectors, base_model, tokenizer, pairs, doc_keys,
            optimizer, routing_layers, epoch, EPOCHS, DEVICE, DTYPE
        )
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f}")

        if (epoch + 1) % 2 == 0:
            checkpoint = {
                'query_projectors': query_projectors.state_dict(),
                'epoch': epoch + 1,
                'config': {
                    'model_name': MODEL_NAME, 'hidden_dim': hidden_dim,
                    'num_layers': num_layers, 'msa_start_layer': msa_start_layer,
                    'routing_layers': routing_layers, 'chunk_size': CHUNK_SIZE,
                }
            }
            torch.save(checkpoint, OUTPUT_DIR / f"router_v2_epoch{epoch+1}.pt")

            recall = evaluate(query_projectors, base_model, tokenizer, pairs, doc_keys,
                              routing_layers, DEVICE, DTYPE, num_eval=50)
            if recall > best_recall:
                best_recall = recall
                torch.save(checkpoint, OUTPUT_DIR / "router_v2_best.pt")
                print(f"  New best: {best_recall:.3f}")

    print(f"\nBest recall@16: {best_recall:.3f}")


if __name__ == '__main__':
    main()
