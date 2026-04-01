"""
MSA Model Architecture: Frozen LLM + trainable router projectors.

The base model processes documents and queries (frozen weights).
Router projectors learn to map hidden states into a routing space
where cosine similarity indicates document relevance to a query.

Two architectures included:
1. KeyProjector + QueryProjector (v1) — projects both sides to a smaller routing space
2. GatedQueryProjector (v2) — keeps document keys as raw hidden states,
   only projects queries via a gated residual bottleneck
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional


# =============================================================================
# v1 Architecture: Dual projectors to a reduced routing space
# =============================================================================

class KeyProjector(nn.Module):
    """Projects document hidden states to routing space (v1)"""
    def __init__(self, hidden_dim: int, router_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, router_dim, bias=False)
        nn.init.xavier_normal_(self.proj.weight)

    def forward(self, x):
        return self.proj(x)


class QueryProjector(nn.Module):
    """Projects query hidden states to routing space (v1)"""
    def __init__(self, hidden_dim: int, router_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, router_dim, bias=False)
        nn.init.xavier_normal_(self.proj.weight)

    def forward(self, x):
        return self.proj(x)


# =============================================================================
# v2 Architecture: Gated residual query projector (no key projection)
# =============================================================================

class GatedQueryProjector(nn.Module):
    """Projects query hidden states to match raw document hidden state space.

    Uses a gated residual bottleneck: starts near identity so untrained
    queries already match documents by semantic similarity. Training
    refines which dimensions matter for routing.

    Architecture: output = input + sigmoid(gate(input)) * up(silu(down(input)))
    """
    def __init__(self, hidden_dim: int):
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
        return x + gate * projected  # Residual connection


# =============================================================================
# Utility functions
# =============================================================================

def chunk_pool(tensor: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    """Mean pool a tensor along the sequence dimension in chunks.
    Input: [seq_len, dim] -> Output: [num_chunks, dim]

    This reduces a variable-length document to a fixed number of
    routing keys, each representing a chunk of ~64 tokens.
    """
    seq_len, dim = tensor.shape
    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_len > 0:
        tensor = F.pad(tensor, (0, 0, 0, pad_len))
    num_chunks = (seq_len + pad_len) // chunk_size
    return tensor.view(num_chunks, chunk_size, dim).mean(dim=1)


def get_routing_layers(msa_start_layer: int, num_layers: int, num_routing_layers: int = 4) -> list:
    """Select evenly-spaced layers from the latter half for routing.

    Using fewer layers (4 vs 18) reduces memory and speeds up
    both encoding and querying with minimal quality loss.
    """
    msa_layers = list(range(msa_start_layer, num_layers))
    if len(msa_layers) <= num_routing_layers:
        return msa_layers
    step = len(msa_layers) / num_routing_layers
    return [msa_layers[int(i * step)] for i in range(num_routing_layers)]


# =============================================================================
# Full MSA Model (v1 — dual projectors)
# =============================================================================

class MSAModel(nn.Module):
    """MSA Memory Model: Frozen LLM base + trainable router projectors.

    v1 architecture: both documents and queries are projected to a
    reduced routing space (router_dim). Document keys are pre-encoded
    and cached. Only query projectors are trained.

    NOTE: This version collapsed in our experiments because random-init
    key projectors destroyed semantic information. See v2 for the fix.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        router_dim: int = 256,
        chunk_size: int = 64,
        top_k: int = 16,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.router_dim = router_dim
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.device = device
        self.dtype = dtype

        # Load base model (frozen)
        print(f"Loading base model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device, trust_remote_code=True,
        )
        for param in self.base_model.parameters():
            param.requires_grad = False

        config = self.base_model.config
        self.hidden_dim = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.msa_start_layer = self.num_layers // 2

        print(f"Model: {self.num_layers} layers, hidden_dim={self.hidden_dim}")
        print(f"MSA layers: {self.msa_start_layer} to {self.num_layers - 1}")

        # Router projectors for the latter half of layers
        self.key_projectors = nn.ModuleDict({
            str(l): KeyProjector(self.hidden_dim, router_dim)
            for l in range(self.msa_start_layer, self.num_layers)
        }).to(device=device, dtype=dtype)

        self.query_projectors = nn.ModuleDict({
            str(l): QueryProjector(self.hidden_dim, router_dim)
            for l in range(self.msa_start_layer, self.num_layers)
        }).to(device=device, dtype=dtype)

        total_params = sum(p.numel() for p in self.key_projectors.parameters()) + \
                       sum(p.numel() for p in self.query_projectors.parameters())
        print(f"Router params: {total_params:,}")

    def encode_document(self, text: str) -> dict:
        """Encode a document into chunk-pooled routing keys per layer."""
        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.base_model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            )

        routing_keys = {}
        for layer_idx in range(self.msa_start_layer, self.num_layers):
            hs = outputs.hidden_states[layer_idx + 1].squeeze(0)
            with torch.no_grad():
                keys = self.key_projectors[str(layer_idx)](hs.to(self.dtype))
            pooled = chunk_pool(keys, self.chunk_size)
            routing_keys[layer_idx] = pooled.cpu()

        return routing_keys

    def retrieve(self, query_text: str, all_routing_keys: list, doc_texts: list,
                 top_k: Optional[int] = None) -> list:
        """Full retrieval: encode query, score all docs, return top-k."""
        top_k = top_k or self.top_k

        tokens = self.tokenizer(
            query_text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.base_model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            )

        scores = torch.zeros(len(all_routing_keys), device=self.device, dtype=self.dtype)
        for layer_idx in range(self.msa_start_layer, self.num_layers):
            q_hs = outputs.hidden_states[layer_idx + 1].squeeze(0).to(self.dtype)
            q_vec = self.query_projectors[str(layer_idx)](q_hs)
            q_vec = q_vec.max(dim=0).values
            q_vec = F.normalize(q_vec, dim=-1)

            for doc_i, doc_keys in enumerate(all_routing_keys):
                if layer_idx not in doc_keys:
                    continue
                k = doc_keys[layer_idx].to(device=self.device, dtype=self.dtype)
                k = F.normalize(k, dim=-1)
                scores[doc_i] += torch.matmul(k, q_vec).max()

        scores /= len(range(self.msa_start_layer, self.num_layers))
        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))

        return [
            {'doc_id': idx.item(), 'score': score.item(), 'text': doc_texts[idx.item()]}
            for score, idx in zip(top_scores, top_indices)
        ]

    def save_router(self, path: str):
        torch.save({
            'key_projectors': self.key_projectors.state_dict(),
            'query_projectors': self.query_projectors.state_dict(),
        }, path)

    def load_router(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.key_projectors.load_state_dict(state['key_projectors'])
        self.query_projectors.load_state_dict(state['query_projectors'])


# =============================================================================
# Contrastive loss function (shared by both v1 and v2)
# =============================================================================

def compute_contrastive_loss(
    query_vector: torch.Tensor,
    positive_keys: List[torch.Tensor],
    negative_keys: List[torch.Tensor],
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE contrastive loss for router training.

    For each positive document, computes:
        -log(exp(sim(q, pos)/T) / (exp(sim(q, pos)/T) + sum(exp(sim(q, neg)/T))))

    Uses max-chunk similarity: each document has multiple routing key chunks,
    and we take the maximum similarity across chunks.
    """
    q = F.normalize(query_vector, dim=-1)

    pos_sims = []
    for pk in positive_keys:
        pk = F.normalize(pk, dim=-1)
        pos_sims.append(torch.matmul(pk, q).max())
    pos_sims = torch.stack(pos_sims)

    neg_sims = []
    for nk in negative_keys:
        nk = F.normalize(nk, dim=-1)
        neg_sims.append(torch.matmul(nk, q).max())
    neg_sims = torch.stack(neg_sims)

    loss = 0.0
    for pos_sim in pos_sims:
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sims]) / temperature
        labels = torch.zeros(1, dtype=torch.long, device=logits.device)
        loss += F.cross_entropy(logits.unsqueeze(0), labels)

    return loss / len(pos_sims)


if __name__ == '__main__':
    print("Testing MSA model architecture...")
    model = MSAModel(
        model_name="Qwen/Qwen3-4B",
        router_dim=256,
        device="cpu",
        dtype=torch.float32,
    )

    doc_keys = model.encode_document("User: Hello, how are you?\nAssistant: I'm doing well.")
    print(f"\nDocument routing keys:")
    for layer_idx, keys in doc_keys.items():
        print(f"  Layer {layer_idx}: {keys.shape}")

    print("\nArchitecture test PASSED!")
