# MSA Memory Experiment

**Applying Memory Sparse Attention to AI companion conversation retrieval — what worked, what didn't, and why.**

This repository documents a real attempt to use [Memory Sparse Attention (MSA)](https://arxiv.org/abs/2603.23516) for learned memory retrieval in an AI companion application. We ran this on a single RTX 5090 (32GB VRAM, 128GB RAM) using Qwen3-4B as the base model. The goal was to replace traditional embedding search (ChromaDB + sentence-transformers) with a system that *learns* which past conversations matter to a current query — not through keyword overlap, but through trained attention routing.

We got it partially working. Then it broke in two different ways. This is the story.

## The Problem

Standard RAG for companion conversations uses embedding similarity: encode the query, encode the documents, cosine similarity, return top-k. This works okay for factual retrieval ("when did we discuss X") but fails for relational retrieval ("what was the mood that week" or "what was happening when I was stressed about work"). Embedding models don't understand temporal relationships, emotional arcs, or conversational context. They match words, not meaning.

The MSA paper proposes something different: train lightweight router projectors on top of a frozen LLM that learn to map queries and documents into a routing space where relevance is defined by the training signal — not by lexical overlap. The LLM already understands language. You just need to teach it what "relevant" means in your specific context.

## Our Setup

- **Base model:** Qwen3-4B (frozen — only router projectors are trainable)
- **Hardware:** RTX 5090 32GB VRAM, 128GB DDR5 RAM, i9-14900KS
- **Dataset:** ~10K conversation turn pairs (~5M tokens, ~520 tokens avg per document)
- **Training pairs:** ~9,700 contrastive pairs (positive = temporal neighbors + entity overlap, negative = random distant documents)
- **Training time:** ~3 hours on GPU for 10 epochs
- **Trainable parameters:** ~23.6M (router projectors only — base model is frozen)

## What We Tried

### Dataset Preparation

Conversations were cleaned and structured as documents: each document is one turn pair (user message + assistant response). We stripped system prompts, memory injections, and thinking blocks — just the raw conversation.

Training pairs use two positive strategies:
1. **Temporal proximity:** Documents within ±2 positions of the query are positive (conversations close in time tend to be topically related)
2. **Entity overlap:** Documents mentioning the same specific entities (people, features, events) are positive

Negatives are random documents >50 positions away. Ratio is roughly 1:7 positive:negative.

This is a simplification from the MSA paper's approach. We didn't have the compute budget for the full training recipe, so we used these heuristics as a proxy for "these conversations are related."

### v1: Dual Projectors (COLLAPSED)

**Architecture:**
- Key projectors (random init) on document hidden states → 256-dim routing keys
- Query projectors (random init) → 256-dim routing queries
- InfoNCE contrastive loss, cosine similarity
- All 18 MSA layers (latter half of Qwen3-4B's 36 layers)

**What happened:**

First problem: we tried to cache full hidden states for all documents in RAM. At 18 layers × 9,783 documents × variable sequence lengths × hidden_dim, we OOM'd at 20% through encoding. Fix: cache only the chunk-pooled routing keys (~700MB total). This worked.

Training ran: 10 epochs, ~18 min/epoch, loss went from 2.46 → 2.17 (steady decrease). Looked healthy.

**Then we tested with real queries.** The router returned the same 4-5 documents for every query regardless of what we asked. An early conversation dominated all results.

**Why it collapsed:** The key projectors were randomly initialized and then frozen (we pre-encoded all documents through them before training). Random linear projections destroy the semantic structure of the hidden states. The keys landed in a random space where a few documents happened to cluster near a common region. The query projectors learned to match this random cluster, not actual meaning.

**Lesson:** Don't project semantically rich representations through random linear layers and then freeze them. The information loss is catastrophic and unrecoverable.

### v2: Raw Keys + Gated Query Projectors (OVERFITTED)

**Architecture changes:**
- **No key projectors** — use the base model's raw chunk-pooled hidden states directly as document representations. These are already semantically rich because Qwen3-4B is pretrained.
- **Gated residual query projectors** — instead of a simple linear projection, use a bottleneck with a learned gate: `output = input + sigmoid(gate(input)) * up(silu(down(input)))`. Starts near identity (untrained queries already match documents by semantic similarity). Training refines this.
- **4 routing layers** instead of 18 — evenly spaced from the latter half. Reduces memory and speeds up encoding/querying.
- **Lower learning rate** — 1e-4 instead of 5e-4, since we're refining a working system, not learning from scratch.
- **Baseline evaluation** — we test retrieval quality BEFORE training to confirm raw hidden states already work.

**Results:**
- Untrained baseline: some retrieval quality confirmed (raw hidden states are semantic)
- Training improved recall initially — peaked around epoch 4 at ~50% recall@16
- Then degraded: by epoch 10, recall dropped back below baseline

**Why it overfitted:** With only ~9,700 training pairs and 4 routing layers worth of projectors, the model quickly memorized the training set's positive/negative associations. The temporal proximity heuristic for positives is noisy — documents ±2 positions apart aren't always semantically related, they're just close in time. The model learned this noise rather than generalizing.

**Lesson:** The contrastive signal needs to be cleaner. Temporal proximity is a weak proxy for semantic relevance. With more compute, you could use the LLM itself to judge relevance (as the MSA paper does), but that's expensive.

## What We Learned

### 1. Raw hidden states are already good embeddings
The zero-training baseline (just mean-pooling the last hidden layer and using cosine similarity) performed surprisingly well. A pretrained LLM already understands conversational relevance better than a 384-dim sentence-transformer. If you have the VRAM to hold the model, you might not need training at all — just use the frozen model as a retrieval engine.

### 2. The contrastive signal matters more than the architecture
v1 vs v2 had different architectures but the same training signal (temporal proximity + entity overlap). Both ran into issues. The MSA paper uses the model's own attention patterns as the training signal, which is richer but requires significantly more compute (you need to run the full sequence through the model to extract attention targets).

### 3. Compute wall is real on consumer hardware
Even our simplified approach took 3 hours on an RTX 5090. The full MSA recipe (continued pretraining on the conversation corpus, then training router projectors against the model's own attention) would take days on a single GPU. The paper used multi-GPU setups. For consumer hardware, we need either smaller models, smarter training signals, or both.

### 4. Pre-encoding is the right move for inference
Caching chunk-pooled routing keys per document means retrieval at inference time is just a matrix multiplication — sub-second on GPU even for 10K documents. The encoding cost is one-time. This scales well.

### 5. Don't project through random linear layers and freeze
This seems obvious in retrospect but it's easy to fall into. If you're going to pre-encode documents, the projection must be meaningful. Either use the raw representations (v2) or train the projectors *before* pre-encoding (which requires the full document to be in memory during training — expensive).

## Open Questions

We parked this experiment and moved on, but these questions remain:

1. **Better contrastive signal:** Could we use a cheap teacher model (e.g., an embedding model or a small LLM) to judge query-document relevance and generate cleaner training pairs?

2. **Early stopping / regularization:** v2's recall peaked at epoch 4 then degraded. Would early stopping, dropout, or a different scheduler help?

3. **Batch scoring optimization:** v1's retrieval was 18 seconds per query because we scored documents individually in a Python loop. v2 pre-stacked tensors for batch matmul, which should be much faster. Has anyone benchmarked chunk-level routing on consumer GPUs?

4. **Scale:** We used Qwen3-4B. Would a larger base model (7B, 14B) produce better routing keys even without training? The hidden state quality should scale with model capability.

5. **Hybrid approach:** Use MSA routing for coarse retrieval (top-100 from 10K docs) then re-rank with a more expensive method. The router doesn't need to be perfect, just better than random.

## Repository Structure

```
├── README.md                      # This file
├── 01_build_dataset.py            # Extract and clean conversation history
├── 02_prepare_documents.py        # Convert to MSA document format
├── 03_generate_training_pairs.py  # Build contrastive training pairs
├── 04_model.py                    # MSA model architecture (frozen LLM + router projectors)
├── 05_train_v1.py                 # v1 training: dual projectors (collapsed)
├── 06_train_v2.py                 # v2 training: raw keys + gated query projectors (overfitted)
├── 07_test_retrieval.py           # Test trained router with real queries
├── 08_test_zero_training.py       # Baseline: raw hidden states, no training
├── run_training.py                # Wrapper for GPU training environment
└── TRAINING_LOG.md                # Detailed training log with metrics
```

## How to Use

These scripts expect a conversation dataset in a specific format. See `01_build_dataset.py` for the expected input structure. You'll need to adapt it to your own conversation data.

**Requirements:**
- Python 3.10+
- PyTorch 2.0+ with CUDA
- transformers
- tqdm
- A GPU with at least 16GB VRAM (we used 32GB)
- 64GB+ RAM recommended (for caching document representations)

```bash
pip install torch transformers tqdm
```

**Quick start:**
```bash
# 1. Prepare your dataset (adapt paths in each script)
python 01_build_dataset.py
python 02_prepare_documents.py
python 03_generate_training_pairs.py

# 2. Test baseline (no training needed)
python 08_test_zero_training.py

# 3. Train v2 (recommended starting point)
python 06_train_v2.py

# 4. Test retrieval
python 07_test_retrieval.py
```

## Related Work

- [Memory Sparse Attention (MSA)](https://arxiv.org/abs/2603.23516) — the paper that inspired this experiment
- [EverMind-AI/MSA](https://github.com/EverMind-AI/MSA) — official MSA implementation
- [Sanctuary](https://github.com/pulseandthread/sanctuary) — the companion application where we planned to deploy this

## Context

This was built as part of [Sanctuary](https://github.com/pulseandthread/sanctuary), an open-source local AI companion platform. The idea was to give the companion a learned memory system that understands relationships between conversations — not just keyword matching. We wanted the companion to remember *how* moments connect, not just *what* was said.

The experiment is parked, not abandoned. If the community has ideas for better training signals or more efficient approaches, we'd love to hear them.

## License

Apache 2.0
