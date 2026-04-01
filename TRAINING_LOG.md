# MSA Training Log

## Dataset
- 9,783 documents (conversation turn pairs)
- ~5M tokens, ~520 tokens avg per document
- 9,743 contrastive training pairs
- Clean: no thinking blocks, no memory injections, no formatting duplicates

## v1: Dual Projectors (COLLAPSED)

### Architecture
- Base model: Qwen3-4B (frozen)
- Key projectors: random init, Linear(hidden_dim, 256)
- Query projectors: random init, Linear(hidden_dim, 256)
- All 18 MSA layers (latter half of 36 total)
- InfoNCE contrastive loss, temperature 0.07
- Batch size 8, LR 5e-4, cosine schedule

### Pre-encoding
- Attempted full hidden state caching: OOM at 20% (~2000 docs)
- Fix: cache only chunk-pooled routing keys (~700MB total)
- Encoding time: ~10 minutes on RTX 5090

### Training
- 10 epochs, ~18 min/epoch, total ~3 hours
- Loss: 2.46 → 2.17 (steady decrease, looked healthy)

### Evaluation
- Recall@16 at epoch 4: 0.66 (best)
- Recall@16 at epoch 10: 0.62

### Collapse
- Testing with real queries: same 4-5 documents returned for EVERY query
- One early conversation dominated all results regardless of query content
- Root cause: random-init key projectors destroyed semantic structure
- The key space was random noise; query projectors learned to match that noise

## v2: Raw Keys + Gated Query Projectors (OVERFITTED)

### Architecture changes
- No key projectors — raw chunk-pooled hidden states as document keys
- Gated residual query projectors: `out = x + sigmoid(gate(x)) * up(silu(down(x)))`
- Only 4 routing layers (evenly spaced) instead of 18
- LR 1e-4 (down from 5e-4)
- Baseline eval before training

### Training
- Encoding: used raw hidden states, normalized, ~700MB cache
- 10 epochs on RTX 5090

### Results
- Untrained baseline: confirmed non-zero retrieval quality
- Peak recall@16: ~0.50 at epoch 4
- Epoch 10: degraded below baseline
- Classic overfitting curve

### Analysis
- ~9,700 training pairs is small for contrastive learning
- Temporal proximity is a noisy positive signal
- Documents ±2 positions aren't always semantically related
- Model memorized the noise rather than generalizing
