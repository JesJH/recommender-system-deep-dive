# Deep Learning Approaches

**Core idea**: replace the matrix factorization dot product with a neural network — more expressive, can fuse heterogeneous features (IDs, text, images, context), and powers modern large-scale retrieval.

## What's in the notebook

1. **Neural Collaborative Filtering (NCF)**: embed user and item IDs, then feed through MLP instead of dot product
2. **Two-Tower model**: separate user tower and item tower, trained with contrastive loss; allows fast ANN retrieval at inference
3. Train on MovieLens with PyTorch
4. Visualize embeddings with UMAP
5. Discuss how two-tower maps to production retrieval (FAISS)

## Key concepts
- **Embedding layers** for sparse categorical IDs
- **NCF** — replaces dot product with MLP, capturing non-linear interactions
- **Two-tower architecture** — the dominant pattern for large-scale retrieval (YouTube, Pinterest, Airbnb)
- **Contrastive / BPR loss** — training on relative preferences (clicked > not clicked)
- **ANN retrieval** — why you can't brute-force dot products across millions of items; FAISS / ScaNN
- **Feature crossing** — combining user and item features before the final layer

## Two-Tower Architecture

```
User Features          Item Features
     │                      │
  [Tower A]             [Tower B]
     │                      │
 user_emb ──── dot ──── item_emb
                │
             similarity score
```

At inference: pre-compute all item embeddings offline → ANN index. Serve user embedding online → ANN lookup. O(1) retrieval.

## Tradeoffs

| Pro | Con |
|---|---|
| State of the art accuracy | Requires large datasets to train well |
| Can fuse any feature type | Much harder to debug |
| Two-tower enables scalable retrieval | Cold start still an issue (needs fallback) |
| Captures complex non-linear patterns | Training cost and infrastructure overhead |
