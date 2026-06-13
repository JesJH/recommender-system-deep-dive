# Matrix Factorization

**Core idea**: decompose the sparse user-item interaction matrix into two dense low-rank matrices — a user embedding matrix and an item embedding matrix. The dot product of a user vector and an item vector predicts their affinity.

## What's in the notebook

1. Explicit feedback: SVD on the ratings matrix (Funk SVD)
2. Implicit feedback: ALS with the `implicit` library (confidence weighting)
3. Visualize the learned item embeddings (PCA → 2D)
4. Inspect what the latent factors seem to capture
5. Evaluate with NDCG@10 on a held-out test set
6. Compare to CF baseline

## Key concepts
- **Funk SVD** — the model that won Netflix Prize; regularized SGD on observed entries only
- **ALS (Alternating Least Squares)** — scales to millions of users/items; standard for implicit
- **Implicit feedback** — clicks/views weighted by confidence, not raw ratings
- **Latent factors** — what do dimensions of the embedding space actually represent?
- Regularization to avoid overfitting sparse data

## Tradeoffs

| Pro | Con |
|---|---|
| Scales well with ALS (parallelizable) | Cold start on both users and items |
| Captures latent taste dimensions | Embeddings aren't human-interpretable |
| Handles implicit feedback well | Harder to incorporate item metadata |
| Strong baseline for production | Static — requires periodic retraining |
