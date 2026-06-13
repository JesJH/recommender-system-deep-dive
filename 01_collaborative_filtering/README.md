# Collaborative Filtering

**Core idea**: users who agreed in the past will agree in the future. No item metadata needed — just the pattern of who liked what.

## What's in the notebook

1. Load MovieLens 100K
2. Build a user-item rating matrix
3. User-User CF: find similar users via cosine similarity → borrow their ratings
4. Item-Item CF: find similar items → recommend items similar to what you liked
5. Compare prediction accuracy (RMSE) and coverage
6. Show where each breaks down

## Key concepts
- **Cosine similarity** vs. Pearson correlation for similarity
- **K-nearest neighbors** for selecting the peer group
- Why item-item CF scales better than user-user CF
- The cold-start problem in practice

## Tradeoffs

| | User-User | Item-Item |
|---|---|---|
| Scale | Poor (O(U²)) | Better (O(I²), items more stable) |
| Cold start | Users only | Items only |
| Sparsity sensitivity | High | Lower |
| Intuition | "People like you loved X" | "Because you liked X, try Y" |
