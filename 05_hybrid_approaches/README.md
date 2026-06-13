# Hybrid Approaches

**Core idea**: combine collaborative and content-based signals to get the strengths of both while mitigating their individual weaknesses.

## Combination Strategies

### 1. Weighted Hybrid
Score = α × CF_score + (1-α) × CB_score

Simple, interpretable. α can be tuned per-context or per-user segment.

### 2. Switching Hybrid
Use CF when user has enough history; fall back to content-based for cold-start users.
Threshold on number of interactions.

### 3. Cascade (Two-Stage)
Stage 1 — CF retrieves a candidate set (fast, coarse)
Stage 2 — Content-based or a ranking model re-ranks the candidates (slow, precise)
This is the dominant production pattern.

### 4. Feature Augmentation
Feed collaborative signals (user embeddings from MF) as features into a content-based model.
Or: feed item content embeddings into a neural CF model.
The two approaches literally train together.

## What's in the notebook

1. Build a weighted hybrid of item-item CF and content-based
2. Implement a switching hybrid with cold-start fallback
3. Simulate the two-stage cascade (CF retrieval → CB re-ranking)
4. Compare NDCG@10 across all approaches on the same test set
5. Show how hybrids handle cold start better than pure CF

## Tradeoffs

| Strategy | Complexity | Cold Start | Best For |
|---|---|---|---|
| Weighted | Low | Partial | When both signals available |
| Switching | Low | ✓ | Simple cold-start fallback |
| Cascade | Medium | Partial | Production retrieval + ranking |
| Feature augmentation | High | ✓ (with content) | Deep learning pipelines |
