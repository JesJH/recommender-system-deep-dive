# Evaluation

A recommender system's value is judged at two levels: does it predict well on historical data (offline), and does it drive better outcomes in production (online)?

These are related but not the same — a model can improve NDCG by 5% and move CTR by 0%.

---

## Offline Metrics

Computed on a held-out test set. Fast iteration signal during development.

### Prediction Accuracy (for explicit ratings)
- **RMSE / MAE** — how close are predicted ratings to actual? Less useful in practice because ratings are not the real business goal.

### Ranking Quality (for top-K recommendations)
These assume we care about which items appear in the top-K, not their exact predicted score.

| Metric | What It Measures |
|---|---|
| **Precision@K** | Of the K items shown, what fraction are relevant? |
| **Recall@K** | Of all relevant items, what fraction appear in top-K? |
| **NDCG@K** | Precision@K weighted by rank position (higher rank = more credit) |
| **MAP** | Mean Average Precision — averages precision at each relevant item's rank |
| **Hit Rate@K** | Did at least one relevant item appear in top-K? (user-level binary) |
| **MRR** | Mean Reciprocal Rank — where does the first relevant item appear? |

### Beyond Accuracy
| Metric | What It Measures |
|---|---|
| **Coverage** | What fraction of the catalog can the system recommend? |
| **Diversity** | How different are the items in a single recommendation list? |
| **Serendipity** | Did the user discover something surprising but liked? |
| **Novelty** | Are recommendations non-obvious / not already known to user? |
| **Popularity Bias** | Is the system over-recommending popular items? |

---

## Online Metrics (A/B Test)

These require running the model in production with real users.

| Metric | Description |
|---|---|
| **CTR** | Click-through rate on recommended items |
| **Conversion Rate** | Did the user purchase / complete the action? |
| **Revenue per session** | Direct business impact |
| **Engagement time** | Did the user spend more time? |
| **Return rate** | Did the user come back? |
| **Guardrail metrics** | Metrics that must not regress: latency, error rate, complaint rate |

---

## The Gap Between Offline and Online

High offline NDCG does not guarantee high CTR. Common failure modes:
1. **Popularity bias** — offline test sets are dominated by popular items; model learns to recommend popular items → looks good offline, but not novel online
2. **Feedback loop** — offline evaluation uses historical interactions generated under the old system; the new system creates new feedback
3. **Context mismatch** — offline evaluation ignores session context, time of day, etc. that matter online

**Rule of thumb**: use offline metrics to eliminate bad models quickly, but trust only A/B test results for shipping decisions.

---

## What's in the Notebook

1. Build train/test split (leave-last-out per user)
2. Implement Precision@K, Recall@K, NDCG@K, Hit Rate from scratch
3. Evaluate all approaches built in previous sections on the same test set
4. Plot coverage and popularity distribution of recommendations
5. A/B test statistical power calculation: minimum sample size for a given effect size
