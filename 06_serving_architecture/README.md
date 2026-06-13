# Serving Architecture: Batch vs. Live

The best model in the world is worthless if it can't serve recommendations under real-world constraints: latency budgets, infrastructure cost, freshness requirements, and traffic spikes.

## The Core Decision

| | Batch (Offline) | Live (Online) |
|---|---|---|
| When computed | Nightly / hourly | At request time |
| Latency to serve | ~1ms (DB lookup) | 50–200ms |
| Freshness | Stale (hours) | Real-time |
| Context awareness | None (no session) | Full (cart, session, query) |
| Infrastructure | Simple | Complex |
| Cost | Cheap to serve | Expensive to scale |
| Best for | Email, push, homepage defaults | Search re-ranking, cart, landing page |

---

## Production Pattern: Two-Stage

Almost every large-scale production system uses a two-stage approach:

```
Stage 1: RETRIEVAL (batch-friendly, fast, coarse)
  ├── Pre-compute ALS / two-tower embeddings for all users nightly
  ├── Build ANN index over item embeddings
  └── At request time: look up user embedding → ANN query → top-500 candidates

Stage 2: RANKING (live, slow, precise)
  ├── Take candidate set from Stage 1
  ├── Apply rich features: session context, inventory, price, recency
  └── Score with a ranking model → return top-10 to the UI
```

Stage 1 is mostly batch (user embeddings recomputed nightly, item index rebuilt when catalog changes).
Stage 2 is fully live — it sees real-time signals.

---

## What's in This Section

- `batch_pipeline.py` — simulates a nightly ALS scoring job: trains model, writes user-item scores to a "database" (CSV/SQLite), measures throughput
- `live_api/app.py` — FastAPI endpoint that looks up pre-computed candidates and applies a simple live re-ranker
- Discussion of where each approach fits in real system design

---

## When to Use What

**Use batch when:**
- Recommendations don't change minute-to-minute (catalog is stable, user taste changes slowly)
- You need to send recommendations proactively (email, push notification)
- Latency budget is tight and serving infra is limited
- You're early-stage and optimizing for simplicity

**Use live when:**
- Session context matters (user just searched for "thriller", show thrillers)
- Inventory changes rapidly (airline seats, flash sales)
- You need to react to in-session behavior (user skipped 3 items, adjust)
- Personalization needs to reflect events from the last few minutes

**Two-stage (batch retrieval + live ranking) when:**
- You need both freshness AND catalog scale
- This is the standard for any system with >100k items and real-time requirements

---

## Latency Budget Example

```
Budget: 150ms total
├── Auth / user lookup:          5ms
├── Stage 1 retrieval (ANN):    10ms   ← pre-computed embeddings
├── Feature fetching:           20ms   ← user features, item features
├── Stage 2 ranking (model):    50ms   ← neural ranking model
├── Post-processing / dedup:     5ms
└── Network / overhead:         60ms
```
