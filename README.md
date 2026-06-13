# Recommender Systems: A Deep Dive

A learning project built to understand recommender systems by implementing every concept from scratch — not by calling library APIs, but by writing the math, seeing it break, and fixing it with real data.

Each section builds on the last: start with the simplest possible approach, understand exactly why it works and where it fails, then layer in complexity. Everything runs on [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — real interactions from real users — so the problems you hit are real too (sparsity, cold start, popularity bias, offline/online metric gaps).

---

## What's Covered

| Section | Topic |
|---|---|
| [00 · Intro](00_intro/recommender_overview.md) | What recommender systems are, why they're hard, key challenges |
| [01 · Collaborative Filtering](01_collaborative_filtering/) | User-user & item-item CF, cosine similarity |
| [02 · Content-Based](02_content_based/) | TF-IDF, item feature vectors, filter bubble |
| [03 · Matrix Factorization](03_matrix_factorization/) | SVD, ALS (implicit feedback), NMF |
| [04 · Deep Learning](04_deep_learning/) | Neural CF, two-tower retrieval, embeddings |
| [05 · Hybrid Approaches](05_hybrid_approaches/) | Ensembles, cascading, weighted combinations |
| [06 · Serving Architecture](06_serving_architecture/) | Batch vs. live, candidate generation + re-ranking |
| [07 · Evaluation](07_evaluation/) | Offline metrics (NDCG, MAP) and online metrics (CTR, revenue) |
| [08 · Business Value](08_business_value/) | A/B test design, measuring lift, selling it internally |

---

## The Core Tradeoff Map

Every approach trades off differently across the dimensions that matter most:

```
                   ┌─────────────────────────────────┐
                   │         SCALABILITY              │
                   │              ▲                   │
                   │  Neural CF   │  Two-Tower        │
                   │      MF/ALS  │                   │
                   │──────────────┼──────────────────▶│
       COLD-START  │              │         COLD-START │
       (existing)  │ Item-Item CF │ Content-Based      │
                   │ User-User CF │                    │
                   │              ▼                   │
                   │          SIMPLICITY               │
                   └─────────────────────────────────┘
```

| Approach | Cold Start | Scale | Explainability | Data Needed |
|---|---|---|---|---|
| User-User CF | ✗ Users | ✗ | ✓ | Ratings |
| Item-Item CF | ✗ Items | ✓ | ✓ | Ratings |
| Content-Based | ✓ Items | ✓ | ✓✓ | Item metadata |
| Matrix Factorization | ✗ Both | ✓✓ | ✗ | Ratings / implicit |
| Neural CF | ✗ Both | ✓✓✓ | ✗ | Lots of data |
| Hybrid | ✓/✗ | ✓✓ | ✗ | Both |

---

## Dataset

All notebooks use [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — 100,000 ratings from 943 users on 1,682 movies.

**Setup:**
```bash
mkdir -p data/ml-100k
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip -d data/
```

---

## Serving Architecture (TL;DR)

```
                   ┌─────────────────────────────────────────────┐
                   │              PRODUCTION PATTERN              │
                   │                                             │
                   │  ┌──────────┐    ┌──────────┐              │
                   │  │  Batch   │    │   Live   │              │
                   │  │  (nightly│    │(real-time│              │
                   │  │  ALS /   │───▶│re-ranking│◀── session   │
                   │  │  ANN     │    │  model)  │    context   │
                   │  │candidates│    │          │              │
                   │  └──────────┘    └──────────┘              │
                   │       ▲                │                    │
                   │  all users         top-K recs               │
                   │  pre-computed      personalized             │
                   └─────────────────────────────────────────────┘
```

- **Batch**: pre-compute candidates for every user overnight. Fast to serve (DB lookup), but stale.
- **Live**: re-rank candidates at request time using fresh session signals. Adds latency, gains freshness.
- **Two-stage (production standard)**: batch retrieval → live re-ranking.

See [06 · Serving Architecture](06_serving_architecture/) for implementation.

---

## Evaluation Philosophy

Offline metrics (NDCG, precision@K) tell you if your model ranks things correctly on historical data. They do **not** tell you if users will actually click, buy, or return. You need both:

- **Offline**: fast iteration signal during development
- **Online (A/B test)**: the only ground truth that matters for business decisions

See [07 · Evaluation](07_evaluation/) and [08 · Business Value](08_business_value/).

---

## Philosophy

Every model in this repo is implemented from scratch in Python/NumPy/PyTorch. The `implicit` library appears in section 03 only after we've already built ALS by hand — to show what the production version looks like once you understand what it's doing. Same pattern for deep learning: write the forward pass yourself before leaning on a framework.

The goal is to finish this repo knowing *why* each line of code exists.

## Setup

```bash
git clone https://github.com/JesJH/recommender_system_deep_dive.git
cd recommender_system_deep_dive
pip install -r requirements.txt
# Download data:
cd data && curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip && unzip ml-100k.zip
jupyter lab
```

Python 3.11+ recommended.

---

## Tech Stack

- `pandas`, `numpy`, `scikit-learn` — data wrangling and baseline models
- `implicit` — ALS for implicit feedback (industrial-strength)
- `PyTorch` — neural CF and two-tower models
- `FastAPI` — live serving demo
- `Jupyter` — all analysis and walkthroughs
