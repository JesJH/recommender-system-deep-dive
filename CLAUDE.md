# Recommender System Deep Dive — Project Context

## Project Goal
A portfolio-quality GitHub repo that teaches recommender systems end-to-end:
how they work, different approaches (with tradeoffs), real implementation with data,
live vs. batch serving architectures, model evaluation, and business value measurement.

## Owner
GitHub: JesJH  
Email: laydnna@gmail.com

---

## Repo Structure

```
recommender_system_deep_dive/
├── README.md                   # Entry point / table of contents
├── CLAUDE.md                   # This file — project context for Claude
│
├── 00_intro/
│   └── recommender_overview.md # What a recommender system is, why it matters
│
├── 01_collaborative_filtering/
│   ├── notebook.ipynb          # User-user and item-item CF, cosine sim, ALS
│   └── README.md
│
├── 02_content_based/
│   ├── notebook.ipynb          # TF-IDF, item feature vectors, cosine similarity
│   └── README.md
│
├── 03_matrix_factorization/
│   ├── notebook.ipynb          # SVD, ALS (implicit), NMF
│   └── README.md
│
├── 04_deep_learning/
│   ├── notebook.ipynb          # Neural CF, two-tower model, embeddings
│   └── README.md
│
├── 05_hybrid_approaches/
│   ├── notebook.ipynb          # Combining CF + content, cascading, weighted
│   └── README.md
│
├── 06_serving_architecture/
│   ├── README.md               # Live vs. batch tradeoffs
│   ├── batch_pipeline.py       # Simulated batch scoring pipeline
│   └── live_api/
│       ├── app.py              # FastAPI endpoint for real-time recs
│       └── requirements.txt
│
├── 07_evaluation/
│   ├── notebook.ipynb          # Precision@K, Recall@K, NDCG, MAP, coverage
│   └── README.md
│
├── 08_business_value/
│   ├── notebook.ipynb          # A/B testing framework, revenue lift, CTR
│   └── README.md
│
├── data/
│   ├── README.md               # Data sources and download instructions
│   └── .gitkeep
│
└── requirements.txt            # Top-level Python dependencies
```

---

## Dataset Plan
**MovieLens 100K** (or 1M) — free, well-understood, widely used in RecSys literature.
- Download: https://grouplens.org/datasets/movielens/
- Also considering: Amazon Reviews subset, Last.fm for diversity

---

## Approach / Narrative Arc
The repo is structured as a learning journey:
1. **Intro** — frame the problem; cold start, sparsity, scalability challenges
2. **Collaborative Filtering** — start simple (memory-based), show limits
3. **Content-Based** — use item metadata, show the filter bubble problem
4. **Matrix Factorization** — latent factors, implicit feedback, scalability
5. **Deep Learning** — neural CF, two-tower for large-scale retrieval
6. **Hybrid** — combine the above; ensemble and cascade strategies
7. **Serving** — batch vs. live architecture; when to use each; latency vs. freshness
8. **Evaluation** — offline metrics (NDCG, MAP) vs. online metrics (CTR, revenue)
9. **Business Value** — how to sell the model internally; A/B test design

---

## Key Tradeoffs to Highlight Per Approach

| Approach | Pros | Cons |
|---|---|---|
| User-User CF | Intuitive, no item metadata needed | Doesn't scale, cold-start on users |
| Item-Item CF | More stable, scalable than user-user | Still cold-start on new items |
| Content-Based | Handles new items, explainable | Filter bubble, needs rich metadata |
| Matrix Factorization | Scalable, captures latent taste | Opaque, cold-start on both |
| Neural CF / Two-Tower | State of the art, handles scale | Needs lots of data, harder to debug |
| Hybrid | Best of both worlds | Complex to build and maintain |

---

## Serving Architecture Summary
- **Batch (offline)**: Pre-compute recs for all users nightly. Fast to serve (just a DB lookup). Stale — misses real-time context. Best for email campaigns, homepage defaults.
- **Live (online)**: Re-score at request time using recent signals (session context, inventory). Fresh but adds latency. Best for homepage re-ranking, search, cart recommendations.
- **Two-stage (typical production)**: Batch retrieval (ANN / candidate generation) → live re-ranking model.

---

## Evaluation Metrics Plan
- **Offline**: Precision@K, Recall@K, NDCG@K, MAP, Hit Rate, Coverage, Diversity, Serendipity
- **Online**: CTR, Conversion Rate, Revenue per session, Retention
- **A/B Test design**: Holdout users, guardrail metrics, minimum detectable effect

---

## Tech Stack
- Python 3.11+
- pandas, numpy, scikit-learn
- implicit (ALS for implicit feedback)
- PyTorch (neural models)
- FastAPI (live serving demo)
- Jupyter notebooks for all analysis
- MovieLens dataset

---

## Status
- [ ] 00_intro — write overview doc
- [ ] 01_collaborative_filtering — notebook
- [ ] 02_content_based — notebook
- [ ] 03_matrix_factorization — notebook
- [ ] 04_deep_learning — notebook
- [ ] 05_hybrid_approaches — notebook
- [ ] 06_serving_architecture — batch pipeline + FastAPI demo
- [ ] 07_evaluation — metrics notebook
- [ ] 08_business_value — A/B test design + value framing
- [ ] README.md — finalize entry point

---

## Working Notes
- When continuing this project from another machine: `git pull origin main` and check the Status checklist above.
- Data is not committed to git — download MovieLens 100K from https://grouplens.org/datasets/movielens/100k/ and place in `data/ml-100k/`.
- Each notebook should be self-contained: load data, build model, evaluate, visualize.
