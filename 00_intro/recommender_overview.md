# What Is a Recommender System?

> **About this project**: Everything here is implemented from scratch using real data (MovieLens 100K). The goal is not to use a library as a black box, but to build each approach by hand — so the math, the tradeoffs, and the failure modes are visible and understood. If you want to *know* recommender systems, not just use them, this is the approach.

---

## The Problem

A recommender system's job is to surface the right item to the right user at the right time — at scale, and without requiring the user to explicitly search for it.

The inputs are almost always some combination of:
- **User behavior** (clicks, purchases, ratings, watch time)
- **Item attributes** (genre, price, description, tags)
- **Context** (time of day, device, location, what else is in the cart)

The output is a ranked list of items the user is likely to find relevant.

---

## Why It's Hard

### 1. Sparsity
A typical user interacts with a tiny fraction of the catalog. Netflix has ~15,000 titles. A user watches ~200 in their lifetime. That's 1.3% coverage — the interaction matrix is 98.7% empty. Any model must generalize from these sparse signals.

### 2. Cold Start
- **New user**: No history. What do you show them?
- **New item**: No interactions yet. How do you recommend it?
Both are unsolved in the pure collaborative world — you need content-based or hybrid strategies.

### 3. Implicit vs. Explicit Feedback
Most real-world signals are implicit (clicks, views, time spent) not explicit (5-star ratings). Implicit signals are noisy: a click could mean interest or curiosity or an accident. Not clicking doesn't mean disinterest — maybe the user never saw the item.

### 4. Popularity Bias
Algorithms trained on interaction data will amplify already-popular items. This creates a feedback loop: popular items get recommended more → get more clicks → become more popular. Long-tail items get buried.

### 5. Filter Bubbles
A system that only recommends what users already like narrows their world over time. This is especially pronounced in content-based approaches.

### 6. Scale
Production systems may need to rank thousands of items for millions of users, in real time, under latency budgets of ~100ms. Brute-force similarity search is infeasible — you need approximate nearest neighbors (ANN) and two-stage architectures.

---

## The Landscape of Approaches

```
Recommender Systems
│
├── Collaborative Filtering       ← based on user-item interaction patterns
│   ├── Memory-Based (user-user, item-item)
│   └── Model-Based (matrix factorization, neural CF)
│
├── Content-Based Filtering       ← based on item/user attributes
│
├── Knowledge-Based               ← explicit rules / constraints (not covered here)
│
└── Hybrid                        ← combine two or more of the above
```

---

## What We'll Build

This repo walks through each major family of approaches, starting simple and building toward production systems:

1. **Collaborative Filtering** (memory-based) — simplest starting point, great intuition
2. **Matrix Factorization** — the workhorse of production RecSys for a decade
3. **Content-Based** — what happens when you bring in item metadata
4. **Neural CF / Two-Tower** — modern deep learning approach at scale
5. **Hybrid** — combining the above
6. **Serving** — how to actually deploy this, batch vs. live
7. **Evaluation** — how to know if it's working
8. **Business Value** — how to measure and communicate impact

---

## Key Papers to Know

| Paper | Why It Matters |
|---|---|
| Amazon's item-to-item CF (Linden et al., 2003) | Showed CF could scale to e-commerce |
| Matrix Factorization Techniques (Koren et al., 2009) | Won Netflix Prize; canonical MF explanation |
| BPR: Bayesian Personalized Ranking (Rendle et al., 2009) | Standard implicit feedback optimization |
| Neural Collaborative Filtering (He et al., 2017) | Replaced MF dot product with neural network |
| Deep Neural Networks for YouTube (Covington et al., 2016) | Two-tower architecture in production |
| Wide & Deep Learning (Cheng et al., 2016) | Memorization + generalization in one model |
