# Content-Based Filtering

**Core idea**: recommend items similar to what a user has liked before, based on item attributes — not other users' behavior.

## What's in the notebook

1. Extract movie metadata (genre, title keywords) from MovieLens
2. Build TF-IDF item feature vectors
3. Compute item-item cosine similarity from features
4. Build a user profile as a weighted average of liked item vectors
5. Score all items against the user profile
6. Show recommendations and explain them
7. Demonstrate the filter bubble effect

## Key concepts
- **TF-IDF** for text feature weighting
- **Item profiles** vs. **User profiles**
- Why content-based handles new items but not new users
- The filter bubble: you only get more of what you've already liked

## Tradeoffs

| Pro | Con |
|---|---|
| Works for new items (no cold start on items) | Cold start on new users |
| Explainable ("because you liked action movies…") | Requires rich item metadata |
| No need for other users' data | Filter bubble — limited serendipity |
| Personalization scales per-user cheaply | Feature engineering is manual |
