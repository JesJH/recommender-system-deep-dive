"""
Live recommendation API.

Reads pre-computed candidates from the batch pipeline's SQLite DB,
then applies a simple live re-ranking based on a recency boost
(simulating what a real re-ranker would do with session context).

Run:
    cd 06_serving_architecture
    python batch_pipeline.py          # populate recs.db first
    uvicorn live_api.app:app --reload
"""

import sqlite3
import math
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

DB_PATH = Path(__file__).parent.parent / "recs.db"

app = FastAPI(
    title="Recommender System — Live API",
    description="Two-stage: batch candidates + live re-ranking demo",
)


class Recommendation(BaseModel):
    item_id: int
    score: float
    rank: int


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[Recommendation]
    strategy: str


def get_candidates(user_id: int, top_k: int) -> list[dict]:
    """Fetch pre-computed candidates from batch pipeline output."""
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Recommendation store not populated. Run batch_pipeline.py first.",
        )
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """
        SELECT item_id, score, rank
        FROM recommendations
        WHERE user_id = ?
        ORDER BY rank
        LIMIT ?
        """,
        (user_id, top_k * 3),  # fetch 3x to allow re-ranking to shuffle
    ).fetchall()
    conn.close()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No recommendations found for user {user_id}")
    return [{"item_id": r[0], "score": r[1], "rank": r[2]} for r in rows]


def live_rerank(candidates: list[dict], context: dict) -> list[dict]:
    """
    Simulate a live re-ranker that boosts scores based on session context.

    In production this would be a trained ranking model (e.g., LightGBM or
    a small neural net) consuming real-time features: session clicks,
    time of day, inventory signals, etc.

    Here we apply a toy boost: items whose ID matches a 'preferred_genre_items'
    list (simulated from session context) get a multiplicative boost.
    """
    boost_items = set(context.get("session_clicked_items", []))
    # Boost items that share an (imaginary) attribute with recently clicked items
    for c in candidates:
        if c["item_id"] % 10 in {item % 10 for item in boost_items}:  # toy signal
            c["score"] *= 1.2
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


@app.get("/health")
def health():
    return {"status": "ok", "db_exists": DB_PATH.exists()}


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=50),
    session_clicked: str = Query(default="", description="Comma-separated item_ids clicked this session"),
):
    """
    Return top-K recommendations for a user.

    - Retrieves batch candidates from the pre-computed store.
    - Applies live re-ranking based on session context.
    """
    context = {}
    if session_clicked:
        try:
            context["session_clicked_items"] = [int(x) for x in session_clicked.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="session_clicked must be comma-separated integers")

    candidates = get_candidates(user_id, top_k)
    strategy = "batch_candidates"

    if context.get("session_clicked_items"):
        candidates = live_rerank(candidates, context)
        strategy = "batch_candidates+live_rerank"

    top = candidates[:top_k]
    return RecommendationResponse(
        user_id=user_id,
        recommendations=[
            Recommendation(item_id=c["item_id"], score=round(c["score"], 4), rank=i + 1)
            for i, c in enumerate(top)
        ],
        strategy=strategy,
    )
