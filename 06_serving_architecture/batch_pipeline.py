"""
Simulated batch recommendation pipeline.

In production this would run nightly as a scheduled job (Airflow, cron, etc.).
Here we simulate: load data → train ALS → score all users → write output.
"""

import time
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

DATA_PATH = Path(__file__).parent.parent / "data" / "ml-100k" / "u.data"
DB_PATH = Path(__file__).parent / "recs.db"
TOP_K = 50  # candidates per user written to the "serving store"


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )


def build_sparse_matrix(df: pd.DataFrame):
    """Build user-item implicit confidence matrix (binarize ratings)."""
    users = df["user_id"].astype("category")
    items = df["item_id"].astype("category")
    confidence = df["rating"].values.astype(np.float32)  # use raw rating as confidence proxy
    matrix = csr_matrix(
        (confidence, (users.cat.codes, items.cat.codes)),
        shape=(users.cat.categories.size, items.cat.categories.size),
    )
    return matrix, users.cat.categories, items.cat.categories


def train_als(matrix: csr_matrix, factors: int = 50, iterations: int = 20):
    """Train ALS model using the implicit library."""
    try:
        import implicit
    except ImportError:
        raise ImportError("Run: pip install implicit")

    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=0.1,
        use_gpu=False,
    )
    model.fit(matrix)
    return model


def score_all_users(model, matrix: csr_matrix, user_ids, item_ids, top_k: int):
    """Generate top-K recommendations for every user."""
    rows = []
    for user_idx in range(matrix.shape[0]):
        item_indices, scores = model.recommend(
            user_idx, matrix[user_idx], N=top_k, filter_already_liked_items=True
        )
        user_id = int(user_ids[user_idx])
        for rank, (item_idx, score) in enumerate(zip(item_indices, scores), 1):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": int(item_ids[item_idx]),
                    "score": float(score),
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)


def write_to_db(recs: pd.DataFrame, db_path: Path):
    """Write recommendations to SQLite (stands in for Redis / DynamoDB in production)."""
    conn = sqlite3.connect(db_path)
    recs.to_sql("recommendations", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON recommendations(user_id)")
    conn.commit()
    conn.close()


def lookup_recs(user_id: int, db_path: Path, top_k: int = 10) -> list[dict]:
    """Simulate the serving-time lookup (what the live API would call)."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT item_id, score FROM recommendations WHERE user_id = ? ORDER BY rank LIMIT ?",
        (user_id, top_k),
    ).fetchall()
    conn.close()
    return [{"item_id": r[0], "score": r[1]} for r in rows]


if __name__ == "__main__":
    print("=== Batch Recommendation Pipeline ===\n")

    print("Loading data...")
    if not DATA_PATH.exists():
        print(f"ERROR: Data not found at {DATA_PATH}")
        print("Download MovieLens 100K: https://grouplens.org/datasets/movielens/100k/")
        exit(1)

    df = load_data(DATA_PATH)
    print(f"  {len(df):,} interactions | {df['user_id'].nunique()} users | {df['item_id'].nunique()} items")

    print("\nBuilding user-item matrix...")
    matrix, user_ids, item_ids = build_sparse_matrix(df)
    print(f"  Matrix shape: {matrix.shape} | Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.2%}")

    print("\nTraining ALS model...")
    t0 = time.time()
    model = train_als(matrix)
    print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\nScoring all {matrix.shape[0]} users (top-{TOP_K} candidates each)...")
    t0 = time.time()
    recs = score_all_users(model, matrix, user_ids, item_ids, TOP_K)
    print(f"  Done in {time.time() - t0:.1f}s | {len(recs):,} rows")

    print(f"\nWriting to {DB_PATH}...")
    write_to_db(recs, DB_PATH)
    print("  Done.")

    print("\n--- Sample lookup for user_id=1 ---")
    sample = lookup_recs(1, DB_PATH, top_k=10)
    for r in sample:
        print(f"  item_id={r['item_id']}  score={r['score']:.4f}")

    print("\nBatch pipeline complete.")
