# Data

All notebooks use **MovieLens 100K** — 100,000 ratings (1–5) from 943 users on 1,682 movies, collected 1997–1998.

It's the standard benchmark dataset for RecSys research. Small enough to run locally, large enough to be meaningful.

## Download

```bash
cd data/
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
# → data/ml-100k/
```

## Key Files

| File | Description |
|---|---|
| `ml-100k/u.data` | All 100K ratings: user_id, item_id, rating, timestamp (tab-separated) |
| `ml-100k/u.item` | Movie metadata: movie_id, title, release_date, genres |
| `ml-100k/u.user` | User demographics: user_id, age, gender, occupation, zip |
| `ml-100k/u1.base` / `u1.test` | Official 80/20 train/test split #1 (there are 5 splits) |

## Schema

**u.data**
```
user_id | item_id | rating | timestamp
196       242       3        881250949
```

**u.item** (pipe-separated, 19 genre binary columns)
```
movie_id | title | release_date | video_date | IMDb_URL | genres...
1        | Toy Story (1995) | 01-Jan-1995 | ...
```

## Larger Datasets (for when you scale up)

| Dataset | Size | Notes |
|---|---|---|
| MovieLens 1M | 1M ratings | Same format, more users |
| MovieLens 25M | 25M ratings | Full-scale; requires more memory |
| Amazon Reviews | 230M+ reviews | Real e-commerce; product metadata |
| Last.fm | 17M plays | Music domain; artist/track metadata |
