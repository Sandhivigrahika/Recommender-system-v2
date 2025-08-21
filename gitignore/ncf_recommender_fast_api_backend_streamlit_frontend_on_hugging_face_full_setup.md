# NCF Recommender – FastAPI Backend + Streamlit Frontend on Hugging Face (Full Setup)

This guide gives you an end‑to‑end, deployable architecture with **FastAPI** as a backend (in one Hugging Face Space) and **Streamlit** as a frontend (in another Space). It uses **CSV files** as the storage layer so you don’t have to rebuild your training pipeline. You can also precompute and cache recommendations to serve instantly.

---

## 1) High‑Level Architecture

```
[User Browser]
      │
      ▼
[Streamlit Frontend Space]
  - Login / Signup (calls backend)
  - Show random items to rate
  - Trigger retrain (optional)
  - Fetch personalized recs (cached or live)
      │  HTTPS (JWT auth)
      ▼
[FastAPI Backend Space (Docker)]
  - Auth (bcrypt + JWT)
  - CSV I/O: users.csv, items.csv, ratings.csv, recommendations.csv
  - Endpoints: /signup, /login, /items/random, /rate, /retrain, /recommend, /recommend/cached, /health
  - (Optional) Load NCF model and compute recs on demand
```

**Storage**: CSV files in the backend Space workspace. (For stronger persistence, later move to a 🤗 Dataset repo or a DB.)

**Security**: Passwords hashed with bcrypt; JWT for session; CORS for frontend origin.

---

## 2) Repo Layouts

### Backend Space (FastAPI, Space type: **Docker**)

```
backend/
├─ app/
│  ├─ main.py                # FastAPI app
│  ├─ auth.py                # JWT + password hashing
│  ├─ io_csv.py              # CSV read/write helpers + file locking
│  ├─ recommender.py         # hooks to your NCF model (or cached)
│  ├─ config.py              # env vars, paths, constants
│  └─ __init__.py
├─ data/
│  ├─ users.csv
│  ├─ items.csv
│  ├─ ratings.csv
│  └─ recommendations.csv    # generated after retrain (optional)
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

### Frontend Space (Streamlit, Space type: **Streamlit**)

```
frontend/
├─ app.py
├─ requirements.txt
└─ README.md
```

> Tip: Start simple: commit minimal CSVs (`items.csv` at least). `users.csv`, `ratings.csv`, and `recommendations.csv` can be created on first run if missing.

---

## 3) Backend Code (FastAPI)

### 3.1 `app/config.py`

```python
import os

# CSV paths (mounted inside the Space)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

USERS_CSV = os.path.join(DATA_DIR, "users.csv")
ITEMS_CSV = os.path.join(DATA_DIR, "items.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")
RECS_CSV = os.path.join(DATA_DIR, "recommendations.csv")

# Auth config
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-hf-secrets")
JWT_ALG = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "720"))  # 12 hours

# CORS (set your frontend Space URL here)
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

# Minimal ratings before allowing model recs
MIN_RATINGS_FOR_MODEL = int(os.getenv("MIN_RATINGS_FOR_MODEL", "3"))
```

### 3.2 `app/auth.py`

```python
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt
from passlib.context import CryptContext
from .config import JWT_SECRET, JWT_ALG, ACCESS_TOKEN_EXPIRE_MINUTES

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(subject: str, expires_minutes: Optional[int] = None) -> str:
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes or ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": subject, "exp": expire}
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)
```

### 3.3 `app/io_csv.py`

```python
import os
import json
import pandas as pd
from contextlib import contextmanager
from filelock import FileLock
from .config import USERS_CSV, ITEMS_CSV, RATINGS_CSV, RECS_CSV, DATA_DIR

# Ensure data dir exists
os.makedirs(DATA_DIR, exist_ok=True)

# Create empty CSVs if not present
for path, cols in [
    (USERS_CSV, ["user_id", "username", "password_hash"]),
    (ITEMS_CSV, ["item_id", "title"]),
    (RATINGS_CSV, ["user_id", "item_id", "rating"]),
    (RECS_CSV, ["user_id", "recommended_items"])  # items as JSON list
]:
    if not os.path.exists(path):
        pd.DataFrame(columns=cols).to_csv(path, index=False)

@contextmanager
def locked_csv(path: str):
    lock = FileLock(path + ".lock")
    with lock:
        yield


def read_csv(path: str) -> pd.DataFrame:
    with locked_csv(path):
        return pd.read_csv(path)


def write_csv(path: str, df: pd.DataFrame):
    with locked_csv(path):
        df.to_csv(path, index=False)


def append_row(path: str, row: dict):
    df = read_csv(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    write_csv(path, df)


def upsert_user(username: str, password_hash: str) -> int:
    users = read_csv(USERS_CSV)
    if (users["username"] == username).any():
        raise ValueError("User already exists")
    new_id = (users["user_id"].max() + 1) if not users.empty else 1
    append_row(USERS_CSV, {"user_id": int(new_id), "username": username, "password_hash": password_hash})
    return int(new_id)


def find_user(username: str):
    users = read_csv(USERS_CSV)
    m = users[users["username"] == username]
    return None if m.empty else m.iloc[0].to_dict()


def ensure_items_exist():
    items = read_csv(ITEMS_CSV)
    if items.empty:
        raise RuntimeError("items.csv is empty. Seed it with at least a few items.")


def get_random_items(n: int = 5):
    items = read_csv(ITEMS_CSV)
    return items.sample(min(n, len(items)), random_state=None).to_dict(orient="records")


def add_rating(user_id: int, item_id: int, rating: float):
    append_row(RATINGS_CSV, {"user_id": int(user_id), "item_id": int(item_id), "rating": float(rating)})


def count_user_ratings(user_id: int) -> int:
    ratings = read_csv(RATINGS_CSV)
    return int((ratings["user_id"] == user_id).sum())


def get_cached_recs(user_id: int):
    recs = read_csv(RECS_CSV)
    row = recs[recs["user_id"] == user_id]
    if row.empty:
        return None
    # stored as JSON string
    items = json.loads(row.iloc[0]["recommended_items"])
    return items


def set_cached_recs(user_id: int, item_ids: list[int]):
    recs = read_csv(RECS_CSV)
    recs = recs[recs["user_id"] != user_id]  # remove existing
    # append new
    recs = pd.concat([recs, pd.DataFrame([{
        "user_id": user_id,
        "recommended_items": json.dumps(item_ids)
    }])], ignore_index=True)
    write_csv(RECS_CSV, recs)
```

### 3.4 `app/recommender.py`

```python
"""
This module plugs your existing NCF notebook logic into two functions:
- train_and_save_model(...): optional, to retrain on demand
- recommend_for_user_live(...): compute recommendations on the fly

If you prefer precomputed recommendations, you can call recommend_for_user_live
for each user after training and then cache with set_cached_recs().
"""

from typing import List
import pandas as pd

# If using Keras/PyTorch, import and load here
# from tensorflow.keras.models import load_model
# model = load_model("/path/to/ncf_model.h5")

# --- Replace these stubs with your notebook logic ---

def train_and_save_model(ratings: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame, model_out_path: str | None = None):
    """Train your NCF here. Save weights to model_out_path if provided."""
    # TODO: integrate your notebook training code
    return "ok"


def recommend_for_user_live(user_id: int, users: pd.DataFrame, items: pd.DataFrame, ratings: pd.DataFrame, top_k: int = 10) -> List[int]:
    """Return a list of top item_ids for a user using the loaded NCF model."""
    # TODO: Use your model to score all unseen items for the user and return top_k item_ids
    # For now, fallback: popular items not yet rated
    rated = set(ratings.loc[ratings.user_id == user_id, "item_id"].astype(int).tolist())
    popular = ratings.groupby("item_id").size().sort_values(ascending=False).index.tolist()
    recs = [iid for iid in popular if iid not in rated][:top_k]
    if len(recs) < top_k:
        # fill from the remaining items
        remaining = [iid for iid in items.item_id.astype(int).tolist() if iid not in rated and iid not in recs]
        recs += remaining[: (top_k - len(recs))]
    return recs
```

### 3.5 `app/main.py`

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import pandas as pd

from .config import FRONTEND_ORIGIN, JWT_SECRET, JWT_ALG, MIN_RATINGS_FOR_MODEL
from .auth import hash_password, verify_password, create_access_token
from .io_csv import (
    ensure_items_exist, upsert_user, find_user, get_random_items,
    add_rating, count_user_ratings, get_cached_recs, set_cached_recs,
    USERS_CSV, ITEMS_CSV, RATINGS_CSV
)
from .recommender import train_and_save_model, recommend_for_user_live

app = FastAPI(title="NCF Recommender API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

auth_scheme = HTTPBearer()


def require_auth(creds: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    token = creds.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload["sub"]  # username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/signup")
def signup(username: str, password: str):
    try:
        ensure_items_exist()
        user_id = upsert_user(username, hash_password(password))
        token = create_access_token(subject=username)
        return {"message": "User created", "user_id": user_id, "access_token": token}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/login")
def login(username: str, password: str):
    user = find_user(username)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(subject=username)
    return {"message": "Login successful", "user_id": int(user["user_id"]), "access_token": token}


@app.get("/items/random")
def items_random(n: int = 5, _user: str = Depends(require_auth)):
    items = get_ra
```
