"""
FastAPI backend that serves the trained Random Forest and XGBoost models.
Endpoints:
  GET  /                  -> health check
  GET  /metrics           -> training metrics + ROC + confusion matrices + feature importances
  GET  /catalog           -> full course catalog
  POST /predict-dropout   -> both models predict dropout probability for a single child
  POST /recommend         -> both models recommend top 3 courses (with title, level, difficulty, fit score)
"""

import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CATALOG_PATH = PROJECT_ROOT / "data" / "course_catalog.json"

LEVEL_ORDER = {"A1": 0, "A2": 1, "B1": 2, "B2": 3}

with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    COURSE_CATALOG = json.load(f)

app = FastAPI(title="MinoLingo Dropout & Recommendation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production to the Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Load artifacts at startup ----------
try:
    rf_dropout = joblib.load(MODELS_DIR / "rf_dropout.pkl")
    xgb_dropout = joblib.load(MODELS_DIR / "xgb_dropout.pkl")
    rf_recommender = joblib.load(MODELS_DIR / "rf_recommender.pkl")
    xgb_recommender = joblib.load(MODELS_DIR / "xgb_recommender.pkl")
    label_encoder = joblib.load(MODELS_DIR / "recommender_label_encoder.pkl")
    with open(MODELS_DIR / "metrics.json", "r") as f:
        METRICS = json.load(f)
    MODELS_LOADED = True
except FileNotFoundError as e:
    print(f"[WARN] Could not load model artifacts: {e}")
    print("[WARN] Run the notebook first to train and persist the models.")
    MODELS_LOADED = False
    METRICS = {}


# ---------- Schemas ----------
class ChildProfile(BaseModel):
    age: int = Field(..., ge=4, le=14)
    english_level: Literal["A1", "A2", "B1", "B2"]
    daily_minutes: float = Field(..., ge=0, le=240)
    sessions_per_week: float = Field(..., ge=0, le=14)
    avg_session_duration: float = Field(..., ge=0, le=120)
    streak_days: int = Field(..., ge=0, le=365)
    time_since_last_login: int = Field(..., ge=0, le=365)
    quiz_attempts: int = Field(..., ge=0, le=500)
    quiz_avg_score: float = Field(..., ge=0, le=100)
    videos_watched_pct: float = Field(..., ge=0, le=100)
    practice_items_completed: int = Field(..., ge=0, le=500)
    writing_submissions: int = Field(..., ge=0, le=200)
    story_quiz_attempts: int = Field(..., ge=0, le=200)
    certification_earned: int = Field(..., ge=0, le=1)
    course_category: Literal["Reading", "Listening", "Speaking", "Writing", "Grammar"]
    course_difficulty: int = Field(..., ge=1, le=5)
    instructor_rating: float = Field(..., ge=1, le=5)
    device_type: Literal["mobile", "tablet", "desktop"]
    subscription_type: Literal["free", "premium", "family"]
    parent_involvement_score: int = Field(..., ge=0, le=10)


def to_frame(profile: ChildProfile) -> pd.DataFrame:
    return pd.DataFrame([profile.model_dump()])


# ---------- Endpoints ----------
@app.get("/")
def health():
    return {"status": "ok", "models_loaded": MODELS_LOADED}


@app.get("/metrics")
def get_metrics():
    if not MODELS_LOADED:
        raise HTTPException(503, "Models not loaded — train via the notebook first.")
    return METRICS


@app.post("/predict-dropout")
def predict_dropout(profile: ChildProfile):
    if not MODELS_LOADED:
        raise HTTPException(503, "Models not loaded.")
    X = to_frame(profile)

    rf_proba = float(rf_dropout.predict_proba(X)[0, 1])
    xgb_proba = float(xgb_dropout.predict_proba(X)[0, 1])

    return {
        "random_forest": {
            "dropout_probability": rf_proba,
            "prediction": "will_drop_out" if rf_proba >= 0.5 else "will_stay",
            "risk_level": _risk_level(rf_proba),
        },
        "xgboost": {
            "dropout_probability": xgb_proba,
            "prediction": "will_drop_out" if xgb_proba >= 0.5 else "will_stay",
            "risk_level": _risk_level(xgb_proba),
        },
        "agreement": (rf_proba >= 0.5) == (xgb_proba >= 0.5),
    }


@app.get("/catalog")
def get_catalog():
    """Expose the full course catalog (used by the frontend)."""
    return COURSE_CATALOG


def _fit_score(course: dict, profile: ChildProfile, category_prob: float) -> float:
    """
    Combine three signals into a 0..1 fit score:
      - 60%: model's confidence in the course's category
      - 25%: how close the course's level is to the child's English level
      - 15%: how close the course's difficulty is to a comfort target
              (slightly above the child's recent course_difficulty so they're
              gently challenged, not bored or overwhelmed)
    """
    level_gap = abs(LEVEL_ORDER[course["level"]] - LEVEL_ORDER[profile.english_level])
    level_score = max(0.0, 1.0 - level_gap / 3.0)

    target_difficulty = min(5, max(1, profile.course_difficulty + 1))
    diff_gap = abs(course["difficulty"] - target_difficulty)
    diff_score = max(0.0, 1.0 - diff_gap / 4.0)

    return round(0.60 * category_prob + 0.25 * level_score + 0.15 * diff_score, 4)


def _rank_courses(category_probs: dict[str, float], profile: ChildProfile, top_k: int = 3):
    """For each catalog course, score it and return the top_k."""
    scored = []
    for course in COURSE_CATALOG:
        cat_prob = category_probs.get(course["category"], 0.0)
        scored.append({
            **course,
            "category_probability": round(cat_prob, 4),
            "fit_score": _fit_score(course, profile, cat_prob),
        })
    scored.sort(key=lambda c: c["fit_score"], reverse=True)
    return scored[:top_k]


@app.post("/recommend")
def recommend(profile: ChildProfile):
    if not MODELS_LOADED:
        raise HTTPException(503, "Models not loaded.")
    X = to_frame(profile)

    classes = label_encoder.classes_.tolist()
    rf_proba = rf_recommender.predict_proba(X)[0]
    xgb_proba = xgb_recommender.predict_proba(X)[0]

    rf_cat_probs = {c: float(p) for c, p in zip(classes, rf_proba)}
    xgb_cat_probs = {c: float(p) for c, p in zip(classes, xgb_proba)}

    rf_top_category = max(rf_cat_probs, key=rf_cat_probs.get)
    xgb_top_category = max(xgb_cat_probs, key=xgb_cat_probs.get)

    rf_courses = _rank_courses(rf_cat_probs, profile, top_k=3)
    xgb_courses = _rank_courses(xgb_cat_probs, profile, top_k=3)

    return {
        "random_forest": {
            "top_category": rf_top_category,
            "category_probabilities": rf_cat_probs,
            "courses": rf_courses,
        },
        "xgboost": {
            "top_category": xgb_top_category,
            "category_probabilities": xgb_cat_probs,
            "courses": xgb_courses,
        },
        "agreement_on_category": rf_top_category == xgb_top_category,
        "agreement_on_top_course": rf_courses[0]["id"] == xgb_courses[0]["id"],
    }


def _risk_level(p: float) -> str:
    if p < 0.30:
        return "low"
    if p < 0.60:
        return "medium"
    return "high"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
