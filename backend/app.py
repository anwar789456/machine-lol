"""
FastAPI backend that serves the trained Random Forest and XGBoost models.
Endpoints:
  GET  /                  -> health check
  GET  /metrics           -> training metrics + ROC + confusion matrices + feature importances
  GET  /catalog           -> full course catalog
  POST /predict-dropout   -> RF + XGB + ANN predict dropout probability for a single child
  POST /recommend         -> both models recommend top 3 courses (with title, level, difficulty, fit score)
  POST /explain           -> SHAP feature contributions + plain-English summary for a single prediction
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

# ANN — separate try/except so the API still works if TF is unavailable
ANN_LOADED = False
ann_dropout = None
ann_preprocessor = None
try:
    from tensorflow.keras.models import load_model  # noqa: E402
    ann_dropout = load_model(MODELS_DIR / "ann_dropout.keras")
    ann_preprocessor = joblib.load(MODELS_DIR / "ann_preprocessor.pkl")
    ANN_LOADED = True
    print("[INFO] ANN loaded successfully.")
except Exception as e:
    print(f"[WARN] ANN not loaded ({type(e).__name__}: {e}). API will run RF + XGB only.")

# SHAP explainers — for the /explain endpoint
SHAP_LOADED = False
shap_explainer_rf = None
shap_explainer_xgb = None
feature_names: list[str] = []
try:
    import numpy as np  # noqa: E402
    shap_explainer_rf = joblib.load(MODELS_DIR / "shap_explainer_rf.pkl")
    shap_explainer_xgb = joblib.load(MODELS_DIR / "shap_explainer_xgb.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    SHAP_LOADED = True
    print(f"[INFO] SHAP explainers loaded ({len(feature_names)} features).")
except Exception as e:
    print(f"[WARN] SHAP not loaded ({type(e).__name__}: {e}). /explain will be unavailable.")


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
    return {
        "status": "ok",
        "models_loaded": MODELS_LOADED,
        "ann_loaded": ANN_LOADED,
        "shap_loaded": SHAP_LOADED,
    }


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

    response = {
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
    }

    # ANN prediction — applied through the same preprocessor saved by the notebook
    if ANN_LOADED:
        X_ann = ann_preprocessor.transform(X).astype("float32")
        ann_proba = float(ann_dropout.predict(X_ann, verbose=0).ravel()[0])
        response["ann"] = {
            "dropout_probability": ann_proba,
            "prediction": "will_drop_out" if ann_proba >= 0.5 else "will_stay",
            "risk_level": _risk_level(ann_proba),
        }
        # All-three agreement (binary decisions match)
        decisions = [rf_proba >= 0.5, xgb_proba >= 0.5, ann_proba >= 0.5]
        response["agreement"] = len(set(decisions)) == 1
    else:
        response["agreement"] = (rf_proba >= 0.5) == (xgb_proba >= 0.5)

    return response


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


def _humanize_feature(name: str) -> str:
    """Turn one-hot/snake_case feature names into something a human can read."""
    pretty = {
        "time_since_last_login": "days since last login",
        "streak_days": "streak length",
        "daily_minutes": "daily usage minutes",
        "quiz_avg_score": "average quiz score",
        "quiz_attempts": "number of quiz attempts",
        "videos_watched_pct": "video completion %",
        "practice_items_completed": "practice items completed",
        "writing_submissions": "writing submissions",
        "story_quiz_attempts": "story quiz attempts",
        "sessions_per_week": "sessions per week",
        "avg_session_duration": "average session duration",
        "parent_involvement_score": "parent involvement",
        "course_difficulty": "course difficulty",
        "instructor_rating": "instructor rating",
        "certification_earned": "certifications earned",
        "age": "age",
    }
    if name in pretty:
        return pretty[name]
    # one-hot encoded — e.g. "course_category_Reading", "device_type_mobile"
    if "_" in name:
        prefix, _, val = name.rpartition("_")
        prefix = pretty.get(prefix, prefix.replace("_", " "))
        return f"{prefix} = {val}"
    return name.replace("_", " ")


def _build_explanation_text(
    contribs: list[dict], dropout_prob: float, model_name: str
) -> str:
    """Generate a 2–3 sentence plain-English explanation."""
    pushing_up = [c for c in contribs if c["shap"] > 0][:3]
    pulling_down = [c for c in contribs if c["shap"] < 0][:3]
    risk_word = "high risk" if dropout_prob >= 0.6 else "medium risk" if dropout_prob >= 0.3 else "low risk"

    parts = [
        f"{model_name} estimates a {dropout_prob:.0%} dropout probability ({risk_word})."
    ]
    if pushing_up:
        causes = ", ".join(
            f"{_humanize_feature(c['feature'])} (+{c['shap']*100:.1f}%)" for c in pushing_up
        )
        parts.append(f"Top dropout-risk drivers: {causes}.")
    if pulling_down:
        protects = ", ".join(
            f"{_humanize_feature(c['feature'])} ({c['shap']*100:.1f}%)" for c in pulling_down
        )
        parts.append(f"Protective factors: {protects}.")
    return " ".join(parts)


@app.post("/explain")
def explain(profile: ChildProfile):
    """
    Return SHAP feature contributions for the given child profile, for both RF and XGB.
    Each contribution shows how much that feature pushed the prediction toward (positive)
    or away from (negative) "will drop out".
    """
    if not MODELS_LOADED:
        raise HTTPException(503, "Models not loaded.")
    if not SHAP_LOADED:
        raise HTTPException(
            503, "SHAP explainers not loaded — re-run the notebook to generate them."
        )

    import numpy as np

    X = to_frame(profile)

    # Push the profile through the same preprocessor used by RF/XGB
    prep_rf = rf_dropout.named_steps["prep"]
    prep_xgb = xgb_dropout.named_steps["prep"]
    X_rf = prep_rf.transform(X)
    X_xgb = prep_xgb.transform(X)
    if hasattr(X_rf, "toarray"):
        X_rf = X_rf.toarray()
    if hasattr(X_xgb, "toarray"):
        X_xgb = X_xgb.toarray()
    X_rf = np.asarray(X_rf, dtype="float32")
    X_xgb = np.asarray(X_xgb, dtype="float32")

    # Compute SHAP for the single row
    sv_rf = shap_explainer_rf.shap_values(X_rf)
    if isinstance(sv_rf, list):
        sv_rf = sv_rf[1]  # class 1 (dropout)
    sv_rf = np.asarray(sv_rf)
    if sv_rf.ndim == 3:
        sv_rf = sv_rf[..., 1]
    sv_rf = sv_rf.flatten()

    sv_xgb = shap_explainer_xgb.shap_values(X_xgb)
    sv_xgb = np.asarray(sv_xgb).flatten()

    # Probabilities for narrative context
    rf_proba = float(rf_dropout.predict_proba(X)[0, 1])
    xgb_proba = float(xgb_dropout.predict_proba(X)[0, 1])

    def build_contribs(shap_vals: np.ndarray, x_row: np.ndarray) -> list[dict]:
        out = []
        for i, name in enumerate(feature_names):
            out.append({
                "feature": name,
                "label": _humanize_feature(name),
                "value": float(x_row[i]),
                "shap": float(shap_vals[i]),
            })
        # Sort by absolute impact, take top 10
        out.sort(key=lambda d: abs(d["shap"]), reverse=True)
        return out[:10]

    contribs_rf = build_contribs(sv_rf, X_rf[0])
    contribs_xgb = build_contribs(sv_xgb, X_xgb[0])

    return {
        "random_forest": {
            "dropout_probability": rf_proba,
            "contributions": contribs_rf,
            "summary": _build_explanation_text(contribs_rf, rf_proba, "Random Forest"),
            "base_value": float(np.asarray(shap_explainer_rf.expected_value).flatten()[-1])
                if hasattr(shap_explainer_rf, "expected_value") else None,
        },
        "xgboost": {
            "dropout_probability": xgb_proba,
            "contributions": contribs_xgb,
            "summary": _build_explanation_text(contribs_xgb, xgb_proba, "XGBoost"),
            "base_value": float(np.asarray(shap_explainer_xgb.expected_value).flatten()[-1])
                if hasattr(shap_explainer_xgb, "expected_value") else None,
        },
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
