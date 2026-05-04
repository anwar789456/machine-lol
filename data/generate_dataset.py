"""
Generate a synthetic-but-realistic dataset of 12,000 children using MinoLingo.
Intentionally DIRTY:
  - missing values (NaN, "", "N/A", "unknown")
  - duplicate rows
  - outliers (impossible ages, negative durations, > 100% progress)
  - inconsistent casing for categorical columns
  - mixed string/number types in some columns
The notebook will clean all of this.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 12_000

CATEGORIES = ["Reading", "Listening", "Speaking", "Writing", "Grammar"]
DEVICES = ["mobile", "tablet", "desktop"]
SUBSCRIPTIONS = ["free", "premium", "family"]
ENGLISH_LEVELS = ["A1", "A2", "B1", "B2"]


def generate():
    age = RNG.integers(5, 13, N)
    english_level = RNG.choice(ENGLISH_LEVELS, N, p=[0.40, 0.35, 0.20, 0.05])
    course_category = RNG.choice(CATEGORIES, N)
    course_difficulty = RNG.integers(1, 6, N)  # 1..5
    instructor_rating = np.round(RNG.normal(4.0, 0.6, N).clip(1, 5), 2)
    device_type = RNG.choice(DEVICES, N, p=[0.55, 0.30, 0.15])
    subscription_type = RNG.choice(SUBSCRIPTIONS, N, p=[0.55, 0.30, 0.15])
    parent_involvement = RNG.integers(0, 11, N)  # 0..10

    # Behavioural features — correlated with engagement
    engagement_latent = RNG.normal(0, 1, N)
    engagement_latent += (parent_involvement - 5) * 0.15
    engagement_latent += (instructor_rating - 4) * 0.3
    engagement_latent -= (course_difficulty - 3) * 0.2

    daily_minutes = (15 + engagement_latent * 8 + RNG.normal(0, 4, N)).clip(0, 120)
    sessions_per_week = (3 + engagement_latent * 1.5 + RNG.normal(0, 1, N)).clip(0, 14)
    avg_session_duration = (daily_minutes / np.maximum(sessions_per_week / 7, 0.3)).clip(2, 90)
    streak_days = (5 + engagement_latent * 4 + RNG.normal(0, 3, N)).clip(0, 60).astype(int)
    time_since_last_login = (7 - engagement_latent * 3 + RNG.normal(0, 2, N)).clip(0, 90).astype(int)

    quiz_attempts = (10 + engagement_latent * 5 + RNG.normal(0, 3, N)).clip(0, 100).astype(int)
    quiz_avg_score = (60 + engagement_latent * 12 + RNG.normal(0, 8, N)).clip(0, 100)
    videos_watched_pct = (50 + engagement_latent * 18 + RNG.normal(0, 12, N)).clip(0, 100)
    practice_items_completed = (15 + engagement_latent * 8 + RNG.normal(0, 5, N)).clip(0, 100).astype(int)
    writing_submissions = (3 + engagement_latent * 2 + RNG.normal(0, 1.5, N)).clip(0, 30).astype(int)
    story_quiz_attempts = (5 + engagement_latent * 2.5 + RNG.normal(0, 2, N)).clip(0, 40).astype(int)
    certification_earned = (RNG.random(N) < (0.10 + engagement_latent * 0.05).clip(0, 0.7)).astype(int)

    # ----- TARGET 1: dropped_out (binary) -----
    # Higher dropout when: low daily_minutes, long since last login, low streak, low quiz_score
    dropout_logit = (
        -1.4
        - engagement_latent * 1.2
        + (time_since_last_login / 10)
        - (streak_days / 15)
        - (quiz_avg_score - 60) / 25
        + (course_difficulty - 3) * 0.25
        - (parent_involvement - 5) * 0.10
        + RNG.normal(0, 0.4, N)  # noise so models don't hit 100%
    )
    dropout_prob = 1 / (1 + np.exp(-dropout_logit))
    dropped_out = (RNG.random(N) < dropout_prob).astype(int)

    # ----- TARGET 2: best_course_category (multi-class) -----
    # The category where the kid would score highest given their profile.
    # Build per-category affinity score, then argmax with noise.
    affinities = np.zeros((N, len(CATEGORIES)))
    # Reading: rewards high videos_watched_pct + low difficulty
    affinities[:, 0] = videos_watched_pct / 100 + (5 - course_difficulty) * 0.1
    # Listening: rewards mobile users + young age
    affinities[:, 1] = (device_type == "mobile") * 0.5 + (12 - age) * 0.05
    # Speaking: rewards high session duration + premium
    affinities[:, 2] = avg_session_duration / 50 + (subscription_type == "premium") * 0.4
    # Writing: rewards writing_submissions + older + B1/B2
    affinities[:, 3] = (writing_submissions / 10) + (age - 5) * 0.05 + np.isin(english_level, ["B1", "B2"]) * 0.5
    # Grammar: rewards quiz_avg_score + parent_involvement
    affinities[:, 4] = quiz_avg_score / 100 + parent_involvement * 0.05

    affinities += RNG.normal(0, 0.35, affinities.shape)  # noise
    best_course_category = np.array(CATEGORIES)[affinities.argmax(axis=1)]

    df = pd.DataFrame({
        "user_id": np.arange(1, N + 1),
        "age": age,
        "english_level": english_level,
        "daily_minutes": np.round(daily_minutes, 1),
        "sessions_per_week": np.round(sessions_per_week, 1),
        "avg_session_duration": np.round(avg_session_duration, 1),
        "streak_days": streak_days,
        "time_since_last_login": time_since_last_login,
        "quiz_attempts": quiz_attempts,
        "quiz_avg_score": np.round(quiz_avg_score, 1),
        "videos_watched_pct": np.round(videos_watched_pct, 1),
        "practice_items_completed": practice_items_completed,
        "writing_submissions": writing_submissions,
        "story_quiz_attempts": story_quiz_attempts,
        "certification_earned": certification_earned,
        "course_category": course_category,
        "course_difficulty": course_difficulty,
        "instructor_rating": instructor_rating,
        "device_type": device_type,
        "subscription_type": subscription_type,
        "parent_involvement_score": parent_involvement,
        "dropped_out": dropped_out,
        "best_course_category": best_course_category,
    })

    # =================== INTENTIONALLY DIRTY THE DATA ===================
    df = df.astype(object)

    # 1) NaN missing values (~3% of cells across some columns)
    for col in ["daily_minutes", "quiz_avg_score", "instructor_rating",
                "videos_watched_pct", "parent_involvement_score"]:
        mask = RNG.random(N) < 0.03
        df.loc[mask, col] = np.nan

    # 2) String-style missing values
    mask = RNG.random(N) < 0.02
    df.loc[mask, "english_level"] = "unknown"
    mask = RNG.random(N) < 0.02
    df.loc[mask, "device_type"] = "N/A"
    mask = RNG.random(N) < 0.015
    df.loc[mask, "subscription_type"] = ""

    # 3) Inconsistent casing
    mask = RNG.random(N) < 0.10
    df.loc[mask, "course_category"] = df.loc[mask, "course_category"].astype(str).str.upper()
    mask = RNG.random(N) < 0.08
    df.loc[mask, "device_type"] = df.loc[mask, "device_type"].astype(str).str.upper()

    # 4) Outliers
    out_idx = RNG.choice(N, size=40, replace=False)
    df.loc[out_idx[:10], "age"] = 999
    df.loc[out_idx[10:20], "daily_minutes"] = -50
    df.loc[out_idx[20:30], "videos_watched_pct"] = 250
    df.loc[out_idx[30:40], "quiz_avg_score"] = -10

    # 5) Duplicate rows (~1.5%)
    n_dup = int(N * 0.015)
    dup_idx = RNG.choice(N, size=n_dup, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    # 6) Mixed-type column: instructor_rating sometimes a string
    mask = RNG.random(len(df)) < 0.01
    df.loc[mask, "instructor_rating"] = "n/a"

    # Shuffle so duplicates are scattered
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)

    out = Path(__file__).parent / "courses_dataset.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows -> {out}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    generate()
