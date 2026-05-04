# MinoLingo ML — Dropout Prediction & Course Recommendation

Machine-learning project for the MinoLingo English-for-children e-learning platform.

- **Business objective:** *Réduire le taux d'abandon des cours.*
- **Data science objective:** *Utiliser le Machine Learning pour recommander un cours mieux adapté.*
- **Models:** Random Forest **vs** XGBoost (hyperparameter tuning via `GridSearchCV`).
- **Live demo:** FastAPI backend on VPS + Next.js frontend on Vercel.

## Folder layout

```
ml-dropout-recommender/
├── data/
│   ├── generate_dataset.py     # Dirty 12k-row synthetic dataset generator
│   └── courses_dataset.csv     # Generated CSV (12,180 rows including duplicates)
├── notebooks/
│   ├── eda_and_training.ipynb  # Load → clean → EDA → train → compare → save
│   └── requirements.txt
├── models/                     # Filled by the notebook (joblib + metrics.json)
├── backend/
│   ├── app.py                  # FastAPI: /predict-dropout, /recommend, /metrics
│   └── requirements.txt
├── webapp/                     # Next.js + Recharts (Vercel-ready)
└── Dockerfile                  # Backend container for VPS
```

## Step-by-step

### 1. Generate the dirty dataset (already done)

```bash
cd data
python generate_dataset.py
```

The CSV contains intentional NaNs, sentinel strings (`unknown`, `N/A`, ``), duplicate rows,
outliers (age 999, negative durations, >100% progress), inconsistent casing, and
mixed-type columns — so the notebook has real cleaning work to do.

### 2. Open the notebook and run all cells

```bash
cd notebooks
pip install -r requirements.txt
jupyter notebook eda_and_training.ipynb
```

The notebook:
1. Loads the dirty CSV.
2. **Cleans** missing values, duplicates, outliers, casing, mixed types.
3. **Visualizes** — class balance, dropout-vs-feature boxplots, correlation heatmap, dropout-rate-by-category bar charts.
4. **Trains** Random Forest with `GridSearchCV` over 5 hyperparameters and XGBoost over 6 hyperparameters.
5. **Compares** them on accuracy, precision, recall, F1, ROC-AUC, training time, and inference time.
6. **Plots** ROC curves, confusion matrices, and feature importances side-by-side.
7. **Persists** `rf_dropout.pkl`, `xgb_dropout.pkl`, `rf_recommender.pkl`, `xgb_recommender.pkl`, and `metrics.json` into `../models/`.

### 3. Run the backend locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Then open `http://localhost:8000/docs` for the OpenAPI UI.

### 4. Run the frontend locally

```bash
cd webapp
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev
```

Open `http://localhost:3000`.

## Deployment

### Backend → VPS (recommended)

```bash
docker build -t minolingo-ml-api .
docker run -d -p 8000:8000 --name ml-api --restart unless-stopped minolingo-ml-api
```

Put it behind your existing reverse proxy / SSL setup. Tighten CORS in `backend/app.py`
to only allow your Vercel domain.

### Frontend → Vercel

1. Push `webapp/` to a Git repo (or use the Vercel CLI from the folder).
2. In the Vercel dashboard set `NEXT_PUBLIC_API_URL` = `https://ml-api.your-vps-domain.com`.
3. Deploy.

## Why this design

- **Vercel for the frontend** gives you a free public demo URL and HTTPS for the live presentation.
- **VPS for the backend** because XGBoost + scikit-learn binaries push past Vercel's serverless function size limit, and your VPS already runs Qwen2.5 — adding one container is trivial.
- **Hyperparameter tuning** (12 × 5 = 60 RF combos and 96 XGB combos with 3-fold CV) covers the "many hyperparameters" requirement.
- **Two complementary tasks** (binary dropout + multi-class recommendation) directly map the two stated objectives.
