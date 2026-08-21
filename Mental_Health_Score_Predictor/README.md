# Mental Health Signal

A full-stack machine learning project that predicts a student's **mental health / wellness score (0–10)** from their social media habits, academic load, sleep, physical activity, and self-reported stress level.

The project has two parts:

1. **Model** — a scikit-learn regression pipeline trained on ~5,000 student survey responses (`ML_Project.ipynb`).
2. **API + UI** — a FastAPI backend (`main.py`) that serves the trained model, and a static HTML/CSS/JS front end (`index.html`, `style.css`, `script.js`) that collects a user's inputs and displays the predicted score on an animated gauge.

> **Disclaimer:** This tool is for informational and educational purposes only. It is **not** a clinical or diagnostic instrument. If you or someone you know is struggling, please talk to a mental health professional or someone you trust.

---

## Table of contents

- [Demo flow](#demo-flow)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Modeling approach](#modeling-approach)
- [Model performance](#model-performance)
- [API reference](#api-reference)
- [Front end](#front-end)
- [Getting started](#getting-started)
- [Deployment](#deployment)
- [Possible improvements](#possible-improvements)

---

## Demo flow

1. User fills out a form on the web page: age, gender, country, academic level, most-used platform, purpose of use, daily screen time, phone unlocks, study hours, physical activity, sleep hours, and stress level.
2. The front end validates the inputs client-side and sends them as JSON to `POST /predict`.
3. The FastAPI backend validates the payload against a Pydantic schema, builds a single-row DataFrame matching the training feature set, and runs it through the saved scikit-learn pipeline.
4. The API returns a `predicted_mental_health_score` between 0 and 10.
5. The UI animates a gauge and shows a short, human-readable interpretation of the score ("strained" / "balanced" / "strong").

---

## Project structure

```
.
├── ML_Project.ipynb                                # Full EDA + preprocessing + model training notebook
├── ML_Project.html                                  # Notebook exported as static HTML (for viewing without Jupyter)
├── Mental_Health_Model.pkl                           # Trained scikit-learn pipeline (preprocessing + model), saved with joblib
├── Student_Social_Media_And_Mental_Health_Impact.csv # Training dataset (5,000 rows)
├── main.py                                           # FastAPI app that serves the model
├── requirements.txt                                  # Python dependencies for the API
├── index.html                                        # Front-end form + results UI
├── style.css                                         # Styling for the UI
└── script.js                                         # Front-end logic: validation, API calls, gauge rendering
```

---

## Dataset

**File:** `Student_Social_Media_And_Mental_Health_Impact.csv`
**Size:** 5,000 rows, 13 columns

| Column | Type | Description |
|---|---|---|
| `Age` | numeric | Student's age |
| `Gender` | categorical | Male / Female |
| `Country` | categorical | Student's country (grouped into top countries + "Other" for modeling — see below) |
| `Academic_Level` | categorical | High School / Undergraduate / Graduate |
| `Most_Used_Platform` | categorical | Facebook, Instagram, Snapchat, Twitter, YouTube, TikTok, LinkedIn, LINE, KakaoTalk, VKontakte, WhatsApp, WeChat |
| `Purpose_Of_Use` | categorical | Networking / Education / Entertainment / News |
| `Avg_Daily_Usage_Hours` | numeric | Average daily social media screen time (hours) |
| `Daily_Unlocks` | numeric | Number of times the phone is unlocked per day |
| `Study_Hours` | numeric | Study hours per day (right-skewed — see preprocessing) |
| `Physical_Activity_Hours` | numeric | Physical activity per day (hours) |
| `Sleep_Hours_Per_Night` | numeric | Average sleep per night (hours) |
| `Stress_Level` | ordinal | Low / Medium / High / Very High |
| `Mental_Health_Score` | numeric (target) | Self-reported wellness score, 0–10 |

**Country grouping:** high-cardinality `Country` values are collapsed into a `Grouped_country` feature — the ten most frequent countries (`India, USA, Canada, Australia, UK, Germany, Mexico, Turkey, France`) keep their own label, everything else is bucketed as `"Other"`. The API applies the exact same grouping logic at inference time so behavior matches training.

---

## Modeling approach

All preprocessing and modeling steps live in `ML_Project.ipynb`. Summary of the pipeline:

**1. Feature groups**

| Group | Columns | Treatment |
|---|---|---|
| Skewed numeric | `Study_Hours` | Impute → `log1p` transform → `StandardScaler` |
| Plain numeric | `Age`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Physical_Activity_Hours`, `Sleep_Hours_Per_Night` | Impute → `StandardScaler` |
| Ordinal | `Stress_Level` | Impute → `OrdinalEncoder` with explicit order `Low < Medium < High < Very High` |
| Nominal | `Gender`, `Academic_Level`, `Most_Used_Platform`, `Purpose_Of_Use`, `Grouped_country` | Impute → `OneHotEncoder(handle_unknown="ignore")` |

A `SimpleImputer` is included in every branch even though the training data has no missing values — it acts as a safety net so the pipeline doesn't break on messy real-world input at inference time.

All of this is glued together with a single `ColumnTransformer`, then wrapped in a `Pipeline` alongside the regressor. That means the saved `.pkl` file contains **both** the preprocessing and the model — the API never has to re-implement scaling or encoding by hand, it just calls `.predict()` on raw input.

**2. Models compared**

- `LinearRegression` — baseline
- `RandomForestRegressor` — default hyperparameters
- `RandomForestRegressor` — tuned via `RandomizedSearchCV` (15 iterations, 5-fold CV, scored on R²) over:
  - `n_estimators`: 100 / 200 / 300
  - `max_depth`: 5 / 10 / 15
  - `min_samples_split`: 2 / 5 / 10
  - `min_samples_leaf`: 1 / 2 / 4

**3. Split:** 70% train / 30% test, `random_state=42`.

**4. Final model:** the **default (untuned) Random Forest** was selected as the deployed model — it produced the best test R² and lowest error of the three candidates evaluated (the randomized search did not beat the default configuration on this dataset, likely because the default `n_estimators=100` with unrestricted depth was already close to optimal).

---

## Model performance

Metrics from the held-out 30% test split:

| Model | Test R² | Train R² | MAE | RMSE |
|---|---|---|---|---|
| Linear Regression | 0.740 | 0.724 | 0.536 | 0.676 |
| **Random Forest (default)** | **0.878** | 0.981 | **0.347** | **0.464** |
| Random Forest (tuned) | 0.865 | 0.955 | 0.369 | 0.487 |

**Takeaways**

- The Random Forest substantially outperforms the linear baseline (R² 0.88 vs 0.74), meaning the relationship between habits and wellness score is non-linear / involves interactions between features.
- The default Random Forest generalizes slightly better on this test split than the tuned variant. Train R² of 0.98 vs test R² of 0.88 indicates some overfitting, which is expected for an untuned tree ensemble — a good next step would be constraining `max_depth` or increasing `min_samples_leaf` slightly while keeping `n_estimators` around 100–200.
- Mean absolute error of ~0.35 (on a 0–10 scale) means predictions are, on average, well within half a point of the true score.

---

## API reference

**Base URL (example deployment):** `https://mansik-santulan-score.onrender.com`
Framework: **FastAPI**. Interactive docs are auto-generated at `/docs` (Swagger UI) and `/redoc`.

### `GET /`
Health/info check.

```json
{
  "status": "ok",
  "service": "Mental Health Signal API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

### `GET /health`
Lightweight liveness check used by the front end to show an "API online" indicator.

```json
{ "status": "healthy" }
```

### `POST /predict`
Runs a prediction against the trained pipeline.

**Request body**

```json
{
  "age": 21,
  "gender": "Male",
  "country": "Pakistan",
  "academic_level": "Undergraduate",
  "most_used_platform": "Instagram",
  "purpose_of_use": "Entertainment",
  "avg_daily_usage_hours": 4.5,
  "daily_unlocks": 120,
  "study_hours": 3.5,
  "physical_activity_hours": 1.0,
  "sleep_hours_per_night": 6.5,
  "stress_level": "Medium"
}
```

| Field | Type | Constraints |
|---|---|---|
| `age` | int | 10–100 |
| `gender` | enum | `Male`, `Female` |
| `country` | string | any value; grouped to `"Other"` server-side if not in the top-10 list |
| `academic_level` | enum | `High School`, `Undergraduate`, `Graduate` |
| `most_used_platform` | enum | see [Dataset](#dataset) list |
| `purpose_of_use` | enum | `Networking`, `Education`, `Entertainment`, `News` |
| `avg_daily_usage_hours` | float | 0–24 |
| `daily_unlocks` | int | ≥ 0 |
| `study_hours` | float | 0–24 |
| `physical_activity_hours` | float | 0–24 |
| `sleep_hours_per_night` | float | 0–24 |
| `stress_level` | enum | `Low`, `Medium`, `High`, `Very High` |

**Response `200`**

```json
{ "predicted_mental_health_score": 6.8 }
```

**Response `422`** — validation error (Pydantic), e.g. an out-of-range value or invalid enum. The front end maps each error back to its form field automatically.

---

## Front end

Plain HTML/CSS/JS, no build step required.

- **`index.html`** — a two-panel layout: a form on the left (profile → digital habits → lifestyle & stress) and a sticky results panel on the right with an idle / loading / result / error state, plus a "Model card" section describing the dataset and metrics above.
- **`style.css`** — a warm, editorial "wellness" theme (serif display type + mono accents on a sage/pine palette), fully responsive down to mobile.
- **`script.js`**
  - Mirrors the API's Pydantic constraints client-side before submitting, so obviously invalid input never hits the network.
  - Calls `POST /predict` and animates an SVG gauge to the returned score.
  - Parses FastAPI's `422` responses and highlights the specific offending field(s).
  - Pings `GET /health` on load to show a live "API online / unreachable" status pill.

To point the UI at a different backend, change `API_BASE` at the top of `script.js`.

---

## Getting started

### 1. Run the API locally

```bash
# from the project root
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload --port 2200
```

The API will be live at `http://127.0.0.1:2200`, with docs at `http://127.0.0.1:2200/docs`.

### 2. Run the UI locally

Update `API_BASE` in `script.js` to `http://127.0.0.1:2200` (or your deployed API URL), then simply open `index.html` in a browser, or serve the folder:

```bash
python -m http.server 5500
```

and visit `http://127.0.0.1:5500`.

### 3. Retrain the model (optional)

Open `ML_Project.ipynb` in Jupyter, run all cells, and it will re-fit the pipeline on `Student_Social_Media_And_Mental_Health_Impact.csv` and re-export `Mental_Health_Model.pkl` via `joblib`.

---

## Deployment

- **API:** any Python host that supports ASGI apps (Render, Railway, Fly.io, a Docker container behind Nginx, etc.). Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- **Front end:** any static host (GitHub Pages, Netlify, Vercel, S3 + CloudFront). Just make sure `API_BASE` in `script.js` points at the deployed API and that CORS is configured to allow the front end's origin (currently `allow_origins=["*"]` for simplicity — tighten this for production).

---

## Possible improvements

- Constrain the Random Forest (`max_depth`, `min_samples_leaf`) to close the train/test R² gap and reduce overfitting.
- Add feature importance / SHAP explanations to the API response so the UI can show *why* a score landed where it did.
- Persist submitted (anonymized) predictions for monitoring model drift over time.
- Add rate limiting and origin allow-listing on the API before public deployment.
- Add automated tests for the Pydantic schema and prediction endpoint.
