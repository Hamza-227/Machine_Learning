# 🎭 Emotion Detector

A text emotion classification app — predicts whether a sentence expresses
**joy, sadness, anger, fear, love,** or **surprise**.

Same stack as the original notebook: **pandas → NLTK preprocessing → TF-IDF →
Logistic Regression**, achieving **86.3% test accuracy** on 16,000 labeled
sentences. Deployed with **Streamlit**.

## Project structure

```
emotion_app/
├── app.py              # Streamlit app
├── train_model.py      # Training script (reproduces model.pkl / vectorizer.pkl)
├── train.txt           # Dataset (text;emotion, semicolon-separated)
├── model.pkl           # Trained Logistic Regression model
├── vectorizer.pkl       # Fitted TF-IDF vectorizer
├── label_map.pkl        # Maps numeric label -> emotion name
├── requirements.txt
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt

# (only needed once, or if you want to retrain from scratch)
python train_model.py

streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deploy to Streamlit Community Cloud (free)

1. Push this folder to a **public GitHub repo** (make sure `model.pkl`,
   `vectorizer.pkl`, and `label_map.pkl` are committed — they're small, no
   need to .gitignore them).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**, pick your repo/branch, and set the main file path to
   `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt`
   automatically and NLTK's `stopwords`/`punkt` will be downloaded on first
   run (cached for the app's lifetime via `@st.cache_resource`).

That's it — you'll get a public `https://<your-app>.streamlit.app` URL.

### Alternative: retrain instead of shipping .pkl files

If you'd rather not commit the `.pkl` files, add this to the top of `app.py`
before `load_artifacts()` is called:

```python
import os
if not os.path.exists("model.pkl"):
    os.system("python train_model.py")
```

This retrains on first boot (takes a few seconds) instead of shipping binary
artifacts in the repo.

## How it works

1. **Preprocessing** (`clean_text`): lowercase → strip punctuation → strip
   digits → strip non-ASCII characters/emojis → remove NLTK English
   stopwords.
2. **Vectorization**: fitted `TfidfVectorizer` transforms cleaned text into
   TF-IDF features.
3. **Classification**: `LogisticRegression` predicts one of 6 emotion
   classes and returns per-class probabilities, which the app displays as a
   bar chart.
