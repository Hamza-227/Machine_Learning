"""
Emotion Detection from Text — Streamlit App
Stack: pandas, scikit-learn (TF-IDF + Logistic Regression), NLTK
"""
import pickle
import string

import nltk
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="centered",
)

EMOTION_STYLE = {
    "joy":      {"emoji": "😊", "color": "#F5C518"},
    "sadness":  {"emoji": "😢", "color": "#4A90D9"},
    "anger":    {"emoji": "😠", "color": "#E5484D"},
    "fear":     {"emoji": "😨", "color": "#8E5FE0"},
    "love":     {"emoji": "❤️", "color": "#E5588A"},
    "surprise": {"emoji": "😲", "color": "#3DBE6C"},
}

# ---------------------------------------------------------------------------
# NLTK setup (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))


stop_words = setup_nltk()

# ---------------------------------------------------------------------------
# Load model artifacts (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)
    return model, vectorizer, label_map


model, vectorizer, label_map = load_artifacts()

# ---------------------------------------------------------------------------
# Preprocessing — identical pipeline to training
# ---------------------------------------------------------------------------
def remove_punc(txt):
    return txt.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(txt):
    return ''.join(ch for ch in txt if not ch.isdigit())


def remove_emojis(txt):
    return ''.join(ch for ch in txt if ch.isascii())


def remove_stopwords(txt):
    return ' '.join(w for w in txt.split() if w not in stop_words)


def clean_text(txt):
    txt = txt.lower()
    txt = remove_punc(txt)
    txt = remove_numbers(txt)
    txt = remove_emojis(txt)
    txt = remove_stopwords(txt)
    return txt


def predict_emotion(text: str):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred_label = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    emotion = label_map[pred_label]
    proba_df = pd.DataFrame({
        "emotion": [label_map[i] for i in range(len(proba))],
        "probability": proba,
    }).sort_values("probability", ascending=False)
    return emotion, proba_df, cleaned


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🎭 Emotion Detector")
st.caption("TF-IDF + Logistic Regression · trained on 16,000 labeled sentences")

with st.sidebar:
    st.header("About")
    st.write(
        "This app classifies the emotion behind a piece of text into one of "
        "six categories: **joy, sadness, anger, fear, love, surprise**."
    )
    st.write("**Pipeline:**")
    st.markdown(
        "- Lowercasing\n"
        "- Punctuation removal\n"
        "- Number removal\n"
        "- Non-ASCII / emoji removal\n"
        "- Stopword removal (NLTK)\n"
        "- TF-IDF vectorization\n"
        "- Logistic Regression classifier"
    )
    st.divider()
    st.write("Test set accuracy: **86.3%**")

text_input = st.text_area(
    "Enter a sentence to analyze:",
    placeholder="e.g. I can't believe how amazing today turned out to be!",
    height=120,
)

analyze_clicked = st.button("Analyze Emotion", type="primary", use_container_width=True)

if analyze_clicked:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        emotion, proba_df, cleaned = predict_emotion(text_input)
        style = EMOTION_STYLE.get(emotion, {"emoji": "🙂", "color": "#888888"})

        st.markdown(
            f"""
            <div style="text-align:center; padding: 1.5rem; border-radius: 12px;
                        background-color: {style['color']}22; border: 2px solid {style['color']};">
                <div style="font-size: 3rem;">{style['emoji']}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {style['color']};
                            text-transform: capitalize;">{emotion}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Confidence by emotion")
        chart = (
            alt.Chart(proba_df)
            .mark_bar()
            .encode(
                x=alt.X("probability:Q", axis=alt.Axis(format="%"), title="Probability"),
                y=alt.Y("emotion:N", sort="-x", title=None),
                color=alt.Color(
                    "emotion:N",
                    scale=alt.Scale(
                        domain=list(EMOTION_STYLE.keys()),
                        range=[v["color"] for v in EMOTION_STYLE.values()],
                    ),
                    legend=None,
                ),
                tooltip=[alt.Tooltip("emotion:N"), alt.Tooltip("probability:Q", format=".2%")],
            )
            .properties(height=220)
        )
        st.altair_chart(chart, use_container_width=True)

        with st.expander("See preprocessed text sent to the model"):
            st.code(cleaned if cleaned else "(empty after cleaning)")

st.divider()
st.caption("Built with scikit-learn, NLTK, and Streamlit.")
