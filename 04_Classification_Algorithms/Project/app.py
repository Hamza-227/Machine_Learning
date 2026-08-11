import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Stroke Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Load saved model, scaler, and expected columns
# ---------------------------------------------------------
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------------------------------------------------------
# Light styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1.5rem;
    }
    .stButton > button {
        width: 100%;
        padding: 0.6rem 0;
        font-weight: 600;
        border-radius: 8px;
        background-color: #e63946;
        color: white;
        border: none;
    }
    .stButton > button:hover {
        background-color: #c1121f;
        color: white;
    }
    .result-box {
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This tool estimates heart disease risk from basic clinical "
        "inputs using a trained K-Nearest Neighbors model."
    )
    st.markdown("---")
    st.caption("Built by **Hamza**")
    st.caption("⚠️ For educational purposes only — not a medical diagnosis.")

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("❤️ Heart Stroke Prediction")
st.caption("by Hamza")
st.markdown("Fill in the details below to check your heart disease risk.")
st.markdown("---")

# ---------------------------------------------------------
# Collect user input (organized into sections/columns)
# ---------------------------------------------------------
st.subheader("🧍 Personal Details")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 40)
with col2:
    sex = st.selectbox("Sex", ["M", "F"])

st.subheader("🩺 Clinical Measurements")
col3, col4 = st.columns(2)
with col3:
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
with col4:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

st.subheader("💓 Heart Symptoms")
col5, col6 = st.columns(2)
with col5:
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
with col6:
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown("---")

# ---------------------------------------------------------
# When Predict is clicked
# ---------------------------------------------------------
if st.button("🔍 Predict"):

    with st.spinner("Analyzing..."):

        # Create a raw input dictionary
        raw_input = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            "Sex_" + sex: 1,
            "ChestPainType_" + chest_pain: 1,
            "RestingECG_" + resting_ecg: 1,
            "ExerciseAngina_" + exercise_angina: 1,
            "ST_Slope_" + st_slope: 1,
        }

        # Create input dataframe
        input_df = pd.DataFrame([raw_input])

        # Fill in missing columns with 0s
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[expected_columns]

        # Scale the input
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    # Recap of entered details
    with st.expander("📋 See details you entered"):
        st.table(pd.DataFrame([raw_input]).T.rename(columns={0: "Value"}))
