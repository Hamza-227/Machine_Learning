# ❤️ Heart Stroke Prediction App

A simple web app built with **Streamlit** that predicts the risk of heart disease based on a patient's clinical details, using a trained **K-Nearest Neighbors (KNN)** classification model.

---

## 🚀 Features

- Interactive form to enter patient details (age, sex, chest pain type, blood pressure, cholesterol, etc.)
- Real-time prediction of heart disease risk (High / Low)
- Clean, organized UI with sidebar info and an expandable summary of entered details
- Powered by a pre-trained scikit-learn KNN model with feature scaling

---

## 🧠 Tech Stack

- **Python**
- **Streamlit** — web app framework
- **scikit-learn** — KNN model & scaling
- **pandas** — data handling
- **joblib** — model/scaler serialization

---

## 📁 Project Files

```
├── app.py                 # Main Streamlit application
├── knn_heart_model.pkl    # Trained KNN classification model
├── heart_scaler.pkl       # Fitted StandardScaler for input features
├── heart_columns.pkl      # Expected feature columns (post one-hot encoding)
└── README.md
```

> **Note:** The `.pkl` files must be in the same directory as `app.py` for the app to run correctly.

---

## ⚙️ Installation

1. Clone or download this project folder.
2. (Recommended) Create and activate a virtual environment.
3. Install the required dependencies:

```bash
pip install streamlit pandas scikit-learn joblib
```

---

## ▶️ Usage

Run the app with the **Streamlit CLI** (not `python app.py`):

```bash
streamlit run app.py
```

This will open the app in your default browser at `http://localhost:8501`.

---

## 📝 Input Features

| Feature | Description | Values |
|---|---|---|
| Age | Patient age | 18–100 |
| Sex | Biological sex | M / F |
| Chest Pain Type | Type of chest pain | ATA, NAP, TA, ASY |
| Resting BP | Resting blood pressure (mm Hg) | 80–200 |
| Cholesterol | Serum cholesterol (mg/dL) | 100–600 |
| Fasting Blood Sugar | > 120 mg/dL | 0 / 1 |
| Resting ECG | Resting electrocardiogram results | Normal, ST, LVH |
| Max Heart Rate | Maximum heart rate achieved | 60–220 |
| Exercise Angina | Exercise-induced angina | Y / N |
| Oldpeak | ST depression induced by exercise | 0.0–6.0 |
| ST Slope | Slope of peak exercise ST segment | Up, Flat, Down |

---

## 🔍 How It Works

1. User inputs are collected via the Streamlit form.
2. Categorical inputs are converted to one-hot encoded columns matching the model's training format.
3. Missing columns are filled with `0`, and columns are reordered to match `heart_columns.pkl`.
4. Numerical inputs are scaled using the saved `StandardScaler`.
5. The KNN model predicts the risk class (`1` = High Risk, `0` = Low Risk).

---

## ⚠️ Disclaimer

This app is built for **educational and portfolio purposes only**. It is **not a medical diagnostic tool** and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## 👤 Author

Built by **Hamza Anjum**