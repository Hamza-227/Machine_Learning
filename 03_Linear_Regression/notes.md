# Linear Regression — Theory & Projects

This repository documents my journey through **Linear Regression**, starting from first-principles theory (derived and hand-worked in my notes) and applied through two end-to-end regression projects: predicting **medical insurance charges** and predicting **used Ford car prices**.

---

## 📚 Table of Contents

1. [Theory Recap (from my notes)](#-theory-recap-from-my-notes)
2. [Project 1: Medical Insurance Cost Prediction](#-project-1-medical-insurance-cost-prediction)
3. [Project 2: Ford Car Price Prediction](#-project-2-ford-car-price-prediction)
4. [Tech Stack](#-tech-stack)
5. [Repository Structure](#-repository-structure)
6. [How to Run](#-how-to-run)
7. [Key Learnings](#-key-learnings)
8. [Future Improvements](#-future-improvements)

---

## 📐 Theory Recap (from my notes)

Before building the projects, I worked through the math and intuition behind Linear Regression by hand. Summary below.

### 1. Problem Statement

Linear Regression is a **supervised learning** algorithm used to predict a **continuous output (y)** from one or more **input features (x)**. Example used in my notes: predicting `Salary` from `Age`.

We assume a roughly linear relationship between input and output, and try to draw a **line of best fit** through the scattered data points.

### 2. The Line Equation

```
y = mx + b
```

- `m` → slope of the line (how much y changes per unit x)
- `b` → intercept (value of y when x = 0)
- `x` → input / data point
- `y` → predicted output

In ML notation this is written as the **hypothesis function**:

```
h(θ) = θ₀ + θ₁x
```

- `θ₀` → intercept
- `θ₁` → slope

The goal of training is to find the **perfect values of θ₀ and θ₁** that make the line fit the data as closely as possible.

### 3. Residual Error & Cost Function

The **residual error** is the vertical distance between an actual data point and the predicted point on the line — essentially how "wrong" the model is for that point.

To measure the overall error across *all* data points, we use the **Cost Function** (Mean Squared Error, halved for calculus convenience):

```
J(θ₀, θ₁) = (1 / 2m) · Σ (h(θ)⁽ⁱ⁾ − y⁽ⁱ⁾)²
```

- `m` → number of training examples
- `h(θ)⁽ⁱ⁾` → predicted value for the i-th example
- `y⁽ⁱ⁾` → actual value for the i-th example

**The entire training problem reduces to: minimize J(θ₀, θ₁).**

I manually worked through a toy example with points `{1,1}, {2,2}, {3,3}`:
- At `θ₀ = 0, θ₁ = 1` → predictions perfectly match → `J(θ) = 0` (zero error, the global minimum)
- At `θ₀ = 0, θ₁ = 0.5` → `J(θ) ≈ 0.58`
- At `θ₀ = 0, θ₁ = 0` → `J(θ) = 2.33`

Plotting `J(θ₁)` against different values of `θ₁` produces a **U-shaped curve**, with the lowest point (the **global minimum**) representing the best-fit slope.

### 4. Gradient Descent

Rather than guessing θ values by trial and error, **Gradient Descent** is the algorithm that automatically walks down the cost function curve toward the global minimum, step by step, guided by:

- The **Learning Rate (α)** — controls step size
- The **Repeat/Convergence Theorem** — keep updating θ₀ and θ₁ until the cost stops decreasing meaningfully (convergence)

Too small a learning rate → slow convergence. Too large → overshoot and fail to converge.

### 5. Multiple Linear Regression

When there is more than one input feature, the hypothesis extends to:

```
h(θ) = θ₀ + θ₁x₁ + θ₂x₂ + θ₃x₃ + ... + θₙxₙ
```

Each feature (e.g. `x₁` = Location, `x₂` = Age, `x₃` = Sleep) gets its own coefficient, and the model becomes a hyperplane fit through multi-dimensional space rather than a single 2D line.

### 6. Performance Metrics — R² and Adjusted R²

**R² (R-Squared)** measures how much of the variance in the target is explained by the model:

```
R² = 1 − (Sum of Squared Residuals / Total Sum of Squares)
   = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²
```

- Always expressed as a decimal, read as a percentage (e.g. `0.86 → 86%`)
- Higher is generally better

**Problem with R²:** it *never decreases* when you add more features — even irrelevant ones — which can be misleading. That's where **Adjusted R²** comes in:

```
Adjusted R² = 1 − [(1 − R²)(n − 1) / (n − p − 1)]
```

- `n` → number of rows (observations)
- `p` → number of features (predictors)

Adjusted R² **penalizes unnecessary features** — if a new feature doesn't genuinely improve the model, Adjusted R² will drop even if R² rises. This makes it a fairer metric for comparing models with different numbers of features. My notes trace an example: adding a feature took `R²` from 81% → 88% → 89%, while `Adjusted R²` went 79% → 89% → **86%** — showing the 3rd feature actually *hurt* the model despite R² looking better.

### 7. Overfitting vs. Underfitting

| | Training Performance | Testing Performance | Bias/Variance |
|---|---|---|---|
| **Overfitting** | Great | Poor | Low bias, high variance |
| **Underfitting** | Poor | Poor | High bias, high variance |
| **Perfect Model** | Good | Good | Low bias, low variance |

Worked example from my notes:

| Model | Training Acc. | Testing Acc. | Verdict |
|---|---|---|---|
| M1 | 92% | 64% | ❌ Overfitting |
| M2 | 92% | 90% | ✅ Perfect Model (low bias, low variance) |
| M3 | 68% | 64% | ❌ Underfitting |

### 8. Ridge & Lasso Regression (Regularization)

To combat overfitting, a **penalty term** is added to the cost function to discourage overly large coefficients (steep slopes):

```
J(θ₀, θ₁) = (1 / 2m) · Σ [h(θ)⁽ⁱ⁾ − y⁽ⁱ⁾]²  +  λ(slope)²
```

- `λ` (lambda) controls how strongly large coefficients are penalized
- **Ridge Regression** uses an L2 penalty (squared coefficients) — shrinks coefficients but rarely to zero
- **Lasso Regression** uses an L1 penalty (absolute coefficients) — can shrink coefficients all the way to zero, effectively performing feature selection

---

## 🏥 Project 1: Medical Insurance Cost Prediction

**Notebook:** `Copy_of_Insurance.ipynb`
**Goal:** Predict individual medical insurance `charges` based on demographic and lifestyle attributes.

### Dataset Features
`age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges` (target)

### Workflow

**1. Exploratory Data Analysis (EDA)**
- Checked shape, data types, summary statistics, and missing values (`df.info()`, `df.describe()`, `df.isnull().sum()`)
- Plotted distributions (histograms with KDE) for `age`, `bmi`, `children`, `charges`
- Count plots for categorical variables: `children`, `sex`, `smoker`
- Boxplots to detect outliers in numerical columns
- Correlation heatmap across numeric features

**2. Data Cleaning & Preprocessing**
- Removed duplicate rows
- Re-verified missing values and data types
- Encoded categorical variables:
  - `sex` → `is_female` (0/1)
  - `smoker` → `is_smoker` (0/1)
  - `region` → one-hot encoded (`drop_first=True`)
- Cast all columns to `int` for consistency

**3. Feature Engineering**
- Bucketed continuous `bmi` into clinically meaningful categories using `pd.cut`:
  - `Underweight` (0–18.5), `Normal` (18.5–24.9), `Overweight` (24.9–29.9), `Obese` (29.9+)
- One-hot encoded `bmi_category`
- Applied `StandardScaler` to `age`, `bmi`, `children` (required for linear regression, which is sensitive to feature magnitude)

**4. Statistical Feature Selection**
- **Pearson Correlation** computed between every candidate feature and `charges`, sorted to identify the strongest linear relationships
- **Chi-Square Test** (`chi2_contingency`) run against a binned version of `charges` (quartiles via `pd.qcut`) to test independence of categorical features from the target, at `α = 0.05` — features with `p < 0.05` were kept

**5. Final Feature Set**
```
['age', 'is_female', 'bmi', 'children', 'is_smoker', 'charges',
 'region_southeast', 'bmi_category_Obese', 'region_northwest']
```

**6. Model Training**
- 80/20 train-test split (`random_state=42`)
- `sklearn.linear_model.LinearRegression`, fit on the training set

**7. Evaluation**
- **Adjusted R² ≈ 0.798 (~80%)** on the test set — the model explains roughly 80% of the variance in insurance charges after accounting for the number of features used

---

## 🚗 Project 2: Ford Car Price Prediction

**Notebook:** `notebook9fdd2dc0b8.ipynb`
**Dataset:** Ford Car Price Prediction dataset (Kaggle: `ford-car-price-prediction/ford.csv`)
**Goal:** Predict used-car `price` from vehicle specifications.

### Dataset Features
`model`, `year`, `price` (target), `transmission`, `mileage`, `fuelType`, `tax`, `mpg`, `engineSize`

### Workflow

**1. EDA**
- Shape, info, describe, missing-value check
- Distribution of `price` (histogram + KDE)
- Correlation heatmap of numeric features
- Boxplots: `price` vs `year`, `price` vs `engineSize`, `price` vs `transmission`, `price` vs `fuelType`, `price` vs `model`
- Scatter plot: `mileage` vs `price` (to visually assess depreciation trend)

**2. Encoding Strategy — Compared Two Approaches**
This project deliberately compares two categorical-encoding strategies to see which produces a better linear model:

- **One-Hot Encoding** (`pd.get_dummies` on `model`, `transmission`, `fuelType`, `drop_first=True`) — cast to `int`
- **Label Encoding** (`sklearn.preprocessing.LabelEncoder` on the same three columns) — an ordinal alternative

**3. Feature Scaling**
- `StandardScaler` applied to numeric columns (`year`, `mileage`, `tax`, `mpg`, `engineSize`) for the one-hot encoded set
- Scaler applied to the full label-encoded feature set as well

**4. Model Training — Two Models for Comparison**

| Model | Encoding | Split |
|---|---|---|
| `model` | One-Hot Encoded | 67/33 train-test |
| `model2` | Label Encoded | 67/33 train-test |

Both trained with `sklearn.linear_model.LinearRegression`.

**5. Results**

| Model | R² Score | Adjusted R² |
|---|---|---|
| **One-Hot Encoded** | **0.8397** | **0.8387** |
| **Label Encoded** | 0.7310 | — |

**Key takeaway:** One-hot encoding significantly outperformed label encoding for this dataset. This makes sense theoretically — label encoding imposes an artificial ordinal relationship on categorical variables like `model` and `fuelType` that have no true rank order, which misleads a linear model. One-hot encoding avoids that false assumption and better captures the true categorical structure, resulting in a noticeably stronger fit (~84% vs ~73% variance explained).

---

## 🛠 Tech Stack

- **Language:** Python 3
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Statistics:** SciPy (`pearsonr`, `chi2_contingency`)
- **Machine Learning:** scikit-learn
  - `LinearRegression`
  - `StandardScaler`, `LabelEncoder`
  - `train_test_split`
  - `r2_score`, `mean_absolute_error`, `mean_squared_error`

---

## 🎯 Key Learnings

- Building the cost function and gradient descent by hand (rather than only calling `.fit()`) made it clear *why* Linear Regression converges to the best-fit line, not just *that* it does.
- Adjusted R² is essential when comparing models with different feature counts — it exposed that a feature boosting raw R² was actually hurting the Insurance model.
- Feature selection isn't purely intuition-driven — combining **Pearson correlation** (for numeric relationships) with a **Chi-square test** (for categorical independence) gave a more statistically grounded feature set for the Insurance project.
- Encoding choice matters a lot for linear models specifically: the Ford project's side-by-side comparison showed a real, measurable accuracy gap (~11 percentage points of R²) purely from switching Label Encoding → One-Hot Encoding.
- Overfitting/underfitting isn't just theoretical — comparing train vs. test accuracy side by side (as in the M1/M2/M3 example) is a fast, practical diagnostic.

---

## 🚀 Future Improvements

- Apply **Ridge/Lasso Regression** (regularization) to both projects and compare against plain Linear Regression, especially for the Ford dataset which has many one-hot encoded `model` columns.
- Try non-linear models (Random Forest, Gradient Boosting/XGBoost) to benchmark against the linear baseline, since car price and insurance cost relationships are unlikely to be perfectly linear.
- Perform hyperparameter tuning (e.g. `GridSearchCV`) once regularized/non-linear models are introduced.
- Add cross-validation instead of a single train-test split for more robust performance estimates.
- Deploy the better-performing model (e.g. via Streamlit or Flask) as an interactive price/cost estimator.