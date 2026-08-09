# Machine Learning — Notes

---

## 1. What is Machine Learning?

### Definition
**Machine Learning** is a part of Artificial Intelligence where a system **learns patterns from data** instead of being given fixed rules — and uses those patterns to make predictions or decisions on new data.

In short: instead of programming every step, we give machines lots of data and let them figure out patterns on their own.

### Traditional Programming vs. Machine Learning

**Traditional Programming:**
```
Rules + Data  →  [Program]  →  Output
```
Give rules + data → get a result. Manual logic writing.

**Machine Learning:**
```
Data + Output  →  [Model]  →  Rules (learned automatically)
```
Give data + result → get rules (the model). It learns patterns automatically.

### Real-World Examples
- **Gmail spam filter:** learns from millions of emails marked "spam" vs "not spam" and predicts it for new emails.
- **Netflix recommendations:** learns your watching habits and suggests shows you're likely to enjoy.
- **YouTube recommendations:** learns from your watch history.
- **Voice assistants (Siri, Alexa):** learn how you speak.
- **Self-driving cars:** learn to identify stop signs, pedestrians, and roads.
- **Face unlock on phones:** learns to recognize your face.
- **Bank fraud detection:** learns normal spending patterns and flags unusual transactions.

Today, Machine Learning powers everything from the apps on your phone to the systems behind hospitals, banks, and even space research.

### Key Terms
- **Model:** A trained system that can make predictions.
- **Training:** Showing the model data so it learns patterns.
- **Algorithm:** The method used to learn from data (e.g., Linear Regression, Decision Tree).
- **Dataset:** The data used to train and test a model.
- **Feature:** An input used to make a prediction (e.g., age, size).
- **Label/Target:** The thing you're trying to predict (e.g., price, spam or not).

---

## 1a. AI vs ML vs DL

People often use the terms AI, Machine Learning, and Deep Learning as if they mean the same thing — but they're not. Here's the distinction:

- **AI (Artificial Intelligence)** is the **big umbrella** — the science of making machines *smart*, just like humans.
  Examples: playing chess like a human, talking to Alexa, driving a car on its own.
  → AI = any system that mimics human intelligence.

- **ML (Machine Learning)** is a **subset of AI** — this is where machines learn from **data** and improve over time.
  Examples: YouTube recommending videos, Netflix predicting your next binge, Gmail filtering spam.

- **DL (Deep Learning)** is a **subset of Machine Learning** — it uses **neural networks**, inspired by the human brain, to handle big, complex data (like images or speech).
  Examples: face recognition on phones, ChatGPT, self-driving car vision.

So the hierarchy is: **AI ⊃ ML ⊃ DL**

---

## 2. Types of Machine Learning

| Type | Data Used | Goal |
|---|---|---|
| **Supervised Learning** | Labeled data (input + correct answer) | Predict outputs for new inputs |
| **Unsupervised Learning** | Unlabeled data (input only) | Find hidden patterns/groups |
| **Semi-supervised Learning** | A little labeled + a lot of unlabeled data | Learn well with limited labels |
| **Reinforcement Learning** | No fixed dataset — learns from feedback | Maximize long-term reward |

### Simple Analogy
- **Supervised:** A teacher shows you fruits and tells you their names.
- **Unsupervised:** You get a basket of unnamed fruits and group similar ones yourself.
- **Semi-supervised:** A few fruits are named for you; you use that to figure out the rest.
- **Reinforcement:** You taste different fruits and learn which ones you like from the outcome.

### Key Terms
- **Labeled data:** Data with the correct answer attached.
- **Unlabeled data:** Data with no attached answer.
- **Agent (RL):** The learner/decision-maker.
- **Reward (RL):** Feedback showing how good/bad an action was.

---

## 3. Supervised Machine Learning

### Definition
The model learns from **labeled data** — every example has an input and the correct output. It learns the mapping input → output, so it can predict on new data.

Supervised Learning is like a student being taught by a teacher: we give the machine input and the correct answer, and it learns to predict.

### Two Main Problem Types
**Regression** — predicting a **number**
**Classification** — predicting a **category**

### Real-World Examples
- **Regression:** predicting house prices, predicting tomorrow's temperature, predicting salary based on experience.
- **Classification:** email spam detection, identifying whether a tumor is malignant or benign, detecting whether a bank transaction is fraudulent.

### Example with Data
| Income($) | Credit Score | Loan |
|---|---|---|
| 40,000 | 750 | Yes |
| 25,000 | 600 | No |
| 50,000 | 800 | Yes |
| 30,000 | 580 | No |

Here, Income and Credit Score are input variables (features) and Loan is the output variable (label) we're trying to predict. We train the model with both input and output variables so it learns the mapping.

### Common Algorithms
- Linear Regression, Logistic Regression, Decision Tree, Naive Bayes, Support Vector Machine, K-Nearest Neighbors, Gradient Boosting

### Key Terms
- **Feature (X):** Input variable(s).
- **Label/Target (y):** The known correct output.
- **Training set:** Data used to teach the model.
- **Test set:** Unseen data used to check performance.
- **Overfitting:** Model learns training data too well, fails on new data.
- **Underfitting:** Model is too simple, performs poorly even on training data.

---

## 4. Unsupervised Machine Learning

### Definition
The model works with **unlabeled data** — there's no correct answer given. We're not going to predict anything; we'll just find patterns, group similar items, reduce complexity, and spot outliers.

### Two Main Problem Types
**Clustering** — grouping similar data points
**Dimensionality Reduction** — reducing the number of features while keeping the important information

### Real-World Examples
- **Customer segmentation:** grouping shoppers by buying habits for targeted marketing.
- **News grouping:** grouping articles about the same topic together automatically.
- **Anomaly/fraud detection:** finding unusual transactions that don't fit any normal pattern.
- **Google search:** clustering/organizing results based on patterns in the data.

Note: unsupervised learning is often used together with supervised learning to build models.

### Common Algorithms
- K-Means Clustering, Hierarchical Clustering, DBSCAN, PCA

### Key Terms
- **Cluster:** A group of similar data points.
- **Centroid:** The center point of a cluster (used in K-Means).
- **Dimensionality:** The number of features/columns in a dataset.
- **Anomaly detection:** Finding data points that don't fit any pattern.

---

## 5. Semi-supervised Machine Learning

### Definition
Uses a **small amount of labeled data** plus a **large amount of unlabeled data** to build a better model than either alone. Useful because labeling data is often expensive and slow.

### Real-World Examples
- **Medical imaging:** a doctor labels a few hundred X-rays as "healthy"/"diseased," and the model uses those plus thousands of unlabeled X-rays to learn better.
- **Customer support tickets:** only a small batch of tickets are manually categorized, but the model learns to categorize the rest.

### Key Terms
- **Pseudo-labeling:** Using the model's own confident predictions on unlabeled data as if they were true labels.
- **Label scarcity:** Having lots of raw data but very few labeled examples.

---

## 6. Reinforcement Learning

### Definition
An **agent** learns to make decisions by interacting with an **environment**, getting **rewards or penalties**, and adjusting its actions to **maximize reward over time.** No fixed dataset — it learns through trial and error.

Reinforcement Learning is like training a dog. You don't teach it everything directly — instead, you reward good behavior and ignore or punish bad actions. Over time, the dog learns what actions give it treats. That's the core idea of RL — learning by **trial and error**.

### Core Components
| Component | Meaning |
|---|---|
| **Agent** | The learner (e.g., a robot, a game AI) |
| **Environment** | The world it interacts with |
| **State** | Current situation of the agent |
| **Action** | A choice the agent makes |
| **Reward** | Feedback after an action |
| **Policy** | The strategy for choosing actions |

### Real-World Examples
- **Self-driving cars:** learning to steer, brake, and change lanes safely.
- **Game-playing AI:** AlphaGo learning to play Go better than humans through millions of games.
- **Robotics:** a robot learning to walk by trial and error, getting "rewarded" for staying balanced.

### Key Terms
- **Episode:** One complete run/attempt (e.g., one full game).
- **Exploration vs. Exploitation:** Trying new actions vs. sticking with known good ones.
- **Cumulative reward:** Total reward collected over time — what the agent tries to maximize.

---

## 7. Applications of Machine Learning

| Domain | Real-World Examples |
|---|---|
| **Healthcare** | Disease diagnosis, medical image analysis, drug discovery |
| **Finance** | Fraud detection, credit scoring, algorithmic trading |
| **E-commerce/Retail** | Product recommendations, demand forecasting, dynamic pricing |
| **Transportation** | Self-driving cars, ride-hailing price prediction |
| **Marketing** | Customer segmentation, churn prediction, targeted ads |
| **NLP** | Chatbots, translation (Google Translate), spam filtering |
| **Computer Vision** | Face unlock on phones, defect detection in factories |
| **Entertainment** | Netflix/YouTube/Spotify recommendations |

---

## 8. Why Data is Important for ML

### Core Idea
**"Garbage in, garbage out."** A model is only as good as the data it learns from. No algorithm can fix bad data.

### What Makes Data Good
- **Quantity:** enough examples to learn general patterns
- **Quality:** accurate, clean, few errors
- **Relevance:** features actually related to what you're predicting
- **Balance:** fair representation of all categories
- **Consistency:** collected/formatted uniformly

### Common Data Problems
- **Missing values:** empty fields
- **Noisy data:** errors or irrelevant info mixed in
- **Imbalanced data:** one class massively outnumbers another (e.g., 99% "not fraud," 1% "fraud")
- **Bias in data:** unfair historical patterns get learned and repeated by the model

### Real-World Example
A loan approval model trained mostly on data from one city may perform poorly or unfairly for applicants from other regions — because the data wasn't representative.

### Key Terms
- **Data preprocessing:** Cleaning/preparing raw data before training.
- **Data cleaning:** Fixing or removing incorrect/irrelevant data.
- **Feature engineering:** Creating more useful features from raw data.
- **Train/Test split:** Splitting data into a training portion and a testing portion.

---

## 9. Exploratory Data Analysis (EDA)

### Definition
EDA stands for **Exploratory Data Analysis** — it's the **process of analyzing, visualizing, and understanding your data** before you build any machine learning model.

Think of EDA as detective work — you're looking at your data with curiosity before building any models or doing fancy transformations.

EDA is the step where you explore the data to: understand it, discover patterns, spot anomalies, generate insights, and decide what to do next.

### Full ML Model-Building Pipeline
1. Problem definition
2. Data collection
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing / Cleaning
5. Feature Selection & Engineering
6. Split the Dataset
7. Model Selection
8. Model Training
9. Model Evaluation
10. Hyperparameter Tuning
11. Model Testing / Validation

### EDA Steps
1) **Viewing the Data** — `head()`, `tail()`, `shape`, `info()`. What columns do I have? What types of data?
2) **Summary Statistics** — mean, median, mode, std, min, max, quartiles. Helps understand spread and central tendency.
3) **Value Counts** — how many unique values in a column? Great for categorical columns.
4) **Missing Value Analysis** — where are the gaps? What percent of the data is missing?
5) **Visualizations:**
   - Histograms → distribution of values
   - Boxplots → outliers and spread
   - Bar plots → comparisons of categories
   - Correlation heatmaps → linear relationships between numerical features
   - Scatter plots → bivariate relationships
6) **Target Variable Exploration** — how does your output (e.g., 'charges' in your dataset) relate to other variables?

### Why EDA Matters
- You can't clean or preprocess what you don't understand.
- It helps you identify mistakes, biases, or limitations in the data.
- It guides the direction of your data cleaning and feature engineering.
- It gives your audience or stakeholders a "story" or overview of what the data is telling you.

---

## 10. Data Cleaning

EDA tells you what's wrong. Data Cleaning fixes it. Cleaning is not glamorous, but it's 80% of the work in real-world projects.

1. **Handle Missing Values** — check which columns have missing values (nulls).
   - Strategies: drop missing rows/columns (only if very few); impute with mean/median (numerical), mode (categorical), or advanced methods like linear regression, KNN, or interpolation (for future learning).
2. **Remove Duplicates** — detect and drop exact duplicate rows.
3. **Fix Data Types** — convert wrong types (e.g., numbers stored as strings, dates as text).
4. **Handle Inconsistent Categories** — clean up categorical values like "Male", "male", "MALE" → all become "male"; "Yes", "yes", "Y" → unify to one format.
5. **Detect and Handle Outliers** — use boxplots, IQR, or Z-score. Handle by removing (if clearly wrong) or capping (e.g., to the 95th percentile).
6. **Fix Logic or Domain Errors** — e.g., age = -5 is invalid, or BMI = 150 is likely an error. Can replace with mean, median, or remove.

---

## 11. Data Preprocessing

If Data Cleaning is about fixing mistakes, Data Preprocessing is about transforming valid data into a usable format so it can be analyzed or used in a machine learning model.

1. **Encoding Categorical Variables** — convert text labels (like "male", "yes", "southeast") into numbers.
   - **Label Encoding (Ordinal):** good for ordered categories like "Low", "Medium", "High"
   - **One-Hot Encoding (Nominal):** for non-ordered categories like region
2. **Feature Transformation** (Log, Square root, etc.) — used to handle skewed data, like right-skewed or left-skewed data.
3. **Feature Scaling** (Normalization or Standardization) — bring numerical values to the same scale, especially useful for distance-based algorithms.
   - **Normalization (Min-Max Scaling):** scales values between 0 and 1
   - **Standardization (Z-score Scaling):** transforms data to have mean 0 and std 1

---

## 12. Feature Engineering

Creating new features or transforming existing ones to expose useful patterns that ML models can learn from.

**Why do we need it?** Because ML models don't know domain logic — we have to give them the right signals.

### Common Feature Engineering Techniques
- Mathematical Combinations
- Target-Based Flags
- Binning (when it helps)
- Time-Based Features (if time exists)

---

## 13. Feature Selection (for ML)

Selecting the most useful features and removing the rest.

**Why is it important?**
- Reduces noise and overfitting
- Speeds up training
- Improves model accuracy
- Makes model interpretation easier

### Methods
1. **Filter Methods (Pure Statistics)**
   - Correlation Matrix → remove highly correlated features
   - Chi-square test (categorical vs categorical)
   - ANOVA F-test (numerical vs categorical target)
2. **Wrapper Methods** — iteratively add/remove features and evaluate model performance
3. **Embedded Methods (Selection built into the model)**
   - Lasso Regression → shrinks coefficients to 0
   - Tree-based models (Random Forest, XGBoost) → feature importance scores

---

## 14. Scikit-learn

### What is it?
**Scikit-learn** (`sklearn`) is a free Python library with ready-made tools for machine learning — classification, regression, clustering, dimensionality reduction, preprocessing, and model evaluation.

### Why It's Popular
- Simple, consistent API across almost all algorithms
- Built on NumPy, SciPy, and matplotlib
- Great documentation and community support

### Typical Workflow
```
1. Import the model         → from sklearn.linear_model import LinearRegression
2. Prepare data (X, y)      → split into features and labels
3. Train/test split         → from sklearn.model_selection import train_test_split
4. Create model instance    → model = LinearRegression()
5. Train the model          → model.fit(X_train, y_train)
6. Make predictions          → model.predict(X_test)
7. Evaluate performance      → accuracy_score, mean_squared_error, etc.
```

### Key Terms
- **`fit()`:** Trains a model on data.
- **`predict()`:** Generates predictions from a trained model.
- **`transform()`:** Modifies data during preprocessing (e.g., scaling).
- **Estimator:** Any sklearn object that learns from data via `.fit()`.
- **Pipeline:** Chains preprocessing steps and a model into one object.

---

## Summary Table

| Concept | One-line takeaway |
|---|---|
| Machine Learning | Systems that learn patterns from data instead of following fixed rules |
| AI vs ML vs DL | AI is the umbrella, ML is a subset of AI, DL is a subset of ML using neural networks |
| Supervised Learning | Learn from labeled data → regression (numbers) or classification (categories) |
| Unsupervised Learning | Find hidden structure in unlabeled data → clustering or dimensionality reduction |
| Semi-supervised Learning | Combine small labeled + large unlabeled data |
| Reinforcement Learning | Agent learns via trial, error, and reward maximization |
| Data | Quality and quantity of data matter more than the algorithm itself |
| EDA | Explore and understand data before modeling — detective work |
| Data Cleaning | Fixing mistakes: missing values, duplicates, wrong types, inconsistent categories, outliers |
| Data Preprocessing | Transforming valid data into a usable format: encoding, transformation, scaling |
| Feature Engineering | Creating new features to expose useful patterns |
| Feature Selection | Picking the most useful features, dropping the rest |
| Scikit-learn | The main Python library to implement all of the above practically |

---

## ⭐ Support This Repository

If you found this notebook useful, consider:

⭐ **Starring this repository** to support the project  
🍴 **Forking it** if you want to experiment with the code  
💬 **Sharing your feedback** or suggestions  
🔗 **Following me on LinkedIn** for more Machine Learning, AI, Data Science, and Python content

👉 **LinkedIn:** [Hamza Anjum](https://www.linkedin.com/in/hamza-anjum-459bba320/)

If you're also learning Machine Learning, feel free to explore the other notebooks in this repository.

**Keep learning. Keep building. 🚀**