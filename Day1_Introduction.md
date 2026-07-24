# Machine Learning — Notes

---

## 1. What is Machine Learning?

### Definition
**Machine Learning** is a part of Artificial Intelligence where a system **learns patterns from data** instead of being given fixed rules — and uses those patterns to make predictions or decisions on new data.

### Traditional Programming vs. Machine Learning

**Traditional Programming:**
```
Rules + Data  →  [Program]  →  Output
```
You write explicit rules yourself.

**Machine Learning:**
```
Data + Output  →  [Model]  →  Rules (learned automatically)
```
You give the data and the known outcomes, and the model figures out the pattern itself.

### Real-World Examples
- **Gmail spam filter:** learns from millions of emails marked "spam" vs "not spam" and predicts it for new emails.
- **Netflix recommendations:** learns your watching habits and suggests shows you're likely to enjoy.
- **Bank fraud detection:** learns normal spending patterns and flags unusual transactions.

### Key Terms
- **Model:** A trained system that can make predictions.
- **Training:** Showing the model data so it learns patterns.
- **Algorithm:** The method used to learn from data (e.g., Linear Regression, Decision Tree).
- **Dataset:** The data used to train and test a model.
- **Feature:** An input used to make a prediction (e.g., age, size).
- **Label/Target:** The thing you're trying to predict (e.g., price, spam or not).

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

### Two Main Problem Types
**Regression** — predicting a **number**
**Classification** — predicting a **category**

### Real-World Examples
- **Regression:** predicting house prices, predicting tomorrow's temperature, predicting salary based on experience.
- **Classification:** email spam detection, identifying whether a tumor is malignant or benign, detecting whether a bank transaction is fraudulent.

### Common Algorithms
- Linear Regression, Logistic Regression, Decision Trees, Random Forest, SVM, KNN

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
The model works with **unlabeled data** — there's no correct answer given. It finds hidden structure or groupings on its own.

### Two Main Problem Types
**Clustering** — grouping similar data points
**Dimensionality Reduction** — reducing the number of features while keeping the important information

### Real-World Examples
- **Customer segmentation:** grouping shoppers by buying habits for targeted marketing.
- **News grouping:** grouping articles about the same topic together automatically.
- **Anomaly/fraud detection:** finding unusual transactions that don't fit any normal pattern.

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

## 9. Scikit-learn

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
| Supervised Learning | Learn from labeled data → regression (numbers) or classification (categories) |
| Unsupervised Learning | Find hidden structure in unlabeled data → clustering or dimensionality reduction |
| Semi-supervised Learning | Combine small labeled + large unlabeled data |
| Reinforcement Learning | Agent learns via trial, error, and reward maximization |
| Data | Quality and quantity of data matter more than the algorithm itself |
| Scikit-learn | The main Python library to implement all of the above practically |