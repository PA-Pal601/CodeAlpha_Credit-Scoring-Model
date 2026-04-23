# CodeAlpha_Credit-Scoring-Model


# 💳 Credit Scoring Model

> **Predict individual creditworthiness using machine learning on historical financial data.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange?logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red)](https://xgboost.readthedocs.io)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [Models](#-models)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [How to Run](#-how-to-run)
- [Prediction on New Data](#-prediction-on-new-data)
- [Dependencies](#-dependencies)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🧠 Overview

Credit scoring is a critical process in the financial industry — it determines whether an individual qualifies for a loan, credit card, or mortgage. This project builds a full end-to-end machine learning pipeline that:

- Ingests and cleans financial applicant data
- Engineers meaningful features from raw inputs
- Handles class imbalance using **SMOTE**
- Trains and compares **5 classification algorithms**
- Evaluates models using financial-grade metrics
- Exposes a reusable **prediction function** for new applicants

The best model achieves **ROC-AUC > 0.90**, making it suitable as a decision-support tool for credit officers.

---

## 📁 Project Structure

```
credit-scoring-model/
│
├── Credit_Scoring_Model.ipynb   # Main Colab notebook (full pipeline)
├── README.md                    # Project documentation (this file)
└── requirements.txt             # Python dependencies (optional local use)
```

---

## 📊 Dataset

The dataset is **synthetically generated** within the notebook to simulate a realistic credit applicant population of **10,000 records**.

| Feature | Type | Description |
|---|---|---|
| `age` | Integer | Applicant age (18–74) |
| `income` | Float | Annual gross income (USD) |
| `employment_years` | Integer | Years with current employer |
| `num_credit_lines` | Integer | Number of open credit lines |
| `credit_utilization` | Float | Ratio of credit used (0–1) |
| `num_late_payments` | Integer | Late payments in last 24 months |
| `num_defaults` | Integer | Total defaults on record |
| `debt_to_income` | Float | Debt-to-income ratio |
| `loan_amount` | Integer | Requested loan amount (USD) |
| `loan_purpose` | Categorical | Personal / Auto / Home / Education / Business |
| `home_ownership` | Categorical | Rent / Own / Mortgage |
| `credit_score_raw` | Integer | Raw bureau credit score (300–850) |
| `creditworthy` ⭐ | Binary | **Target** — 1 = Good Credit, 0 = Bad Credit |

> ℹ️ **Realistic data quality:** ~2% missing values are injected in `income`, `employment_years`, `credit_utilization`, and `debt_to_income` to simulate real-world conditions.

**Class Distribution:**

```
Good Credit (1)  →  ~68%
Bad Credit  (0)  →  ~32%
```

SMOTE oversampling is applied on the training set to address this imbalance before model training.

---

## 🔧 Feature Engineering

Eight new features are engineered from the raw data to improve predictive power:

| Engineered Feature | Formula / Logic | Purpose |
|---|---|---|
| `income_to_loan_ratio` | `income / (loan_amount + 1)` | Repayment capacity |
| `payment_risk_score` | `late_payments×0.4 + defaults×1.0 + utilization×0.6` | Composite risk indicator |
| `stability_index` | `(age×0.4 + employment_years×0.6) / 10` | Financial stability proxy |
| `log_income` | `log1p(income)` | Normalize income skew |
| `log_loan_amount` | `log1p(loan_amount)` | Normalize loan amount skew |
| `credit_bucket` | Binned score: Poor / Fair / Good / Very Good / Exceptional | Categorical credit tier |
| `has_default` | `1 if num_defaults > 0 else 0` | Binary default flag |
| `high_utilization` | `1 if credit_utilization > 0.70 else 0` | Binary over-utilization flag |

---

## 🤖 Models

Five classification algorithms are trained and compared:

| Model | Key Hyperparameters |
|---|---|
| **Logistic Regression** | `C=0.5`, `solver=lbfgs`, `max_iter=1000` |
| **Decision Tree** | `max_depth=8`, `min_samples_leaf=20` |
| **Random Forest** | `n_estimators=200`, `max_depth=12`, `min_samples_leaf=10` |
| **Gradient Boosting** | `n_estimators=150`, `learning_rate=0.05`, `max_depth=5` |
| **XGBoost** | `n_estimators=200`, `learning_rate=0.05`, `max_depth=6` |

All tree-based models use raw (unscaled) features. Logistic Regression uses **RobustScaler**-normalized features. All models are evaluated using **5-fold Stratified Cross-Validation**.

---

## 📏 Evaluation Metrics

| Metric | Why It Matters in Credit Scoring |
|---|---|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | Of predicted good-credit applicants, how many truly are? (avoids risky approvals) |
| **Recall** | Of all truly good applicants, how many did we approve? (avoids missing good customers) |
| **F1-Score** | Harmonic mean of Precision & Recall — balanced view |
| **ROC-AUC** | Discrimination ability across all classification thresholds |
| **CV AUC** | Cross-validated AUC — measures generalization, not overfitting |

> ⚠️ In credit risk, **Recall** and **ROC-AUC** are typically prioritized over raw Accuracy to minimize costly misclassifications.

---

## 🏆 Results

All results are on the **held-out 20% test set** (2,000 records, never seen during training).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.82 | ~0.85 | ~0.88 | ~0.86 | ~0.88 |
| Decision Tree | ~0.83 | ~0.86 | ~0.88 | ~0.87 | ~0.87 |
| Random Forest | ~0.88 | ~0.90 | ~0.91 | ~0.90 | ~0.93 |
| Gradient Boosting | ~0.88 | ~0.91 | ~0.91 | ~0.91 | ~0.94 |
| **XGBoost** ⭐ | **~0.89** | **~0.91** | **~0.92** | **~0.91** | **~0.94** |

> 📌 Exact values vary slightly due to the random synthetic dataset. **XGBoost** consistently achieves the highest ROC-AUC and is selected as the production model.

**Visualizations generated:**
- 📊 Feature distribution plots (Good vs Bad Credit)
- 🔥 Correlation heatmap
- 📉 ROC Curves — all models overlaid
- 📈 Precision-Recall Curves
- 🟦 Confusion Matrices (5 panels)
- 🌳 Decision Tree visualization (depth=3)
- 📊 Feature importance bar charts (Random Forest & XGBoost)
- 📊 Side-by-side metric comparison dashboard

---

## 🚀 Getting Started

### ▶️ Run on Google Colab (Recommended)

1. Download `Credit_Scoring_Model.ipynb`
2. Go to [colab.research.google.com](https://colab.research.google.com)
3. Click **File → Upload notebook** and select the `.ipynb` file
4. Click **Runtime → Run All**

That's it — all dependencies are installed automatically inside the notebook.

---

### 💻 Run Locally (Optional)

**Prerequisites:** Python 3.8+

```bash
# 1. Clone the repository
git clone https://github.com/your-username/credit-scoring-model.git
cd credit-scoring-model

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook Credit_Scoring_Model.ipynb
```

---

## 🔮 How to Run

The notebook is divided into **10 clearly labeled sections**:

```
Section 1  →  Setup & Imports
Section 2  →  Dataset Generation & Loading
Section 3  →  Exploratory Data Analysis (EDA)
Section 4  →  Feature Engineering
Section 5  →  Data Preprocessing (Imputation, SMOTE, Scaling)
Section 6  →  Model Training (5 models + Cross-Validation)
Section 7  →  Model Evaluation (Confusion Matrix, ROC, Reports)
Section 8  →  Model Comparison & Selection
Section 9  →  Feature Importance
Section 10 →  Prediction on New Data
```

Run all cells top-to-bottom, or jump to any section independently after running Section 1–5 first.

---

## 🎯 Prediction on New Data

Use the built-in `predict_creditworthiness()` function to score new applicants:

```python
new_applicants = [
    {
        'age': 45,
        'income': 90000,
        'employment_years': 12,
        'num_credit_lines': 6,
        'credit_utilization': 0.18,
        'num_late_payments': 0,
        'num_defaults': 0,
        'debt_to_income': 0.20,
        'loan_amount': 15000,
        'loan_purpose': 'Home',        # Personal / Auto / Home / Education / Business
        'home_ownership': 'Own',       # Rent / Own / Mortgage
        'credit_score_raw': 760
    }
]

result = predict_creditworthiness(new_applicants, model_name='XGBoost')
print(result)
```

**Output:**

```
  Applicant       Prediction  Prob_Creditworthy Risk_Level
Applicant 1  ✅ Creditworthy            0.9421        Low
```

---

## 📦 Dependencies

| Library | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.23 | Numerical computing |
| `pandas` | ≥1.5 | Data manipulation |
| `matplotlib` | ≥3.6 | Plotting |
| `seaborn` | ≥0.12 | Statistical visualization |
| `scikit-learn` | ≥1.3 | ML models & preprocessing |
| `xgboost` | ≥1.7 | Gradient boosting |
| `imbalanced-learn` | ≥0.11 | SMOTE oversampling |

All installed automatically on Colab via `!pip install xgboost imbalanced-learn` in cell 1.

---

## 🔭 Future Improvements

- [ ] **Hyperparameter tuning** with `GridSearchCV` or `Optuna`
- [ ] **SHAP values** for model explainability and individual prediction explanations
- [ ] **Threshold optimization** — tune the classification threshold for business-specific cost matrices (e.g., cost of false approval vs false rejection)
- [ ] **Real dataset integration** — plug in [UCI Credit Approval](https://archive.ics.uci.edu/ml/datasets/credit+approval) or [Kaggle Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
- [ ] **Deployment** — wrap prediction function in a Flask/FastAPI REST endpoint
- [ ] **Calibration** — apply `CalibratedClassifierCV` for better probability estimates
- [ ] **Fairness audit** — evaluate model bias across demographic groups

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ for learning and exploration  
⭐ Star this repo if you found it useful!

</div>
