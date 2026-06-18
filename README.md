# Improved Detection of Fraud Cases for E-commerce and Bank Transactions

## Project Overview

This project develops machine learning models to detect fraudulent transactions in **e-commerce** and **credit card** datasets. The objective is to accurately identify fraudulent activities while minimizing false positives through robust data preprocessing, feature engineering, ensemble learning, hyperparameter tuning, and explainable AI techniques.

The project follows an end-to-end machine learning workflow, from raw data preparation to model interpretation using SHAP.

---

## Objectives

- Perform exploratory data analysis (EDA) on fraud datasets.
- Engineer behavioral and temporal features.
- Handle class imbalance using SMOTE.
- Train and compare multiple machine learning models.
- Optimize selected models through hyperparameter tuning.
- Evaluate models using fraud-sensitive metrics.
- Explain model predictions using SHAP.
- Generate business recommendations based on explainability insights.

---

## Dataset Description

The project uses three datasets:

### 1. Fraud Transaction Dataset
Contains e-commerce transaction information including:

- User demographics
- Purchase information
- Browser and device information
- Traffic source
- IP address
- Purchase and signup timestamps
- Fraud label

### 2. Credit Card Fraud Dataset

Contains:

- Time
- Amount
- PCA-transformed features (V1–V28)
- Fraud class

### 3. IP Address to Country Dataset

Maps IP address ranges to countries for geolocation enrichment.

---

# Project Structure

```
fraud_detection/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── eda.ipynb
│   ├── model_training.ipynb
│   └── model_explainability.ipynb
│
├── scripts/
│
├── src/
│   ├── data_processing.py
│   ├── data_preparation.py
│   ├── train.py
│   ├── tune.py
│   ├── validate.py
│   ├── explainability.py
│   └── plots.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Workflow

The project follows the pipeline below:

```
Raw Data
    │
    ▼
Data Cleaning
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Feature Engineering
    │
    ▼
Data Transformation
    │
    ▼
SMOTE (Training Set Only)
    │
    ▼
Baseline Model (Logistic Regression)
    │
    ▼
Ensemble Models
(Random Forest, XGBoost, LightGBM)
    │
    ▼
Hyperparameter Tuning
    │
    ▼
Cross Validation
    │
    ▼
Model Explainability (SHAP)
    │
    ▼
Business Recommendations
```

---

# Feature Engineering

The following features were engineered for the Fraud Transaction dataset:

- Hour of transaction
- Day of week
- Time since signup
- Transaction frequency
- Transaction velocity
- Country (IP mapping)

The Credit Card dataset was standardized using its numerical features.

---

# Models Evaluated

### Baseline

- Logistic Regression

### Ensemble Models

- Random Forest
- XGBoost
- LightGBM

---

# Selected Models

| Dataset | Selected Model |
|----------|----------------|
| Fraud Transaction | LightGBM |
| Credit Card | Random Forest |

Model selection was based on:

- AUC-PR
- F1-Score
- Precision
- Recall

---

# Evaluation Metrics

The following metrics were used to evaluate model performance:

- AUC-PR
- F1-Score
- Precision
- Recall
- Confusion Matrix

Cross-validation was performed using **5-fold Stratified K-Fold**.

---

# Model Explainability

Model predictions were interpreted using **SHAP (SHapley Additive exPlanations)**.

The explainability analysis includes:

- Feature Importance
- SHAP Summary Plots
- SHAP Force Plots
- Comparison between built-in feature importance and SHAP
- Business recommendations derived from SHAP insights

---

# Key Findings

- LightGBM achieved the best performance on the Fraud Transaction dataset.
- Random Forest achieved the best performance on the Credit Card dataset.
- Hyperparameter tuning improved the Random Forest model while confirming the robustness of LightGBM.
- Time since signup was the strongest predictor of fraud in the Fraud Transaction dataset.
- V14, V12, V4, V3, and V10 were the most influential features in the Credit Card dataset.
- SHAP provided transparent explanations for both global and individual model predictions.

---

# Business Recommendations

Based on SHAP analysis:

- Apply additional verification for transactions made shortly after account creation.
- Increase monitoring of high-value transactions from newly created accounts.
- Incorporate browser type and traffic source into fraud risk scoring.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/fraud_detection.git
cd fraud_detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

### Exploratory Data Analysis

```bash
jupyter notebook notebooks/eda.ipynb
```

### Model Training

```bash
jupyter notebook notebooks/model_training.ipynb
```

### Model Explainability

```bash
jupyter notebook notebooks/model_explainability.ipynb
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

---

# Future Improvements

- Explore additional ensemble methods.
- Incorporate more behavioral and device-related features.
- Deploy the selected models for real-time fraud detection.
- Continuously retrain models using newly collected transaction data.
- Improve fraud detection rules using SHAP insights.

---

# Author

**Bemnet Bekele**
