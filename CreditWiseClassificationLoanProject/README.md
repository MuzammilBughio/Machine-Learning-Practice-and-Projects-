# 💳 CreditWise Loan System

### Machine Learning Based Loan Approval Prediction

CreditWise is a machine learning classification project designed to predict whether a loan application should be **Approved or Rejected** based on an applicant's financial, personal, employment, and loan-related information.

The project simulates a real-world banking scenario where machine learning can assist loan officers by providing fast and consistent preliminary decisions.

> **Note:** This project is intended for educational and machine-learning practice purposes. It should not be used as a standalone system for real-world financial decisions.

---

## 🎯 Problem Statement

Traditional loan verification can be time-consuming and may produce inconsistent decisions when applications are manually evaluated.

CreditWise aims to build a machine learning system that learns patterns from historical loan applications and predicts:

* ✅ **Loan Approved**
* ❌ **Loan Rejected**

The objective is to create a model that can help identify applicants who are more likely to receive loan approval before the final human verification process.

---

## 📊 Dataset

Each row represents a loan applicant with information related to their financial and personal profile.

Some of the features used in the project include:

* Applicant Income
* Coapplicant Income
* Age
* Dependents
* Credit Score
* Existing Loans
* DTI Ratio
* Savings
* Collateral Value
* Loan Amount
* Loan Term
* Education Level
* Employment Status
* Marital Status
* Loan Purpose
* Property Area
* Gender
* Employer Category

### Target Variable

`Loan_Approved`

The target represents whether the applicant's loan was approved or rejected.

---

## 🔄 Machine Learning Workflow

```text
Raw Loan Data
      ↓
Data Loading
      ↓
Missing Value Handling
      ↓
Exploratory Data Analysis
      ↓
Categorical Encoding
      ↓
Correlation Analysis
      ↓
Feature Engineering
      ↓
Train/Test Split
      ↓
Feature Scaling
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Loan Approval Prediction
```

---

## 🧹 Data Preprocessing

### 1. Missing Value Handling

Missing numerical values were handled using **mean imputation**, while missing categorical values were replaced using the **most frequent value**.

### 2. Categorical Encoding

Two encoding techniques were used:

**Label Encoding**

* Education Level
* Loan Approved

**One-Hot Encoding**

* Employment Status
* Marital Status
* Loan Purpose
* Property Area
* Gender
* Employer Category

One-Hot Encoding was configured with `drop='first'` and `handle_unknown='ignore'`.

### 3. Feature Scaling

`StandardScaler` was used to standardize the features before model training.

---

## 🔎 Exploratory Data Analysis

The project includes exploratory analysis such as:

* Loan approval class distribution
* Gender distribution
* Numerical feature analysis
* Correlation matrix
* Correlation heatmap

The correlation analysis showed that some variables had stronger relationships with loan approval than others.

For example:

* Credit Score showed a positive relationship with loan approval.
* DTI Ratio showed a negative relationship with loan approval.
* Loan Amount also showed a negative relationship with the target.

---

## 🛠️ Feature Engineering

To improve the Logistic Regression model, additional nonlinear features were created:

```python
df['DTI_Ratio_sq'] = df['DTI_Ratio'] ** 2
df['Credit_Score_sq'] = df['Credit_Score'] ** 2
```

The original `DTI_Ratio`, `Credit_Score`, and `Applicant_Income` features were then excluded from the final feature matrix used in this experiment.

---

## 🤖 Models Tested

### 1. Logistic Regression

Used as the main linear classification model.

Initial performance:

* Accuracy: **86%**
* Precision for Approved: **0.78**
* Recall for Approved: **0.77**
* F1-score for Approved: **0.78**

After feature engineering:

* Accuracy: **87%**
* Precision for Approved: **0.78**
* Recall for Approved: **0.80**
* F1-score for Approved: **0.79**

---

### 2. K-Nearest Neighbors

KNN was tested with hyperparameter tuning using `GridSearchCV`.

The search evaluated different values of `n_neighbors` from 1 to 199 using 5-fold cross-validation and F1 scoring.

Best parameter found:

```text
n_neighbors = 3
```

Performance:

* Accuracy: **74%**
* Precision for Approved: **0.57**
* Recall for Approved: **0.57**
* F1-score for Approved: **0.57**

---

### 3. Gaussian Naive Bayes

Gaussian Naive Bayes was also evaluated.

Performance:

* Accuracy: **86%**
* Precision for Approved: **0.80**
* Recall for Approved: **0.74**
* F1-score for Approved: **0.77**

---

## 📈 Model Comparison

| Model                | Accuracy | Precision — Approved | Recall — Approved | F1 — Approved |
| -------------------- | -------: | -------------------: | ----------------: | ------------: |
| Logistic Regression  | **87%*** |                 0.78 |          **0.80** |      **0.79** |
| Gaussian Naive Bayes |      86% |             **0.80** |              0.74 |          0.77 |
| KNN                  |      74% |                 0.57 |              0.57 |          0.57 |

`*` Final Logistic Regression result after feature engineering.

---

## 🏆 Final Result

The final Logistic Regression model achieved:

### **87% Accuracy**

For the **Approved** class:

* Precision: **78%**
* Recall: **80%**
* F1-score: **79%**

The final confusion matrix was:

```text
[[125  14]
 [ 12  49]]
```

This means the model correctly classified most of the test applications while improving its ability to identify approved loans after feature engineering.

---

## 🧰 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab
* Jupyter Notebook

---

## 📚 Machine Learning Concepts Practiced

This project helped practice:

* Data Cleaning
* Missing Value Imputation
* Exploratory Data Analysis
* Categorical Encoding
* One-Hot Encoding
* Feature Scaling
* Correlation Analysis
* Feature Engineering
* Train/Test Split
* Logistic Regression
* K-Nearest Neighbors
* Naive Bayes
* Hyperparameter Tuning
* GridSearchCV
* Cross-Validation
* Confusion Matrix
* Precision
* Recall
* F1-score
* Model Comparison

---

## 🚀 Future Improvements

Possible improvements for the next version:

* Test additional classification algorithms.
* Perform more systematic feature selection.
* Tune Logistic Regression hyperparameters.
* Evaluate ROC-AUC and PR-AUC.
* Analyze class imbalance more deeply.
* Add explainability using feature importance/SHAP.
* Build a prediction interface using Streamlit.
* Package the preprocessing and model into a complete ML pipeline.
* Add fairness analysis across applicant groups.
* Deploy the final model as an API.

---

## 📂 Project Structure

```text
CreditWiseClassificationLoanProject/
│
├── CreditWiseClassificationLoanProject.ipynb
├── README.md
└── dataset/
    └── loan_approval_data.csv
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Open the notebook

Open:

```text
CreditWiseClassificationLoanProject.ipynb
```

using Jupyter Notebook or Google Colab.

### 3. Load the dataset

Make sure the loan dataset is available at the path expected by the notebook.

### 4. Run the notebook

Execute the cells sequentially to reproduce the preprocessing, analysis, model training, and evaluation.

---

## 👨‍💻 Author

**Muzammil Bughio**

Machine Learning | Python | Data Science | Cybersecurity

---

⭐ If you find this project useful, consider giving the repository a star!
