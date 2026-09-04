# 🤖 Machine Learning Classification Project

A practical machine learning classification project comparing multiple supervised learning algorithms using preprocessing, model training, hyperparameter tuning, and evaluation techniques.

The goal of this project is to understand how different classification algorithms perform on the same dataset and determine which model provides the best generalization performance.

---

## 📌 Table of Contents

* [Project Overview](#-project-overview)
* [Objectives](#-objectives)
* [Dataset](#-dataset)
* [Machine Learning Workflow](#-machine-learning-workflow)
* [Data Preprocessing](#-data-preprocessing)
* [Models Used](#-models-used)
* [Hyperparameter Tuning](#-hyperparameter-tuning)
* [Evaluation Metrics](#-evaluation-metrics)
* [Model Results](#-model-results)
* [Confusion Matrices](#-confusion-matrices)
* [Model Comparison](#-model-comparison)
* [Best Model](#-best-model)
* [Key Observations](#-key-observations)
* [Technologies Used](#-technologies-used)
* [Project Structure](#-project-structure)
* [How to Run](#-how-to-run)
* [Future Improvements](#-future-improvements)
* [Conclusion](#-conclusion)

---

# 📖 Project Overview

This project focuses on solving a **binary classification problem** using several machine learning algorithms.

Instead of relying on a single model, multiple classification algorithms were trained and evaluated under different configurations.

The project demonstrates the complete machine learning workflow:

```text
Dataset
   ↓
Data Understanding
   ↓
EDA
   ↓
Data Preprocessing
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Model Comparison
```

The main purpose is not only to obtain a high score, but also to understand:

* How different algorithms behave
* The effect of preprocessing
* The effect of hyperparameter tuning
* Overfitting and underfitting
* The importance of train/test performance
* How to interpret confusion matrices
* How to select an appropriate final model

---

# 🎯 Objectives

The main objectives of this project are:

* Perform exploratory data analysis (EDA)
* Understand the features and target variable
* Prepare the data for machine learning
* Apply appropriate preprocessing techniques
* Train multiple classification algorithms
* Use pipelines where appropriate
* Perform hyperparameter tuning using `GridSearchCV`
* Evaluate models using multiple metrics
* Compare training and testing performance
* Analyze overfitting
* Select the best-performing model

---

# 📊 Dataset

The dataset contains **1,910 test samples** in the evaluated test set and represents a binary classification problem.

The target contains two classes:

```text
0
1
```

The dataset was divided into training and testing sets before model evaluation.

The test set contains:

| Class     |   Samples |
| --------- | --------: |
| 0         |       900 |
| 1         |     1,010 |
| **Total** | **1,910** |

The classes are reasonably balanced, although class `1` has slightly more observations.

---

# 🔄 Machine Learning Workflow

The project follows a standard supervised machine learning workflow.

### 1. Data Loading

The dataset is loaded into a Pandas DataFrame.

### 2. Data Exploration

The dataset is inspected to understand:

* Number of observations
* Number of features
* Data types
* Target distribution
* Duplicate values
* Missing values
* Feature distributions
* Relationships between variables

### 3. Data Preprocessing

Depending on the model, preprocessing can include:

* Feature scaling
* Encoding categorical variables
* Handling missing values
* Removing unnecessary columns
* Feature transformation

### 4. Train/Test Split

The data is divided into:

```text
Training Data → Model learning
Testing Data  → Final evaluation
```

### 5. Model Training

Several classification algorithms are trained.

### 6. Hyperparameter Tuning

Important models are tuned using `GridSearchCV`.

### 7. Evaluation

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

---

# 🧹 Data Preprocessing

Preprocessing depends on the requirements of each algorithm.

## Feature Scaling

Feature scaling is particularly important for distance- and margin-based algorithms.

It was used for models such as:

* Logistic Regression
* KNN
* SVM

`StandardScaler` was used where appropriate.

### Why?

Suppose one feature ranges from:

```text
0 → 1
```

while another ranges from:

```text
0 → 100,000
```

Distance-based and optimization-based models can be strongly affected by this difference.

Scaling places features on a comparable scale.

---

# 🔗 Pipeline

`Pipeline` was used to combine preprocessing and model training into a single workflow.

Conceptually:

```text
Input Data
    ↓
Preprocessing
    ↓
Model
    ↓
Prediction
```

Using a pipeline provides several advantages:

* Keeps preprocessing organized
* Prevents inconsistent transformations
* Helps avoid data leakage
* Makes GridSearchCV easier to use
* Ensures the same preprocessing is applied during training and prediction

---

# 🔍 ColumnTransformer

`ColumnTransformer` can be used when different groups of features require different preprocessing.

For example:

```text
Numerical Features
       ↓
StandardScaler

Categorical Features
       ↓
OneHotEncoder
```

This allows different transformations to be applied to different columns.

---

# 🤖 Models Used

The following classification algorithms were evaluated.

## 1. Logistic Regression

Logistic Regression was used as a simple baseline classification model.

It is useful because it provides a relatively simple reference point for comparing more complex algorithms.

---

## 2. Naive Bayes

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

It assumes that features are conditionally independent given the class.

Two configurations were tested:

* Without a pipeline
* With a pipeline

The results were identical because the preprocessing in the pipeline did not change the data in a way that affected the model.

---

## 3. K-Nearest Neighbors (KNN)

KNN classifies an observation based on nearby observations.

Because KNN relies on distances between observations, feature scaling is particularly important.

`GridSearchCV` was used to search for suitable hyperparameters.

---

## 4. Decision Tree

Decision Tree learns a sequence of decision rules to divide the data into different classes.

Two configurations were evaluated:

### Without pruning

The unrestricted tree achieved:

```text
Training F1 = 100%
Testing F1  = 89.76%
```

This demonstrates significant overfitting.

### With pruning and GridSearchCV

After controlling tree complexity:

```text
Training F1 = 95.50%
Testing F1  = 91.70%
```

The testing performance improved while the train-test gap decreased.

---

## 5. Random Forest

Random Forest combines multiple decision trees to produce a more robust ensemble model.

Two configurations were evaluated:

* 100 estimators
* 200 estimators

The 200-estimator configuration achieved the strongest result among the tested models.

---

## 6. XGBoost

XGBoost is a gradient boosting algorithm that builds an ensemble of trees sequentially.

It was included as an additional ensemble comparison.

---

## 7. Support Vector Machine (SVM)

SVM attempts to find an optimal decision boundary that separates classes while maximizing the margin between them.

GridSearchCV and a pipeline were used to optimize the model.

---

# ⚙️ Hyperparameter Tuning

`GridSearchCV` was used to systematically evaluate different hyperparameter combinations.

Instead of manually guessing a parameter, GridSearchCV evaluates multiple combinations using cross-validation and selects the best configuration according to the chosen scoring metric.

Conceptually:

```text
Parameter combinations
        ↓
Cross-validation
        ↓
Performance comparison
        ↓
Best parameters
        ↓
Best estimator
```

### Important observation

GridSearchCV does **not guarantee** that a model will perform better than its default configuration.

Its purpose is to find a good configuration within the parameter search space.

Similarly, a `Pipeline` does not automatically improve model performance. Its main purpose is to make preprocessing and modeling consistent and reduce the risk of data leakage.

---

# 📏 Evaluation Metrics

Several metrics were used to evaluate the models.

## Accuracy

Accuracy represents the proportion of correctly classified observations.

```text
Accuracy =
Correct Predictions / Total Predictions
```

Higher is generally better.

---

## Precision

Precision answers:

> Of all observations predicted as positive, how many were actually positive?

```text
Precision = TP / (TP + FP)
```

---

## Recall

Recall answers:

> Of all actual positive observations, how many did the model correctly identify?

```text
Recall = TP / (TP + FN)
```

---

## F1 Score

F1 Score combines precision and recall using their harmonic mean.

```text
F1 = 2 × (Precision × Recall) /
     (Precision + Recall)
```

F1 is particularly useful when both precision and recall are important.

---

# 📊 Model Results

The following results were obtained on the test set.

| Model                   |    Train F1 |    Test F1 |   Accuracy |  Precision |     Recall |
| ----------------------- | ----------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression     |      82.46% |     83.21% |     82.15% |     82.76% |     83.66% |
| Naive Bayes             |      81.63% |     82.71% |     82.09% |     84.50% |     80.99% |
| KNN + GridSearch        |      94.45% |     92.95% |     92.62% |     93.93% |     91.98% |
| Decision Tree           |     100.00% |     89.76% |     89.27% |     90.62% |     88.91% |
| Decision Tree + Pruning |      95.50% |     91.70% |     91.26% |     92.11% |     91.29% |
| Random Forest (100)     |      96.93% |     93.61% |     93.25% |     93.66% |     93.56% |
| XGBoost                 |      95.57% |     93.27% |     92.93% |     93.97% |     92.57% |
| SVM + GridSearch        |      98.48% |     94.60% |     94.29% |     94.65% |     94.55% |
| **Random Forest (200)** | **100.00%** | **94.70%** | **94.40%** | **94.84%** | **94.55%** |

---

# 🏆 Model Ranking

Based on **test F1 score**:

| Rank | Model                   |    Test F1 |
| ---: | ----------------------- | ---------: |
| 🥇 1 | **Random Forest (200)** | **94.70%** |
| 🥈 2 | **SVM**                 | **94.60%** |
| 🥉 3 | Random Forest (100)     |     93.61% |
|    4 | XGBoost                 |     93.27% |
|    5 | KNN                     |     92.95% |
|    6 | Decision Tree + Pruning |     91.70% |
|    7 | Decision Tree           |     89.76% |
|    8 | Logistic Regression     |     83.21% |
|    9 | Naive Bayes             |     82.71% |

---

# 🎯 Accuracy Ranking

| Rank | Model                   |   Accuracy |
| ---: | ----------------------- | ---------: |
| 🥇 1 | **Random Forest (200)** | **94.40%** |
| 🥈 2 | SVM                     |     94.29% |
| 🥉 3 | Random Forest (100)     |     93.25% |
|    4 | XGBoost                 |     92.93% |
|    5 | KNN                     |     92.62% |
|    6 | Decision Tree + Pruning |     91.26% |
|    7 | Decision Tree           |     89.27% |
|    8 | Logistic Regression     |     82.15% |
|    9 | Naive Bayes             |     82.09% |

---

# 🔲 Confusion Matrices

## Logistic Regression

```text
[[724 176]
 [165 845]]
```

The model produced:

* True Negatives = 724
* False Positives = 176
* False Negatives = 165
* True Positives = 845

---

## Naive Bayes

```text
[[750 150]
 [192 818]]
```

The model produced:

* True Negatives = 750
* False Positives = 150
* False Negatives = 192
* True Positives = 818

---

## KNN

```text
[[840  60]
 [ 81 929]]
```

The model produced:

* True Negatives = 840
* False Positives = 60
* False Negatives = 81
* True Positives = 929

---

## Decision Tree

```text
[[807  93]
 [112 898]]
```

---

## Decision Tree + Pruning

```text
[[821  79]
 [ 88 922]]
```

---

## Random Forest — 100 Estimators

```text
[[836 64]
 [ 65 945]]
```

---

## XGBoost

```text
[[840 60]
 [ 75 935]]
```

---

## SVM

```text
[[846 54]
 [ 55 955]]
```

---

## Random Forest — 200 Estimators

```text
[[848 52]
 [ 55 955]]
```

The Random Forest with 200 estimators made:

```text
52 + 55 = 107
```

incorrect predictions out of 1,910 test observations.

---

# 🔎 Overfitting Analysis

One of the most important observations from this project was the difference between training and testing performance.

## Decision Tree

```text
Train F1 = 100%
Test F1  = 89.76%
```

Difference:

```text
10.24 percentage points
```

This is a strong indication of overfitting.

The tree was able to fit the training data extremely well but did not generalize as effectively to unseen data.

---

## Decision Tree with Pruning

After applying pruning and hyperparameter tuning:

```text
Train F1 = 95.50%
Test F1  = 91.70%
```

The test performance improved from **89.76% → 91.70%**.

This demonstrates how controlling model complexity can improve generalization.

---

## Random Forest

The 200-estimator Random Forest achieved:

```text
Train F1 = 100%
Test F1  = 94.70%
```

Although the training score is perfect, the test score remains very high.

The difference is:

```text
5.30 percentage points
```

This indicates some overfitting, but the model generalizes considerably better than the unrestricted single Decision Tree.

---

# 🏅 Best Model

Based on the current test-set results:

## Random Forest with 200 estimators

It achieved:

```text
F1 Score  : 94.70%
Accuracy  : 94.40%
Precision : 94.84%
Recall    : 94.55%
```

Confusion matrix:

```text
[[848  52]
 [ 55 955]]
```

However, the SVM result is extremely close:

```text
Random Forest F1 = 94.70%
SVM F1           = 94.60%
```

The difference is only:

```text
0.10 percentage points
```

Therefore, the Random Forest should not be considered dramatically better than SVM based on this single test split.

Cross-validation should be used to determine whether the difference is consistent.

---

# 💡 Key Observations

### 1. More complex models performed better

The ensemble and nonlinear models generally performed better than the simpler baseline models.

---

### 2. Decision Tree overfit strongly

The unrestricted Decision Tree achieved 100% training F1 but only 89.76% testing F1.

This clearly demonstrates overfitting.

---

### 3. Pruning improved generalization

Adding pruning reduced the training performance but increased testing performance.

This is a good example of the trade-off between fitting the training data and generalizing to unseen data.

---

### 4. KNN performed well

KNN achieved:

```text
Test F1 = 92.95%
```

Feature scaling and hyperparameter tuning helped make KNN effective.

---

### 5. Random Forest performed strongly

Increasing the number of trees from 100 to 200 improved the test F1 score:

```text
100 estimators → 93.61%
200 estimators → 94.70%
```

However, increasing `n_estimators` does not guarantee improvement indefinitely.

---

### 6. SVM was highly competitive

SVM achieved:

```text
Test F1 = 94.60%
```

which was only slightly below the 200-estimator Random Forest.

---

### 7. Pipeline does not automatically increase performance

A pipeline is primarily a **workflow and data-leakage prevention tool**.

If the same preprocessing is performed manually and correctly, the model may produce nearly identical results.

---

### 8. GridSearchCV does not guarantee improvement

GridSearchCV searches for good hyperparameters.

If the default parameters are already suitable, tuning may produce little or no improvement.

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **XGBoost**
* **Jupyter Notebook / Google Colab**

---

# 📦 Important Libraries

The project uses tools such as:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

Important Scikit-learn components include:

```text
train_test_split
Pipeline
ColumnTransformer
StandardScaler
OneHotEncoder
GridSearchCV
classification_report
confusion_matrix
accuracy_score
precision_score
recall_score
f1_score
```

---




# 🔮 Future Improvements

Possible future improvements include:

* Perform repeated cross-validation
* Compare cross-validation F1 scores
* Expand hyperparameter search spaces
* Tune Random Forest more thoroughly
* Tune SVM more thoroughly
* Analyze feature importance
* Perform feature selection
* Investigate class-specific errors
* Test different decision thresholds
* Evaluate model stability across multiple train/test splits
* Save the final trained model
* Create a prediction interface
* Deploy the final model as an API or web application

---

# ⚠️ Important Note About Model Selection

The reported results are based on the current train/test split.

A test score should not be interpreted as a universal measure of how the model will perform on every future dataset.

For a more reliable comparison, the final candidate models should be evaluated using cross-validation.

In particular, because Random Forest and SVM differ by only about **0.10 percentage points in F1**, cross-validation can help determine whether this difference is meaningful or simply caused by the particular test split.

---

# 🧠 What I Learned

Through this project, I learned how to:

* Understand classification datasets
* Perform exploratory data analysis
* Prepare data for machine learning
* Apply feature scaling
* Use encoding techniques
* Build Scikit-learn pipelines
* Use ColumnTransformer
* Train classification models
* Tune hyperparameters with GridSearchCV
* Interpret confusion matrices
* Calculate classification metrics
* Identify overfitting
* Understand the purpose of pruning
* Compare multiple machine learning algorithms
* Select a model based on test performance rather than training performance

---

# 🏁 Conclusion

This project demonstrates that **model selection is not simply about choosing the most complex algorithm**.

Different models produced significantly different results on the same dataset.

The strongest results came from ensemble and margin-based methods.

The current best-performing model was:

> **Random Forest with 200 estimators**

with a test F1 score of approximately:

> **94.70%**

SVM was extremely competitive with a test F1 score of:

> **94.60%**

The project also demonstrated an important machine learning principle:

> **A model that performs extremely well on training data is not necessarily the best model.**

The unrestricted Decision Tree achieved 100% training F1 but only 89.76% testing F1, demonstrating overfitting.

Therefore, **generalization performance on unseen data is more important than simply maximizing training performance**.

---

## ⭐ Final Result

```text
Best Model:
Random Forest

n_estimators:
200

Test Accuracy:
94.40%

Test Precision:
94.84%

Test Recall:
94.55%

Test F1 Score:
94.70%
```

---

## 👨‍💻 Author

**Muzammil**

This project is part of my practical journey in **Machine Learning, Python, and AI**.
