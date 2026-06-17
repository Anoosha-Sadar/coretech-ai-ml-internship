# Model Optimization and Hyperparameter Tuning

## Project Overview

This project focuses on improving the performance of a machine learning classification model using hyperparameter tuning and model optimization techniques. The IBM HR Employee Attrition dataset was used to predict employee attrition and analyze factors affecting employee turnover.

---

## Objectives

* Train a baseline classification model
* Apply GridSearchCV for hyperparameter tuning
* Perform K-Fold Cross Validation
* Analyze feature importance
* Plot learning curves
* Detect overfitting and underfitting
* Compare model performance before and after optimization

---

## Dataset Information

Dataset Name: WA_Fn-UseC_-HR-Employee-Attrition.csv

### Target Variable

* Attrition (Yes/No)

### Dataset Description

The dataset contains employee information such as:

* Age
* Job Role
* Department
* Monthly Income
* Job Satisfaction
* Years at Company
* Overtime
* Education
* Attrition Status

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn

---

## Project Workflow

### Step 1: Import Libraries

Imported all required libraries for machine learning, evaluation, visualization, and optimization.

### Step 2: Load Dataset

Loaded the IBM HR Employee Attrition dataset.

### Step 3: Data Exploration

Performed:

* Dataset inspection
* Shape analysis
* Missing value analysis
* Feature review

### Step 4: Data Preprocessing

Applied:

* Target encoding
* One-hot encoding for categorical features

### Step 5: Feature Selection

Separated input features (X) and target variable (y).

### Step 6: Train-Test Split

Split dataset into training and testing sets.

### Step 7: Baseline Model Training

Trained a Random Forest Classifier with default parameters.

### Step 8: Baseline Model Evaluation

Measured model accuracy before optimization.

### Step 9: Hyperparameter Tuning

Applied GridSearchCV to find optimal values for:

* Number of Trees (n_estimators)
* Maximum Tree Depth (max_depth)
* Minimum Samples Split (min_samples_split)

### Step 10: Optimized Model Training

Trained the model using the best parameters found by GridSearchCV.

### Step 11: Optimized Model Evaluation

Measured model accuracy after optimization.

### Step 12: K-Fold Cross Validation

Performed 5-Fold Cross Validation to validate model consistency.

### Step 13: Feature Importance Analysis

Identified the most influential features affecting employee attrition.

### Step 14: Learning Curve Analysis

Generated learning curves to evaluate model learning behavior.

### Step 15: Overfitting and Underfitting Analysis

Compared training and testing accuracy to determine model fit quality.

### Step 16: Performance Comparison

Compared results before and after optimization.

---

## Hyperparameter Tuning

Technique Used:

* GridSearchCV

Purpose:

* Automatically search for the best parameter combination.
* Improve model performance.
* Reduce overfitting risk.

---

## Cross Validation

Method Used:

* 5-Fold Cross Validation

Benefits:

* Better model reliability
* Reduced bias
* More stable performance estimate

---

## Feature Importance Analysis

Feature importance was calculated using the optimized Random Forest model.

The analysis identifies the most influential variables contributing to employee attrition prediction.

---

## Learning Curve Analysis

Learning curves were plotted to compare:

* Training Accuracy
* Validation Accuracy

Purpose:

* Evaluate model learning performance
* Detect overfitting or underfitting

---

## Overfitting and Underfitting Analysis

### Overfitting

Occurs when:

Training Accuracy >> Testing Accuracy

### Underfitting

Occurs when:

Training Accuracy and Testing Accuracy are both low.

### Good Fit

Occurs when:

Training Accuracy ≈ Testing Accuracy and both are high.

---

## Results Comparison

| Metric   | Before Optimization | After Optimization |
| -------- | ------------------- | ------------------ |
| Accuracy | Baseline Accuracy   | Optimized Accuracy |

The optimized model achieved improved predictive performance through hyperparameter tuning.

---

## Output Files

### Feature Importance Graph

feature_importance.png

### Learning Curve

learning_curve.png

---

## Folder Structure

Task-9-Model-Optimization/

├── WA_Fn-UseC_-HR-Employee-Attrition.csv

├── task9_model_optimization.ipynb

├── feature_importance.png

├── learning_curve.png

├── requirements.txt

└── README.md

---

## Conclusion

This project successfully improved a classification model using hyperparameter tuning and optimization techniques. GridSearchCV identified optimal model parameters, cross-validation verified model stability, feature importance analysis highlighted key attrition factors, and learning curves helped evaluate model behavior. The optimized model achieved better accuracy and generalization compared to the baseline model.

---

## Author

Your Name
