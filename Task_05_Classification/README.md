# Task 05: Supervised Learning - Classification

## Objective
The objective of this task is to build and compare multiple classification models to predict employee attrition using machine learning techniques.

---

## Dataset Information

**Dataset:** IBM HR Analytics Employee Attrition & Performance

**File Name:** WA_Fn-UseC_-HR-Employee-Attrition.csv

**Target Variable:** Attrition

---

## Tools and Libraries Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Models Implemented

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

---

## Steps Performed

### 1. Data Loading
- Loaded the dataset into a Pandas DataFrame.

### 2. Data Preprocessing
- Checked dataset information.
- Encoded categorical features using Label Encoding.

### 3. Feature Selection
- Selected input features (X).
- Selected target variable (y).

### 4. Train-Test Split
- Split data into 80% training and 20% testing sets.

### 5. Model Training
- Trained Logistic Regression model.
- Trained Decision Tree model.
- Trained Random Forest model.

### 6. Model Evaluation
Evaluated each model using:
- Accuracy Score
- Confusion Matrix
- Classification Report

### 7. Visualization
- Generated Confusion Matrix Heatmap for the best-performing model.

---

## Results

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 0.860544 |
| Decision Tree | 0.765306 |
| Random Forest | 0.867347 |

---

## Best Model

Random Forest achieved the highest accuracy among all models and provided the most reliable predictions. Therefore, it was selected as the best-performing model.

---

## Performance Analysis

Three classification algorithms were trained and evaluated on the IBM HR Analytics dataset. Random Forest outperformed Logistic Regression and Decision Tree in terms of accuracy and overall classification performance. The confusion matrix showed fewer misclassifications, making Random Forest the most suitable model for predicting employee attrition.

---

## Files Included

- Task5_Classification.ipynb
- confusion_matrix.png
- README.md

---

## Conclusion

This task demonstrated the implementation and comparison of multiple classification algorithms. Among the tested models, Random Forest produced the best results and was selected as the final model for employee attrition prediction.
