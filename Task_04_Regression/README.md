# Task 4: Supervised Learning – Regression

## Objective
Build a regression model to predict insurance charges using machine learning techniques and evaluate its performance.

---

## Dataset
- Dataset: insurance.csv
- Total Records: 1338
- Features:
  - age
  - sex
  - bmi
  - children
  - smoker
  - region
- Target Variable:
  - charges

---

## Task Roadmap

### 1. Data Loading
- Imported the insurance dataset using Pandas.
- Examined dataset structure and data types.

### 2. Data Preprocessing
- Checked for missing values.
- Converted categorical variables (sex, smoker, region) into numerical format using Label Encoding.

### 3. Feature Selection
- Selected all columns except `charges` as input features.
- Chose `charges` as the target variable.

### 4. Train-Test Split
- Split the dataset into:
  - 80% Training Data
  - 20% Testing Data

### 5. Linear Regression Model
- Trained a Linear Regression model on the training dataset.
- Generated predictions on the test dataset.

### 6. Model Evaluation
Evaluated the model using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### 7. Ridge Regression Model
- Trained a Ridge Regression model.
- Compared its performance with Linear Regression.

### 8. Visualization
- Created an Actual vs Predicted Charges scatter plot to visualize model performance.

### 9. Performance Comparison
- Compared Linear Regression and Ridge Regression results.
- Selected the model with the better R² Score.

---

## Results

| Model | R² Score |
|---------|---------|
| Linear Regression | ~0.78 |
| Ridge Regression | ~0.78 |

---

## Performance Analysis

The dataset was successfully preprocessed and used to train both Linear Regression and Ridge Regression models. The models achieved an R² score of approximately 0.78, indicating that they explain about 78% of the variation in insurance charges. Linear Regression performed slightly better than Ridge Regression. The Actual vs Predicted plot shows a reasonable alignment between predicted and actual values, demonstrating good predictive performance.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
