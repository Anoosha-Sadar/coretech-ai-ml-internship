  # Task 09: Model Optimization and Hyperparameter Tuning

## Intern Information

- **Name:** Anoosha Sadar
- **Education:** 2nd Year Information Technology Student

---

## Project Overview

This project focuses on optimizing and validating a **Client Retention Prediction** model using Hyperparameter Tuning techniques. The best-performing model from the previous task, **Logistic Regression**, was selected and optimized using **GridSearchCV**.

The project also includes Cross-Validation, Feature Importance Analysis, and Learning Curve Evaluation to assess model performance, stability, and generalization capability.

---

## Dataset Features

* Monthly_Spend
* Support_Tickets
* Contract_Length
* Service_Usage
* Customer_Satisfaction
* Renewal_History
* Client_Status (Target Variable)

---

## Methodology

* Data Preprocessing and Feature Scaling
* Baseline Logistic Regression Model
* Hyperparameter Tuning using GridSearchCV
* Cross-Validation Analysis
* Feature Importance Evaluation
* Learning Curve Analysis
* Performance Comparison

---

## Results

| Model                         | Accuracy |
| ----------------------------- | -------- |
| Baseline Logistic Regression  | 94.29%   |
| Optimized Logistic Regression | 94.29%   |

### Cross-Validation Performance

* Average Cross-Validation Score: **96.86%**

Although the accuracy remained unchanged after optimization, GridSearchCV successfully validated the model configuration and confirmed that the baseline model was already operating near its optimal performance level.

The strong cross-validation results further demonstrated model consistency and reliability across multiple data splits.

---

## Key Findings

* Logistic Regression achieved strong classification performance.
* Hyperparameter tuning confirmed the effectiveness of the selected model configuration.
* Cross-validation demonstrated excellent model stability and generalization capability.
* Feature importance analysis identified the most influential factors affecting customer retention.
* Learning curve analysis showed balanced learning behavior without significant overfitting or underfitting.

---

## Business Insights

The model can help organizations identify customer retention patterns and proactively detect clients who may be at risk of leaving. Such insights can support customer engagement strategies, improve retention rates, and enable data-driven business decisions.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Google Colab
* GitHub

---

## Repository Structure

```text
Task_09/
│
├── task_09_client_retention_optimization.ipynb
├── coretech_client_retention.csv
├── optimization_comparison.png
├── feature_importance.png
├── learning_curve.png
├── confusion_matrix.png
└── README.md
```

---
### Impact of Hyperparameter Tuning on Model Performance

The baseline Logistic Regression model achieved an accuracy of 94.29% before hyperparameter tuning. After applying GridSearchCV, the optimized model achieved the same accuracy score.

Although the overall accuracy remained unchanged, the optimization process successfully evaluated multiple parameter combinations and confirmed that the selected model configuration was already operating near its optimal performance level.

The primary contribution of hyperparameter tuning in this project was model validation rather than accuracy improvement. The process verified the effectiveness of the model parameters and increased confidence in the reliability of the final solution.

Furthermore, the cross-validation results produced an average score of 96.86%, demonstrating that the model maintained strong and consistent performance across multiple data partitions. This confirms that the model is stable, reliable, and capable of generalizing effectively to unseen data.

## Conclusion

This project successfully achieved the objective of model optimization and validation for client retention prediction. While hyperparameter tuning did not increase the accuracy score, it confirmed that the Logistic Regression model was already performing near its optimal level.

The combination of GridSearchCV, cross-validation, feature importance analysis, and learning curve evaluation provided strong evidence of model reliability, stability, and generalization. Therefore, the task objectives were successfully achieved, resulting in a robust and trustworthy machine learning solution for client retention prediction.
