# Task 08: Neural Network Basics with Keras — Client Retention Prediction

## Intern 

- **Name:** Anoosha Sadar
- **Education:** 2nd Year Information Technology Student

---

## Project Overview

This project implements an Artificial Neural Network (ANN) using TensorFlow and Keras to predict client retention behavior. The objective is to classify customers as **Retained**, **At Risk**, or **Churned** based on customer engagement and business-related features.

A custom dataset was created to simulate realistic customer retention scenarios and demonstrate the practical application of deep learning for business analytics.

---

## Dataset Features

* Client_ID
* Monthly_Spend
* Support_Tickets
* Contract_Length
* Service_Usage
* Customer_Satisfaction
* Renewal_History
* Client_Status (Target Variable)

---

## Methodology

### Data Preprocessing

* Feature Selection
* Label Encoding
* Standardization using StandardScaler
* Train-Test Split

### Neural Network Architecture

* Input Layer
* Hidden Layer 1 → 128 Neurons (ReLU)
* Hidden Layer 2 → 64 Neurons (ReLU)
* Hidden Layer 3 → 32 Neurons (ReLU)
* Output Layer → Softmax

### Model Training

* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Epochs: 50
* Batch Size: 16

---

## Model Evaluation

The model was evaluated using:

* Accuracy Score
* Classification Report
* Confusion Matrix
* Accuracy & Loss Curves

### Model Comparison

| Model                     | Accuracy |
| ------------------------- | -------- |
| Artificial Neural Network | 91.43%   |
| Logistic Regression       | 94.29%   |

Although Logistic Regression achieved slightly higher accuracy, the ANN demonstrated strong predictive performance and effectively learned customer retention patterns.

---

## Business Insights

* Customer Satisfaction strongly influences retention behavior.
* Higher Service Usage is associated with retained customers.
* Frequent Support Tickets may indicate churn risk.
* Renewal History contributes significantly to customer loyalty prediction.

These insights can help organizations improve customer retention strategies and reduce churn.

---

## Technologies Used

* Python
* Pandas
* NumPy
* TensorFlow
* Keras
* Scikit-Learn
* Matplotlib
* Seaborn
* Google Colab
* GitHub

---

## Repository Structure

```text
Task_08/
│
├── task_08_client_retention_ann.ipynb
├── coretech_client_retention.csv
├── client_status_distribution.png
├── accuracy_curve.png
├── loss_curve.png
├── confusion_matrix.png
├── model_comparison.png
└── README.md
```

---

## Conclusion

This project successfully developed a Neural Network-based client retention prediction system using TensorFlow and Keras. The model demonstrated strong classification performance and provided valuable insights into customer behavior. The results highlight the practical role of deep learning in customer analytics and business decision-making.

