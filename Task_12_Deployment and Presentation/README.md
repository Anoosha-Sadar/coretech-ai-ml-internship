# SentimentAI Pro – Enterprise Customer Feedback Intelligence Platform

## Project Overview

SentimentAI Pro is an Artificial Intelligence powered Customer Feedback Intelligence Platform that analyzes customer reviews, comments, and feedback using Natural Language Processing (NLP) and Machine Learning.

The system automatically classifies customer feedback into one of four sentiment categories:

* Positive
* Negative
* Neutral
* Irrelevant

In addition to sentiment prediction, the platform generates confidence scores, customer health analysis, business risk assessment, keyword intelligence, executive summaries, and actionable business recommendations.

---

# Problem Statement

Organizations receive large volumes of customer feedback every day through reviews, surveys, social media, and support channels.

Manually analyzing this feedback is time-consuming and inefficient.

The objective of this project is to develop an AI-based system capable of automatically understanding customer opinions and converting them into meaningful business insights.

---

#  Dataset Description

Dataset Used: Twitter Entity Sentiment Analysis Dataset

The dataset contains customer opinions and social media posts labeled with sentiment categories.

Target Classes:

1. Positive
2. Negative
3. Neutral
4. Irrelevant

The dataset was used to train and evaluate multiple machine learning models for sentiment classification.

---

#  Data Preprocessing Pipeline

The following preprocessing steps were performed:

### Text Cleaning

* Converted text to lowercase
* Removed URLs
* Removed special characters
* Removed punctuation
* Removed numbers

### Stopword Removal

Common English stopwords were removed using NLTK.

### Lemmatization

Words were converted to their root form using WordNet Lemmatizer.

### Feature Engineering

TF-IDF Vectorization was applied to transform textual data into numerical feature vectors.

Maximum Features:

* 5000 Features

---

# Exploratory Data Analysis (EDA)

The following visualizations were created:

### 1. Sentiment Distribution

Shows the distribution of sentiment classes.

### 2. Tweet Length Distribution

Analyzes the length of customer feedback.

### 3. Tweet Length by Sentiment

Compares text length across sentiment categories.

### 4. Model Accuracy Comparison

Compares performance of multiple machine learning models.

### 5. Confusion Matrix

Visualizes prediction performance of the final model.

### 6. Cross Validation Analysis

Evaluates model stability across multiple folds.

### 7. Sentiment Probability Distribution

Displays prediction probabilities for each sentiment class.

### 8. Customer Health Analysis

Shows overall customer satisfaction score.

---

# Machine Learning Models Evaluated

The following models were trained and compared:

| Model                  | Accuracy |
| ---------------------- | -------- |
| Logistic Regression    | 68.15%   |
| Naive Bayes            | 63.60%   |
| Random Forest          | 87.32%   |
| Gradient Boosting      | 52.30%   |
| Extra Trees Classifier | 89.71%   |

Best Performing Model:

Extra Trees Classifier

---

#  Hyperparameter Tuning

GridSearchCV was used to optimize the Extra Trees Classifier.

Best Parameters:

* n_estimators = 100
* max_depth = None

Best Cross Validation Score:

89.88%

---

#  Model Evaluation

### Accuracy

89.71%

### Cross Validation Score

89.88%

### Classification Performance

The model achieved strong Precision, Recall, and F1-Scores across all sentiment categories.

### Confusion Matrix

A confusion matrix was generated to evaluate class-wise prediction performance.

---

#  Streamlit Web Application

A professional interactive web application was developed using Streamlit.

### Features

* Real-time Sentiment Prediction
* Confidence Analysis
* Probability Distribution Charts
* Customer Health Score
* Business Risk Meter
* Keyword Intelligence
* Executive Summary Generation
* Executive Decision Panel
* AI Recommendation System
* Downloadable Reports
* Interactive Analytics Dashboard

---

#  Business Intelligence Features

The platform provides:

### Customer Satisfaction Analysis

Measures customer experience level.

### Business Risk Assessment

Identifies potential business threats.

### Customer Health Score

Evaluates overall customer relationship strength.

### Executive Recommendations

Provides actionable business strategies.

### Keyword Intelligence

Highlights important terms from feedback.

---

#  Technologies Used

Programming Language:

* Python

Libraries:

* Pandas
* NumPy
* Scikit-learn
* NLTK
* Plotly
* Streamlit
* Joblib

Machine Learning:

* Extra Trees Classifier
* TF-IDF Vectorizer
* GridSearchCV
* Cross Validation

---

# 📁 Project Structure

```text
SentimentAI-Pro/
│
├── Task.ipynb
├── app.py
├── Pickle files (Google Drive Link)
├── requirements.txt
├── README.md
│
├── Streamlit/
│   ├── Interface.pdf
│
├── Dataset/
│   ├── twitter_training.csv
│   ├── twitter_validation.csv
│
├── EDA/
│   ├── sentiment_distribution.png
│   ├── tweet_length_distribution.png
│   ├── tweet_length_by_sentiment.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│
└── Demo_Video.mp4
```
---

# Key Insights

* Positive feedback indicates strong customer satisfaction.
* Negative feedback highlights areas requiring immediate attention.
* Neutral feedback represents opportunities for customer engagement.
* Irrelevant feedback can be filtered automatically.
* AI can significantly reduce manual sentiment analysis efforts.

---

# Recommendations

* Integrate the system with CRM platforms.
* Deploy the model for real-time customer monitoring.
* Use feedback trends to improve business strategy.
* Continuously retrain the model using new customer data.
* Implement automated alert systems for negative feedback.

---

# Academic Information

Project Type:
Machine Learning & Natural Language Processing

Task:
Final Project Deployment and Presentation (Task 12)

Domain:
Sentiment Analysis & Business Intelligence

Platform:
Streamlit Web Application

---

# Developer

Anoosha Sadar

Artificial Intelligence & ML Engineer

Specializations:

* Machine Learning
* Natural Language Processing
* Data Analytics
* Business Intelligence

---

# Conclusion

SentimentAI Pro successfully demonstrates the complete Machine Learning lifecycle, including data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, optimization, deployment, and business intelligence generation.

The project provides a practical AI-powered solution for automated customer feedback analysis and decision support, achieving an accuracy of 89.71% and a cross-validation score of 89.88%.
