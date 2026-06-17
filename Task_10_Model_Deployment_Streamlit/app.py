import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("student_model.pkl")

st.title("Student Exam Score Predictor")

hours_studied = st.number_input(
    "Hours Studied",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=12.0,
    value=7.0
)

attendance_percent = st.number_input(
    "Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

previous_scores = st.number_input(
    "Previous Scores",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

if st.button("Predict"):

    input_data = np.array([[
        hours_studied,
        sleep_hours,
        attendance_percent,
        previous_scores
    ]])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Exam Score: {prediction[0]:.2f}"
    )

st.subheader("Model Explanation")

st.write("""
This application uses Linear Regression to predict
a student's exam score based on:
- Hours Studied
- Sleep Hours
- Attendance Percentage
- Previous Scores
""")