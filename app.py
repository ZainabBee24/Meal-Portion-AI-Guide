import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load model
model = joblib.load("model.pkl")

# Manual encoders
gender_map = {'Male': 0, 'Female': 1}
activity_map = {
    'Sedentary': 1.2,
    'Lightly Active': 1.375,
    'Moderately Active': 1.55,
    'Very Active': 1.725
}

sleep_map = {i: i for i in range(1, 6)}  # 1 to 5

# App layout
st.title("🍽️ AI-Powered Meal Portion Guide")
st.markdown("Estimate your Total Daily Energy Expenditure (TDEE) and get meal portion suggestions.")

# User inputs
age = st.slider("Age", min_value=10, max_value=100, step=1)
gender = st.selectbox("Gender", list(gender_map.keys()))
weight = st.number_input("Current Weight (lbs)", min_value=1.0)
activity = st.selectbox("Physical Activity Level", list(activity_map.keys()))
sleep = st.slider("Sleep Quality (1 = Poor, 5 = Excellent)", min_value=1, max_value=5, step=1)
stress = st.slider("Stress Level (1 = Low, 5 = High)", min_value=1, max_value=5, step=1)

if st.button("Generate Meal Plan"):
    gender_enc = gender_map.get(gender, 0)
    activity_factor_val = activity_map.get(activity, 1.2)
    sleep_quality_score_val = sleep_map.get(sleep, 1)

    # Prepare input
    features = [[age, gender_enc, weight, activity_factor_val, sleep_quality_score_val, stress]]
    input_df = pd.DataFrame(features, columns=[
        'age', 'gender
