
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib

import os
import streamlit as st  # if not already imported

st.write("model.pkl found:", os.path.exists("model.pkl"))
st.write("le_gender.pkl found:", os.path.exists("le_gender.pkl"))
st.write("le_activity.pkl found:", os.path.exists("le_activity.pkl"))

# Load model and encoders
model = joblib.load("model.pkl")
le_gender = joblib.load("le_gender.pkl")
le_activity = joblib.load("le_activity.pkl")

def gradio_predict(age, gender, weight, activity, sleep, stress):
    gender_enc = le_gender.transform([gender])[0]
    activity_enc = le_activity.transform([activity])[0]

    activity_map = {
        'Sedentary': 1.2,
        'Lightly Active': 1.375,
        'Moderately Active': 1.55,
        'Very Active': 1.725
    }

    sleep_map = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5
    }

    activity_factor_val = activity_map.get(activity, 1.2)
    sleep_quality_score_val = sleep_map.get(sleep, 1)

    features = [[age, gender_enc, weight, activity_factor_val, sleep_quality_score_val, stress]]
    input_df = pd.DataFrame(features,
                            columns=['age', 'gender', 'weight', 'activity_factor', 'sleep_quality_score', 'stress'])

    bmr_pred = model.predict(input_df)[0]
    tdee = bmr_pred * activity_factor_val

    portions = {
        'Breakfast': round(tdee * 0.25),
        'Lunch': round(tdee * 0.35),
        'Dinner': round(tdee * 0.30),
        'Snacks': round(tdee * 0.10)
    }

    summary = f"Estimated TDEE: {round(tdee)} calories/day\nSuggested meal portions:\n"
    for meal, cal in portions.items():
        summary += f"- {meal}: {cal} calories\n"

    fig, ax = plt.subplots()
    ax.bar(portions.keys(), portions.values(), color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
    ax.set_ylabel('Calories')
    ax.set_title('Meal Portion Calories')

    return summary, fig

inputs = [
    gr.Slider(minimum=10, maximum=100, step=1, label="Age"),
    gr.Dropdown(choices=list(le_gender.classes_), label="Gender"),
    gr.Number(label="Current Weight (lbs)"),
    gr.Dropdown(choices=list(le_activity.classes_), label="Physical Activity Level"),
    gr.Slider(minimum=1, maximum=5, step=1, label="Sleep Quality (1=Poor to 5=Excellent)"),
    gr.Slider(minimum=1, maximum=5, step=1, label="Stress Level (1=Low to 5=High)")
]

outputs = [
    gr.Textbox(label="Meal Plan Summary"),
    gr.Plot(label="Meal Portion Bar Chart")
]

demo = gr.Interface(fn=gradio_predict, inputs=inputs, outputs=outputs, title="AI-Powered Meal Portion Guide")
demo.launch()
