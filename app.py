import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib

import os
import streamlit as st  # if not already imported


# Load model
model = joblib.load("model.pkl")

# Manual encoding for gender and activity
gender_map = {'Male': 0, 'Female': 1}
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

def gradio_predict(age, gender, weight, activity, sleep, stress):
    gender_enc = gender_map.get(gender, 0)
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
    gr.Dropdown(choices=["Male", "Female"], label="Gender"),
    gr.Number(label="Current Weight (lbs)"),
    gr.Dropdown(choices=["Sedentary", "Lightly Active", "Moderately Active", "Very Active"], label="Physical Activity Level"),
    gr.Slider(minimum=1, maximum=5, step=1, label="Sleep Quality (1=Poor to 5=Excellent)"),
    gr.Slider(minimum=1, maximum=5, step=1, label="Stress Level (1=Low to 5=High)")
]

outputs = [
    gr.Textbox(label="Meal Plan Summary"),
    gr.Plot(label="Meal Portion Bar Chart")
]

demo = gr.Interface(fn=gradio_predict, inputs=inputs, outputs=outputs, title="AI-Powered Meal Portion Guide")
demo.launch()
