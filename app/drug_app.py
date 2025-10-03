import gradio as gr
import joblib
import os

# Load the model (using joblib since we're using that format)
model_path = os.path.join("..", "model", "drug_pipeline.joblib")
if os.path.exists(model_path):
    pipe = joblib.load(model_path)
else:
    # Fallback for when running from different directory
    pipe = joblib.load("model/drug_pipeline.joblib")

def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    """Predict drugs based on patient features.
    
    Args:
        age (int): Age of patient
        sex (str): Gender of patient
        blood_pressure (str): Blood pressure level
        cholesterol (str): Cholesterol level
        na_to_k_ratio (float): Ratio of sodium to potassium in blood
    
    Returns:
        str: Predicted drug label
    """
    features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]
    predicted_drug = pipe.predict([features])[0]
    label = f"Predicted Drug: {predicted_drug}"
    return label

# Define inputs
inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Gender"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K"),
]

# Define outputs
outputs = [gr.Label(num_top_classes=5)]

# Example inputs for testing
examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8],
    [50, "M", "HIGH", "HIGH", 34],
]

# App metadata
title = "Drug Classification"
description = "Enter the patient details to correctly identify the appropriate drug type. This ML model predicts the most suitable drug based on age, gender, blood pressure, cholesterol level, and sodium-to-potassium ratio."

# Create and launch the Gradio interface
gr.Interface(
    fn=predict_drug,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    theme=gr.themes.Soft(),
).launch()