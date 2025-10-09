import gradio as gr
import joblib

pipe = joblib.load("./Model/drug_pipeline.joblib")

def predict_drug(age, gender, blood_pressure, cholesterol, na_to_k_ratio):
    features = [age, gender, blood_pressure, cholesterol, na_to_k_ratio]
    predicted_drug = pipe.predict([features])[0]
    return f"Predicted Drug: {predicted_drug}"

inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Gender"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K"),
]

examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8],
    [50, "M", "HIGH", "HIGH", 34],
]

gr.Interface(
    fn=predict_drug,
    inputs=inputs,
    outputs="text",
    examples=examples,
    title="Drug Classification",
    description="Predict the appropriate drug based on patient information.",
    theme=gr.themes.Soft()
).launch()
