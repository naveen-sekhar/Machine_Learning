import gradio as gr
import joblib
import os

# Load the model
pipe = joblib.load("model/drug_pipeline.joblib")

def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    """Predict drugs based on patient features."""
    features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]
    predicted_drug = pipe.predict([features])[0]
    return f"Predicted Drug: {predicted_drug}"

# Simple interface for testing
demo = gr.Interface(
    fn=predict_drug,
    inputs=[
        gr.Slider(15, 74, step=1, label="Age", value=30),
        gr.Radio(["M", "F"], label="Gender", value="M"),
        gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure", value="NORMAL"),
        gr.Radio(["HIGH", "NORMAL"], label="Cholesterol", value="NORMAL"),
        gr.Slider(6.2, 38.2, step=0.1, label="Na/K Ratio", value=15.0),
    ],
    outputs=gr.Label(num_top_classes=5),
    title="Drug Classification Test",
    description="Simple test interface for drug prediction",
    examples=[
        [30, "M", "HIGH", "NORMAL", 15.4],
        [35, "F", "LOW", "NORMAL", 8.0],
    ]
)

if __name__ == "__main__":
    demo.launch()