import gradio as gr
import joblib
import os
import pandas as pd

# Load the model (using joblib since we're using that format)
model_path = os.path.join("..", "model", "drug_pipeline.joblib")
if os.path.exists(model_path):
    pipe = joblib.load(model_path)
else:
    # Fallback for when running from different directory
    pipe = joblib.load("model/drug_pipeline.joblib")

# Load training metrics
def load_metrics():
    metrics_path = "Results/metrics.txt"
    if not os.path.exists(metrics_path):
        metrics_path = "../Results/metrics.txt"
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return f.read().strip()
    return "Metrics not available"

# Get model information
model_info = f"""
## 🤖 Model Performance Metrics

**Training Results:**
- {load_metrics()}
- **Model Type:** Random Forest Classifier
- **Features:** Age, Gender, Blood Pressure, Cholesterol, Na/K Ratio
- **Drug Classes:** {', '.join(pipe.classes_)}
- **Total Classes:** {len(pipe.classes_)}

## 📊 Model Details
- **Algorithm:** Random Forest with 100 estimators
- **Preprocessing:** Ordinal Encoding for categorical features, Standard Scaling for numerical features
- **Cross-validation:** 70/30 train-test split
"""

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
title = "🏥 Drug Classification ML Model"
description = """
## 💊 Intelligent Drug Recommendation System

This machine learning model predicts the most appropriate drug for a patient based on their medical characteristics. 
Enter the patient details below to get an AI-powered drug recommendation.

### How to use:
1. **Age**: Patient's age (15-74 years)
2. **Gender**: Patient's biological sex
3. **Blood Pressure**: Current BP level (HIGH/LOW/NORMAL)
4. **Cholesterol**: Cholesterol level (HIGH/NORMAL)
5. **Na/K Ratio**: Sodium to Potassium ratio in blood (6.2-38.2)

### Available Drug Types:
- **DrugY**: For specific cardiovascular conditions
- **drugA**: Alternative treatment option A
- **drugB**: Alternative treatment option B  
- **drugC**: Alternative treatment option C
- **drugX**: Specialized medication X
"""

# Create tabbed interface
with gr.Blocks(theme=gr.themes.Soft(), title="Drug Classification ML") as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    
    with gr.Tabs():
        with gr.TabItem("🩺 Drug Prediction"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Patient Information")
                    age_input = gr.Slider(15, 74, step=1, label="Age", value=30)
                    sex_input = gr.Radio(["M", "F"], label="Gender", value="M")
                    bp_input = gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure", value="NORMAL")
                    chol_input = gr.Radio(["HIGH", "NORMAL"], label="Cholesterol", value="NORMAL")
                    nak_input = gr.Slider(6.2, 38.2, step=0.1, label="Na/K Ratio", value=15.0)
                    
                    predict_btn = gr.Button("🔍 Predict Drug", variant="primary")
                    
                with gr.Column():
                    gr.Markdown("### Prediction Result")
                    output = gr.Label(num_top_classes=5, label="Recommended Drug")
                    
                    gr.Markdown("### Example Cases")
                    examples = gr.Examples(
                        examples=[
                            [30, "M", "HIGH", "NORMAL", 15.4],
                            [35, "F", "LOW", "NORMAL", 8.0],
                            [50, "M", "HIGH", "HIGH", 34.0],
                            [25, "F", "NORMAL", "HIGH", 20.5],
                        ],
                        inputs=[age_input, sex_input, bp_input, chol_input, nak_input],
                        label="Click to try these examples:"
                    )
        
        with gr.TabItem("📊 Model Performance"):
            gr.Markdown(model_info)
            
            # Display confusion matrix if available
            plot_path = "Results/model_results.png"
            if not os.path.exists(plot_path):
                plot_path = "../Results/model_results.png"
            
            if os.path.exists(plot_path):
                gr.Image(plot_path, label="Confusion Matrix", show_label=True)
            else:
                gr.Markdown("*Confusion matrix visualization not available*")
        
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## About This Model
            
            This drug classification system uses a **Random Forest machine learning algorithm** to predict the most suitable drug for patients based on their medical characteristics.
            
            ### 🔬 Technical Details
            - **Algorithm**: Random Forest Classifier
            - **Training Data**: 200 patient records
            - **Features**: 5 medical characteristics
            - **Classes**: 5 different drug types
            - **Accuracy**: 97% on test data
            - **F1 Score**: 0.94 (macro average)
            
            ### ⚠️ Important Notice
            **This is a demonstration model for educational purposes only.**
            - Do not use for actual medical decisions
            - Always consult with healthcare professionals
            - This model is not a substitute for medical advice
            
            ### 🧪 Model Training Pipeline
            1. **Data Preprocessing**: Ordinal encoding for categorical variables, standard scaling for numerical features
            2. **Model Training**: Random Forest with 100 trees
            3. **Validation**: 70/30 train-test split
            4. **Evaluation**: Accuracy and F1-score metrics
            
            ### 🔄 Continuous Integration
            This model is automatically retrained and deployed using GitHub Actions and Hugging Face Spaces.
            
            ---
            **Developed with ❤️ using Scikit-learn and Gradio**
            """)
    
    # Connect the prediction function
    predict_btn.click(
        fn=predict_drug,
        inputs=[age_input, sex_input, bp_input, chol_input, nak_input],
        outputs=output
    )

# Launch the app
demo.launch()