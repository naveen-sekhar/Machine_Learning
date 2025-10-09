---
title: Drug Classification ML Model
emoji: 💊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.16.0
app_file: app/drug_app.py
pinned: false
license: apache-2.0
short_description: AI-powered drug recommendation system using Random Forest ML
---

# 🏥 Drug Classification ML Model

An intelligent machine learning system that predicts the most appropriate drug for patients based on their medical characteristics.

## 🎯 Model Performance
- **Accuracy**: 97%
- **F1 Score**: 0.94
- **Algorithm**: Random Forest Classifier
- **Training Data**: 200 patient records

## 🔬 Features Used
1. **Age**: Patient's age (15-74 years)
2. **Gender**: Biological sex (M/F)
3. **Blood Pressure**: Current BP level (HIGH/LOW/NORMAL)
4. **Cholesterol**: Cholesterol level (HIGH/NORMAL)
5. **Na/K Ratio**: Sodium to Potassium ratio in blood (6.2-38.2)

## 💊 Drug Classes
- **DrugY**: For specific cardiovascular conditions
- **drugA**: Alternative treatment option A
- **drugB**: Alternative treatment option B
- **drugC**: Alternative treatment option C
- **drugX**: Specialized medication X

## ⚠️ Disclaimer
This is a demonstration model for educational purposes only. Do not use for actual medical decisions. Always consult with healthcare professionals.

## 🚀 Technical Stack
- **ML Framework**: Scikit-learn
- **Web Interface**: Gradio
- **Deployment**: Hugging Face Spaces
- **CI/CD**: GitHub Actions