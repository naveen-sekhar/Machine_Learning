# train.py - Fixed for CI/CD compatibility
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# Use relative paths that work in CI environment
data_path = os.path.join("data", "drug200.csv")
model_path = os.path.join("model", "drug_pipeline.joblib")
metrics_path = os.path.join("Results", "metrics.txt")
plot_path = os.path.join("Results", "model_results.png")

# Load and prepare data
print("Loading data...")
drug_df = pd.read_csv(data_path)
drug_df = drug_df.sample(frac=1, random_state=42)  # Added random_state for reproducibility

# Prepare features and target
X = drug_df.drop("Drug", axis=1).values
y = drug_df.Drug.values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

# Define preprocessing pipeline
cat_col = [1, 2, 3]  # Categorical columns
num_col = [0, 4]     # Numerical columns

transform = ColumnTransformer([
    ("encoder", OrdinalEncoder(), cat_col),
    ("num_imputer", SimpleImputer(strategy="median"), num_col),
    ("num_scaler", StandardScaler(), num_col),
])

# Create complete pipeline
pipe = Pipeline(steps=[
    ("preprocessing", transform),
    ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
])

print("Training model...")
pipe.fit(X_train, y_train)

# Make predictions
predictions = pipe.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print(f"Accuracy: {round(accuracy, 2) * 100}%, F1: {round(f1, 2)}")

# Save metrics
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
with open(metrics_path, "w") as outfile:
    outfile.write(f"Accuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}")

# Create and save confusion matrix plot
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.title("Drug Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()

# Save model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(pipe, model_path)
print(f"Training complete — model saved to {model_path}")

print("Training pipeline completed successfully!")