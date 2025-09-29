import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# Ensure Results folder exists
os.makedirs("Results", exist_ok=True)

# Load and shuffle dataset
drug_df = pd.read_csv("data/drug200.csv")
drug_df = drug_df.sample(frac=1, random_state=42)
print(drug_df.head(3))

# Split features and target
X = drug_df.drop("Drug", axis=1).values
y = drug_df["Drug"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

# Define column indices
cat_col = [1, 2, 3]
num_col = [0, 4]

# Build preprocessing pipeline
transform = ColumnTransformer([
    ("encoder", OrdinalEncoder(), cat_col),
    ("num_imputer", SimpleImputer(strategy="median"), num_col),
    ("num_scaler", StandardScaler(), num_col),
])

pipe = Pipeline([
    ("preprocessing", transform),
    ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
])

# Train model
pipe.fit(X_train, y_train)

# Evaluate model
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")
print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))

# Save metrics
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}.")

# Confusion matrix
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)
plt.close()




joblib.dump(pipe, "model/drug_pipeline.joblib")
joblib.load("model/drug_pipeline.joblib")