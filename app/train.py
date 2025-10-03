# train.py (minimal)
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
drug_df = pd.read_csv("S:\Studies\Theory\ML\Module-5\ML\Machine_Learning\data\drug200.csv")
drug_df = drug_df.sample(frac=1)
drug_df.head(3)
from sklearn.model_selection import train_test_split
X = drug_df.drop("Drug", axis=1).values
y = drug_df.Drug.values
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.3, random_state=125
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
cat_col = [1,2,3]
num_col = [0,4]
transform = ColumnTransformer(
 [
 ("encoder", OrdinalEncoder(), cat_col),
 ("num_imputer", SimpleImputer(strategy="median"), num_col),
 ("num_scaler", StandardScaler(), num_col),
 ]
)
pipe = Pipeline(
 steps=[
 ("preprocessing", transform),
 ("model", RandomForestClassifier(n_estimators=100,
random_state=125)),
 ]
)
pipe.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, f1_score
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")
print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:",
round(f1, 2))
with open("S:\Studies\Theory\ML\Module-5\ML\Machine_Learning\Results\metrics.txt", "w") as outfile:
 outfile.write(f"\n Accuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}.")

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=pipe.classes_)
disp.plot()
plt.savefig("S:\Studies\Theory\ML\Module-5\ML\Machine_Learning\Results\model_results.png", dpi=120)
###################
# Save model
joblib.dump(pipe, "S:\Studies\Theory\ML\Module-5\ML\Machine_Learning\model\drug_pipeline.joblib")
print("Training complete — model saved to S:\Studies\Theory\ML\Module-5\ML\Machine_Learning\model\drug_pipeline.joblib")
#import skops.io as sio
#sio.dump(pipe, "model/drug_pipeline.skops")
#sio.load("model/drug_pipeline.skops", trusted=True)
import joblib
# Save pipeline
joblib.dump(pipe, "S:\Studies\Theory\ML\Module-5\ML\Machine_Learning\model\drug_pipeline.joblib")
# Load pipeline
pipe = joblib.load("S:\Studies\Theory\ML\Module-5\ML\Machine_Learning\model\drug_pipeline.joblib")