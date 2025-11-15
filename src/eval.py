import pandas as pd
import joblib
from sklearn.metrics import classification_report

# --------------------------
# 1. Load Model
# --------------------------
model_path = "models/best_model.joblib"
clf = joblib.load(model_path)
print("Loaded model:", model_path)

# --------------------------
# 2. Load Test Data
# --------------------------
test_path = "data/UNSW_NB15_testing-set.csv"
test_df = pd.read_csv(test_path)

X_test = test_df.drop(columns=["label", "attack_cat"])
y_test = test_df["label"]

# --------------------------
# 3. Predict
# --------------------------
print("Running predictions...")
y_pred = clf.predict(X_test)

# --------------------------
# 4. Report
# --------------------------
print("\nTEST SET PERFORMANCE:")
print(classification_report(y_test, y_pred))
