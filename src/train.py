import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --------------------------
# 1. Load Training Data
# --------------------------
train_path = "data/UNSW_NB15_training-set.csv"
df = pd.read_csv(train_path)

X = df.drop(columns=["label", "attack_cat"])
y = df["label"]

# Identify categorical & numeric columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

# --------------------------
# 2. Preprocessing
# --------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# --------------------------
# 3. Pipeline
# --------------------------
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(random_state=42)),
    ]
)

# --------------------------
# 4. Grid Search Parameters
# --------------------------
param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [10, 20, None],
    "clf__min_samples_split": [2, 5],
    "clf__class_weight": [None, "balanced"],
}

# --------------------------
# 5. Train/Test Split
# --------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# 6. GridSearchCV
# --------------------------
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="f1",
    verbose=2,
    n_jobs=-1
)

print("Training GridSearchCV...")
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)

# --------------------------
# 7. Validation Metrics
# --------------------------
y_pred = grid.predict(X_val)
print("\nValidation Performance:")
print(classification_report(y_val, y_pred))

# --------------------------
# 8. Save the best model
# --------------------------
joblib.dump(grid.best_estimator_, "models/best_model.joblib")
print("\nModel saved to models/best_model.joblib")
