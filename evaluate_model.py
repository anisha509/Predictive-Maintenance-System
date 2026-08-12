import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

dataset_path = r"C:\Users\Lenovo\ai4i2020.csv"

df = pd.read_csv(dataset_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. CREATE TEMP DIFFERENCE
# ============================================================

df["Temp Difference"] = (
    df["Process temperature [K]"]
    - df["Air temperature [K]"]
)


# ============================================================
# 3. ENCODE MACHINE TYPE
# ============================================================

type_mapping = {
    "H": 0,
    "L": 1,
    "M": 2
}

df["Type"] = df["Type"].map(type_mapping)


# ============================================================
# 4. SELECT FEATURES
# ============================================================

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp Difference"
]

X = df[features]

y = df["Machine failure"]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. LOAD TRAINED MODEL
# ============================================================

model_data = joblib.load(
    r"C:\Users\Lenovo\OneDrive\Desktop\Predictive_Maintenance\predictive_maintenance_model.pkl"
)

model = model_data["model"]


# ============================================================
# 7. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 8. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score  : {f1:.4f} ({f1 * 100:.2f}%)")


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Machine Failure",
            "Machine Failure"
        ],
        zero_division=0
    )
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("\nConfusion Matrix Interpretation:")
print("TN =", cm[0][0])
print("FP =", cm[0][1])
print("FN =", cm[1][0])
print("TP =", cm[1][1])