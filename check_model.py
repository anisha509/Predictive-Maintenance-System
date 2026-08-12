import joblib

model_data = joblib.load("predictive_maintenance_model.pkl")

model = model_data["model"]

print("Model details:")
print(model_data)

print("\nTarget classes:")
print(model.classes_)