import joblib

data = joblib.load("models/car_price_model.pkl")

model = data.get("model")
transformer = data.get("transformer")

print("Model type:", type(model))
print("Transformer type:", type(transformer))

if hasattr(model, "predict"):
    print("✅ Model is callable and ready for SHAP/LIME")
else:
    print("❌ Model is not callable — it’s probably a string.")