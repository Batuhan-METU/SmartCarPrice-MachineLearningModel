import joblib
import pandas as pd

def load_model(path):
    data = joblib.load(path)
    model = data["model"]
    transformer = data["transformer"]
    return model, transformer


def predict_price(model, transformer, input_dict):
    df = pd.DataFrame([input_dict])
    X = transformer.transform(df)
    pred = model.predict(X)[0]
    return float(pred)