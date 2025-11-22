# train_pipeline.py

import os
import json
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.data_processing import load_data, preprocess_data
from src.model_training import tune_hyperparameters, save_model

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_PATH = "data/17-cardekho.csv"
MODEL_PATH = "models/car_price_model.pkl"
DEBUG_MODE = True  # ⚡️ True: quick test, False: full hyperparameter tuning


def main():
    print("🚀 Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"✅ Data loaded successfully! Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("⚙️ Preprocessing data...")
    X_train, X_test, y_train, y_test, transformer, encoded_cols = preprocess_data(df)
    print("✅ Preprocessing complete.")

    # -------------------------------------------------
    # MODEL TRAINING
    # -------------------------------------------------
    if DEBUG_MODE:
        print("🧪 DEBUG MODE: Using default RandomForestRegressor (no tuning)...")
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        best_params = model.get_params()
    else:
        print("🧠 Training model with hyperparameter tuning...")
        model, best_params = tune_hyperparameters(X_train, y_train)

    # -------------------------------------------------
    # EVALUATION
    # -------------------------------------------------
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("\n✅ MODEL PERFORMANCE:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE: {mae:,.2f}")
    print(f"   MSE: {mse:,.2f}\n")

    # -------------------------------------------------
    # SAVE MODEL + METADATA
    # -------------------------------------------------
    print("💾 Saving model and metadata...")
    save_model(model, transformer, MODEL_PATH)
    os.makedirs("models", exist_ok=True)

    sample_n = min(200, X_test.shape[0])
    X_test_sample = X_test[:sample_n]
    y_test_sample = y_test[:sample_n]

    # Convert sparse matrix to DataFrame if needed
    if hasattr(X_test_sample, "toarray"):
        X_test_sample = pd.DataFrame(
            X_test_sample.toarray(),
            columns=transformer.get_feature_names_out()
        )

    # Convert sparse to dense if needed
    X_test_trans = X_test_sample
    if hasattr(X_test_trans, "toarray"):
        X_test_trans = X_test_trans.toarray()

    feature_names = transformer.get_feature_names_out().tolist()
    X_test_trans_df = pd.DataFrame(X_test_trans, columns=feature_names)

    X_test_sample.to_parquet("models/eval_X_raw.parquet")
    y_test_sample.to_frame("price").to_parquet("models/eval_y.parquet")
    X_test_trans_df.to_parquet("models/eval_X_trans.parquet")

    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=4)

    with open("models/metadata.json", "w") as f:
        json.dump({
            "r2": r2,
            "mae": mae,
            "mse": mse,
            "best_params": best_params
        }, f, indent=4)

    print("✅ All model artifacts saved under 'models/'.")
    print("🏁 Pipeline completed successfully.")


if __name__ == "__main__":
    main()
