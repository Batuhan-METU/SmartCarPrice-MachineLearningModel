from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os

def train_random_forest(X_train, y_train): 
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def tune_hyperparameters(X_train, y_train):
    params = {
        "criterion": ["squared_error", "friedman_mse", "poisson", "absolute_error"],
        "max_depth": [10, 15, 20],
        "max_features": ["log2", "sqrt"],
    }

    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=15),
        cv=5,
        param_distributions=params,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    return {"R2": r2, "MAE": mae, "MSE": mse}


def save_model(model, transformer, path):
    ##Save model and transformer together in one file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(
        {"model": model, "transformer": transformer},
        path
    )
    print(f"✅ Model and transformer saved successfully at: {path}")
