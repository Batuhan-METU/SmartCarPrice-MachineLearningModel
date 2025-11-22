import streamlit as st

st.set_page_config(page_title = "SmartCarPrice 🏁", page_icon = "🚘", layout = "wide",)

import os
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
import joblib
from PIL import Image



#📁 PATHS

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = (BASE_DIR / "../models").resolve()

CSS_PATH = BASE_DIR / "styles.css"


#CUSTOM STYLING (CSS)

def local_css(path):
    if path.exists():
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ CSS file not found at {path}")

local_css(CSS_PATH)




#LOAD MODEL & METADATA 

@st.cache_resource(show_spinner=False)
def load_model():
    model_path = MODEL_DIR / "car_price_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model_data = joblib.load(model_path)
    return model_data["model"], model_data["transformer"]


@st.cache_data(show_spinner=False)
def load_metadata():
    meta_path = MODEL_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"r2": 0.0, "mae": 0.0, "mse": 0.0}


try:
    model, transformer = load_model()
    metadata = load_metadata()
except Exception as e:
    st.error(f"Model is not loading: {e}")
    st.stop()

# SHAP & LIME HELPER FUNCTIONS 

@st.cache_resource(show_spinner=False)
def compute_shap_values(_model, X_sample):
    try:
        explainer = shap.TreeExplainer(_model)
        shap_values = explainer.shap_values(X_sample)
        return shap_values
    except Exception as exc:
        raise RuntimeError(f"SHAP is not calculating!!: {exc}")


@st.cache_resource(show_spinner=False)
def explain_with_lime(_model, X_train, X_sample, feature_names):
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        verbose=False,
        mode="regression",
    )
    exp = explainer.explain_instance(X_sample, _model.predict, num_features=8)
    return exp



#HERO SECTION
st.markdown("<h1><span>Know Your Car's True Value </span>🚖</h1>", unsafe_allow_html=True)
st.markdown("<h2>Using MachineLearning-powered car price prediction - fast, accurate, data-managed</h2>", unsafe_allow_html=True)
st.markdown("<h3>Enter your car's details below and discover its market value instantly.</h3>", unsafe_allow_html = True)

st.markdown("<hr>", unsafe_allow_html=True)






## INPUT FORM
st.markdown("<h3 style='text-align:center;'>Enter Car Details</h3>", unsafe_allow_html=True)



col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", ["Maruti", "Hyundai", "Honda", "Toyota", "Tata", "Mahindra", "Kia", "Ford", "Volkswagen"])
    car_model = st.text_input("Model", placeholder="e.g. Swift VXI, Creta SX")
    year = st.number_input("Year of Manufacture", min_value=1995, max_value=2025, step=1)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmission_type = st.selectbox("Transmission", ["Manual", "Automatic"])

with col2:
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
    owner_type = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, step=500)
    engine = st.number_input("Engine Capacity (CC)", min_value=600, max_value=5000, step=100)
    seats = st.number_input("Number of Seats", min_value=2, max_value=10, step=1)



## PREDICTION 

if st.button("💰 Estimate My Car's Value"):
    try:
        input_df = pd.DataFrame([
            {
                "year": year,
                "km_driven": km_driven,
                "fuel_type": fuel_type,
                "seller_type": seller_type,
                "transmission_type": transmission_type,
                "owner_type": owner_type,
                "engine": engine,
                "seats": seats,
                "car_name": model,
                "brand": brand,
                "model": car_model
            }
        ])

        transformed = transformer.transform(input_df)
        pred = float(model.predict(transformed)[0])
        st.markdown(
            f"<div class='prediction-box'>Estimated Car Price: ${pred:,.2f}</div>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

### PERFORMANCE METRICS 

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 📊 Model Performance Metrics")
st.caption("<span style='color:#93c5fd;'>Metrics computed on the held-out test split (30%) after hyperparameter tuning.</span>", unsafe_allow_html=True)


c1, c2, c3 = st.columns(3)
c1.metric("R² Score", f"{metadata.get('r2', 0):.3f}")
c2.metric("MAE", f"{metadata.get('mae', 0):,.0f}")
mse_value = float(metadata.get("mse", 0))
rmse_value = math.sqrt(mse_value)
c3.metric("RMSE", f"{rmse_value:,.0f}")

st.info("⚠️ This model is trained about 15.000 data located on cardekho dataset on Kaggle, the model can give different results for other databases.")


### SHAP SECTION 
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("## 🧠 Model Explainability (SHAP Insights)")

@st.cache_data(show_spinner=False)
def load_eval_data():
    X_path = MODEL_DIR / "eval_X_trans.parquet"
    y_path = MODEL_DIR / "eval_y.parquet"
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError("Evaluation parquet files not found.")
    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)
    return X, y

try:
    X_eval, _ = load_eval_data()
    X_sample = X_eval.sample(min(300, len(X_eval)), random_state=42).copy()

    def clean_feature_name(name: str) -> str:
        name = name.replace("onehot_", "")
        name = name.replace("remainder_", "")
        name = name.replace("_fuel_type_", "Fuel Type - ")
        name = name.replace("_max_p", "Max P")
        name = name.replace("_vehicle_age", "Vehicle Age")
        name = name.replace("_engine", "Engine")
        name = name.replace("_km_driven", "Km Driven")
        name = name.replace("_mileage", "MileAge")
        name = name.replace("_transmission_type_", "Transmisson Type - ")
        name = name.replace("_brand_freq", "Brand Freq")
        name = name.replace("_model_freq", "Model Freq")
        name = name.replace("_car_name_freq", "Car Freq")
        name = name.replace("_seller_type_", "Seller Type - ")
        name = name.replace("_seats", "Seats")
        return name

    X_sample.columns = [clean_feature_name(c) for c in X_sample.columns]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    tab_imp, tab_sum = st.tabs(["📊 Feature Importance", "🎨 SHAP Summary Plot"])

    with tab_imp:
        st.markdown("#### 🔹 Average Feature Importance")
        fig1, _ = plt.subplots(figsize=(12, 7))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.xlabel("")
        plt.tight_layout()
        st.pyplot(fig1, clear_figure=True)

    with tab_sum:
        st.markdown("#### 🔹 SHAP Value Distribution")
        fig2, _ = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True)

except Exception as e:
    st.warning(f"SHAP analysis unavailable: {e}")


###LIME SECTION 
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("## 🧩 LIME Explanation (Single Prediction Insight)")
try:
    X_eval, y_eval = load_eval_data()
    feature_names = list(X_eval.columns)
    X_array = np.array(X_eval)

    def clean_feature_name2(name: str) -> str:
        name = name.replace("onehot_", "")
        name = name.replace("remainder_", "")
        name = name.replace("_fuel_type_", "Fuel Type - ")
        name = name.replace("_max_p", "Max P")
        name = name.replace("_vehicle_age", "Vehicle Age")
        name = name.replace("_engine", "Engine")
        name = name.replace("_km_driven", "Km Driven")
        name = name.replace("_mileage", "MileAge")
        name = name.replace("_transmission_type_", "Transmisson Type - ")
        name = name.replace("_brand_freq", "Brand Freq")
        name = name.replace("_model_freq", "Model Freq")
        name = name.replace("_car_name_freq", "Car Freq")
        name = name.replace("_seller_type_", "Seller Type - ")
        name = name.replace("_seats", "Seats")
        return name

    clean_feature_names = [clean_feature_name2(c) for c in feature_names]

    sample_index = st.slider("🔍 Choose a flight sample for explanation", 0, len(X_eval) - 1, 10)
    X_instance = X_array[sample_index]

    lime_exp = explain_with_lime(model, X_eval, X_instance, clean_feature_names)

    st.markdown(
        """
        <div style=" background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 15px; padding: 20px; margin-top: 15px; box-shadow: 0 0 25px rgba(56, 189, 248, 0.2); ">
            <h4 style="color:#93c5fd; text-align:center;">🎨 Understanding the LIME Explanation</h4>
            <ul style="font-size:1.05rem; color:black; list-style:none;">
                <li>🟥 <b>Red bars</b> → Features that <b>increase</b> the predicted flight price.</li>
                <li>🟩 <b>Green bars</b> → Features that <b>decrease</b> the predicted flight price.</li>
                <li>💡 Each bar shows how much that feature pushed the model's decision <b>up or down</b> for this specific flight.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<h4 style='text-align:center;'> 🔹 Local Explanation ==> Car #{sample_index}</h4>", unsafe_allow_html = True)
    fig_lime = lime_exp.as_pyplot_figure()
    plt.tight_layout()
    st.pyplot(fig_lime, clear_figure=True)

except Exception as e:
    st.warning(f"LIME explanation unavailable: {e}")

st.markdown("<hr>", unsafe_allow_html=True)


# REAL VS PREDICTED GRAPH """

try:
    X_eval, y_eval = load_eval_data()
    y_pred_eval = model.predict(X_eval)
    df_eval = pd.DataFrame({"Actual": y_eval["price"].values, "Predicted": y_pred_eval})

    st.markdown("### 📈 Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_eval["Actual"], df_eval["Predicted"], alpha=0.4)
    mn, mx = df_eval["Actual"].min(), df_eval["Actual"].max()
    ax.plot([mn, mx], [mn, mx], linewidth=2)
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("Model Fit Visualization")
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Visualization unavailable: {e}")


#PROJECT OVERVIEW CARDS """


st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;'>💡 Project Overview & Tools Used</h2>", unsafe_allow_html=True)

st.markdown("<div class='cards-grid'>", unsafe_allow_html=True)
colA, colB, colC = st.columns(3)
with colA:
    st.markdown(
        """
        <div class="card">
            <h3>📊 Data Cleaning</h3>
            <p>Used <b>Pandas</b> and <b>NumPy</b> for data preprocessing — handling missing values, encoding categorical variables, and normalizing flight attributes to ensure clean input for the model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with colB:
    st.markdown(
        """
        <div class="card">
            <h3>🧠 Model Training</h3>
            <p>Trained a <b>Random Forest Regressor</b> using <b>Scikit-learn</b> with extensive hyperparameter tuning via <b>RandomizedSearchCV</b>. Achieved reliable accuracy and balanced bias-variance performance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with colC:
    st.markdown(
        """
        <div class="card">
            <h3>🚀 Deployment</h3>
            <p>Deployed interactively using <b>Streamlit</b>. Enhanced with custom CSS animations, gradient styling, and responsive design for a modern, professional ML dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

colD, colE, colF = st.columns(3)
with colD:
    st.markdown(
        """
        <div class="card">
            <h3>🎨 SHAP Explainability</h3>
            <p>Implemented <b>SHAP (SHapley Additive exPlanations)</b> to visualize global feature importance — showing how each variable impacts model predictions across the entire dataset.</p>
            <ul style="color:darkgray;">
                <li>🟦 Blue bars → Stronger impact</li>
                <li>📈 Global view of model behavior</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with colE:
    st.markdown(
        """
        <div class="card">
            <h3>🧩 LIME Interpretation</h3>
            <p>Integrated <b>LIME (Local Interpretable Model-Agnostic Explanations)</b> to explain individual predictions. Helps identify why a specific flight's price was high or low.</p>
            <ul style="color:darkgray;">
                <li>🟥 Red → Increased price</li>
                <li>🟩 Green → Decreased price</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with colF:
    st.markdown(
        """
        <div class="card">
            <h3>📈 Visualization</h3>
            <p>Created high-quality plots using <b>Matplotlib</b> to show <b>Actual vs Predicted</b> values and performance trends. Added dynamic tabs for SHAP and LIME visualizations.</p>
            <p>Visuals are interactive and scrollable for a clear, immersive experience.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


#FOOTER 
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("""
<div class="footer-wrapper">
    <p class="footer-text">
        Made by <b>Batuhan Başoda</b> 🚗<br>
        From Kaggle CarDekho Dataset → Production ML App
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html = True)

st.markdown("""
    <p style="text-align: center; margin-top: 6px;">
        <a href="https://www.kaggle.com/batuhanbasoda" target="_blank">
            <svg width="30" height="30" viewBox="0 0 24 24"> <path fill="currentColor" d="M6.04 6.57v10.86h1.61v-4.72h1.09l3.38 4.72h1.9l-3.76-5.08L14.86 6.57h-1.86l-3.32 4.49H7.65V6.57H6.04Z"/> </svg>
        </a>
        <a href="https://github.com/Batuhan-METU" target="_blank">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="32" style="margin: 0 8px;">
        </a>
        <a href="https://www.linkedin.com/in/batuhan-ba%C5%9Foda-b78799377/" target="_blank">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="32" style="margin: 0 8px;">
        </a>
    </p>
""", unsafe_allow_html = True)

