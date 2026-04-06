import streamlit as st
import numpy as np
import pandas as pd
import joblib
from utils import *

st.set_page_config(page_title="Smart Farming AI 🌱", layout="centered")

st.title("🌱 Smart Farming Anomaly Detection")
st.markdown("AI-powered crop monitoring using sensor data")

# ===============================
# LOAD EVERYTHING
# ===============================
@st.cache_resource
def load_all():

    encoders = joblib.load("model/encoders.pkl")

    df = pd.read_csv("data/Smart_Farming_Crop_Yield_2024.csv")

    # 🔥 FIX: encode dataset
    for col in ['region', 'crop_type']:
        df[col] = encoders[col].transform(df[col])

    if_models = joblib.load("model/if_models.pkl")
    lof_models = joblib.load("model/lof_models.pkl")
    scalers = joblib.load("model/scalers.pkl")

    scaler_if = joblib.load("model/scaler_if.pkl")
    scaler_lof = joblib.load("model/scaler_lof.pkl")
    scaler_rule = joblib.load("model/scaler_rule.pkl")

    weights = joblib.load("model/weights.pkl")
    best_thresh = joblib.load("model/threshold.pkl")

    return df, if_models, lof_models, scalers, scaler_if, scaler_lof, scaler_rule, weights, best_thresh, encoders


(df, if_models, lof_models, scalers,
 scaler_if, scaler_lof, scaler_rule,
 weights, best_thresh, encoders) = load_all()

# ===============================
# INPUT UI
# ===============================
st.subheader("📥 Enter Farm Details")

mode = st.radio("Select Input Mode", ["Slider", "Manual"])

col1, col2 = st.columns(2)

with col1:
    region = st.selectbox("🌍 Region", encoders['region'].classes_)
    crop = st.selectbox("🌾 Crop Type", encoders['crop_type'].classes_)

with col2:
    if mode == "Manual":
        soil_moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0, 40.0)
        soil_pH = st.number_input("Soil pH", 0.0, 14.0, 6.5)
        temperature = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 50.0)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
        ndvi = st.number_input("NDVI Index", 0.0, 1.0, 0.5)
    else:
        soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 40)
        soil_pH = st.slider("Soil pH", 0.0, 14.0, 6.5)
        temperature = st.slider("Temperature (°C)", -10, 60, 25)
        rainfall = st.slider("Rainfall (mm)", 0, 500, 50)
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        ndvi = st.slider("NDVI Index", 0.0, 1.0, 0.5)

# ===============================
# PREDICT
# ===============================
if st.button("🔍 Analyze"):

    user_data = {
        'region': region,
        'crop_type': crop,
        'soil_moisture_%': soil_moisture,
        'soil_pH': soil_pH,
        'temperature_C': temperature,
        'rainfall_mm': rainfall,
        'humidity_%': humidity,
        'NDVI_index': ndvi
    }

    result = predict_user_input(
        user_data,
        if_models,
        lof_models,
        None,
        scalers,
        scaler_if,
        scaler_lof,
        None,
        scaler_rule,
        weights,
        best_thresh,
        encoders,
        df
    )

    if "error" in result:
        st.error(result["error"])

    else:
        st.subheader("📊 Prediction Result")

        if result["prediction"] == "ANOMALY":
            st.error("⚠️ Anomaly Detected!")
        elif result["prediction"] == "TENDENCY":
            st.warning("🚨 Tendency towards anomaly")
        else:
            st.success("✅ Conditions Normal")

        # ===============================
        # ANALYSIS
        # ===============================
        st.subheader("🔍 Analysis")

        if result["parameter_issues"]:
            st.markdown("### 📊 Parameter Issues")
            for i in result["parameter_issues"]:
                st.write("•", i)

        if result["sensor_issues"]:
            st.markdown("### 🔗 Sensor Issues")
            for i in result["sensor_issues"]:
                st.write("•", i)

        if not result["parameter_issues"] and not result["sensor_issues"]:
            st.success("🌱 All conditions are optimal")

        # ===============================
        # 🔥 RECOMMENDATIONS
        # ===============================
        st.subheader("💡 Smart Recommendations")

        for rec in result["recommendations"]:
            st.write("👉", rec)

# ===============================
# FOOTER
# ===============================
st.markdown("---")

