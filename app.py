import streamlit as st
import numpy as np
import pandas as pd
import joblib
from utils import *

st.set_page_config(page_title="Smart Farming AI", layout="centered")

# ===============================
# 🔥 CUSTOM UI STYLING
# ===============================
st.markdown("""
<style>
.big-title {
    font-size: 32px;
    font-weight: bold;
    color: #2e7d32;
}
.card {
    padding: 15px;
    border-radius: 10px;
    background-color: #f5f5f5;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🌱 Smart Farming AI</div>', unsafe_allow_html=True)
st.write("AI-powered anomaly detection using sensor data")

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_all():
    df = pd.read_csv("data/Smart_Farming_Crop_Yield_2024.csv")

    if_models = joblib.load("model/if_models.pkl")
    lof_models = joblib.load("model/lof_models.pkl")
    scalers = joblib.load("model/scalers.pkl")

    scaler_if = joblib.load("model/scaler_if.pkl")
    scaler_lof = joblib.load("model/scaler_lof.pkl")
    scaler_rule = joblib.load("model/scaler_rule.pkl")

    weights = joblib.load("model/weights.pkl")
    best_thresh = joblib.load("model/threshold.pkl")
    encoders = joblib.load("model/encoders.pkl")

    return df, if_models, lof_models, scalers, scaler_if, scaler_lof, scaler_rule, weights, best_thresh, encoders


(df, if_models, lof_models, scalers,
 scaler_if, scaler_lof, scaler_rule,
 weights, best_thresh, encoders) = load_all()

# ===============================
# INPUT MODE (SLIDER / MANUAL)
# ===============================
mode = st.radio("Choose Input Mode", ["Slider", "Manual Input"])

region = st.selectbox("Region", encoders['region'].classes_)
crop = st.selectbox("Crop Type", encoders['crop_type'].classes_)

if mode == "Slider":
    soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 40)
    soil_pH = st.slider("Soil pH", 0.0, 14.0, 6.5)
    temperature = st.slider("Temperature (°C)", -10, 60, 25)
    rainfall = st.slider("Rainfall (mm)", 0, 500, 50)
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    ndvi = st.slider("NDVI Index", 0.0, 1.0, 0.5)

else:
    soil_moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0, 40.0)
    soil_pH = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    temperature = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 50.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    ndvi = st.number_input("NDVI Index", 0.0, 1.0, 0.5)

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

        # ===============================
        # OUTPUT SAME AS YOUR ORIGINAL CODE
        # ===============================
        st.markdown("### 📊 Scores")
        st.write(
            f"IF: {result['scores']['if']} | "
            f"LOF: {result['scores']['lof']} | "
            f"AE: {result['scores']['ae']} | "
            f"Rule: {result['scores']['rule']}"
        )

        st.markdown("### 🔥 Final Score")
        st.write(result["final_score"])

        # Decision
        if result["prediction"] == "ANOMALY":
            st.error("⚠️ ANOMALY DETECTED!")
        elif result["prediction"] == "TENDENCY":
            st.warning("🚨 Tendency towards ANOMALY!")
        else:
            st.success("✅ Conditions are NORMAL")

        # ===============================
        # DETAILED ANALYSIS (SAME STYLE)
        # ===============================
        st.markdown("### 🔍 Detailed Analysis")

        if result["parameter_issues"]:
            st.markdown("**📊 Parameter Issues:**")
            for i in result["parameter_issues"]:
                st.write("-", i)

        if result["sensor_issues"]:
            st.markdown("**🔗 Sensor Issues:**")
            for i in result["sensor_issues"]:
                st.write("-", i)

        if not result["parameter_issues"] and not result["sensor_issues"]:
            st.success("🌱 All conditions are optimal")
