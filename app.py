import streamlit as st
import numpy as np
import pandas as pd
import joblib
from utils import *

st.set_page_config(page_title="Smart Farming AI 🌱", layout="centered")

st.title("🌱 Smart Farming Anomaly Detection")
st.markdown("AI-powered crop monitoring")

# ===============================
# LOAD
# ===============================
@st.cache_resource
def load_all():

    df = pd.read_csv("data/Smart_Farming_Crop_Yield_2024.csv")

    # Normalize columns
    df.columns = df.columns.str.strip()

    df.rename(columns={
        'Crop Type': 'crop_type',
        'Region': 'region',
        'Soil Moisture (%)': 'soil_moisture_%',
        'Soil pH': 'soil_pH',
        'Temperature(C)': 'temperature_C',
        'Rainfall (mm)': 'rainfall_mm',
        'Humidity (%)': 'humidity_%'
    }, inplace=True)

    if_models = joblib.load("model/if_models.pkl")
    lof_models = joblib.load("model/lof_models.pkl")

    scaler_if = joblib.load("model/scaler_if.pkl")
    scaler_lof = joblib.load("model/scaler_lof.pkl")
    scaler_rule = joblib.load("model/scaler_rule.pkl")

    weights = joblib.load("model/weights.pkl")
    best_thresh = joblib.load("model/threshold.pkl")
    encoders = joblib.load("model/encoders.pkl")

    return df, if_models, lof_models, scaler_if, scaler_lof, scaler_rule, weights, best_thresh, encoders


df, if_models, lof_models, scaler_if, scaler_lof, scaler_rule, weights, best_thresh, encoders = load_all()

# ===============================
# INPUT UI
# ===============================
st.subheader("📥 Enter Farm Details")

mode = st.radio("Input Mode", ["Manual", "Slider"])

col1, col2 = st.columns(2)

with col1:
    region = st.selectbox("🌍 Region", encoders['region'].classes_)
    crop = st.selectbox("🌾 Crop Type", encoders['crop_type'].classes_)

with col2:
    if mode == "Manual":
        sm = st.number_input("Soil Moisture (%)", 0.0, 100.0, 40.0)
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
        temp = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0)
        rain = st.number_input("Rainfall (mm)", 0.0, 500.0, 50.0)
        hum = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
        ndvi = st.number_input("NDVI Index", 0.0, 1.0, 0.5)
    else:
        sm = st.slider("Soil Moisture (%)", 0, 100, 40)
        ph = st.slider("Soil pH", 0.0, 14.0, 6.5)
        temp = st.slider("Temperature (°C)", -10, 60, 25)
        rain = st.slider("Rainfall (mm)", 0, 500, 50)
        hum = st.slider("Humidity (%)", 0, 100, 60)
        ndvi = st.slider("NDVI Index", 0.0, 1.0, 0.5)

# ===============================
# PREDICT
# ===============================
if st.button("🔍 Analyze"):

    # ✅ FIXED VARIABLE NAMES HERE
    user_data = {
        'Region': region,
        'Crop Type': crop,
        'Soil Moisture (%)': sm,
        'Soil pH': ph,
        'Temperature(C)': temp,
        'Rainfall (mm)': rain,
        'Humidity (%)': hum,
        'NDVI_index': ndvi
    }

    result = predict_user_input(
        user_data,
        if_models,
        lof_models,
        None,
        None,
        scaler_if,
        scaler_lof,
        None,
        scaler_rule,
        weights,
        best_thresh,
        encoders,
        df
    )

    # ===============================
    # OUTPUT
    # ===============================
    if "error" in result:
        st.error(result["error"])
    else:
        st.subheader("📊 Result")

        if result["prediction"] == "ANOMALY":
            st.error("⚠️ Anomaly Detected")
        elif result["prediction"] == "TENDENCY":
            st.warning("🚨 Tendency towards anomaly")
        else:
            st.success("✅ Normal Conditions")

        st.subheader("🔍 Analysis")

        if result["parameter_issues"]:
            st.markdown("### 📊 Parameter Issues")
            for i in result["parameter_issues"]:
                st.write("•", i)

        if result["sensor_issues"]:
            st.markdown("### 🔗 Sensor Issues")
            for i in result["sensor_issues"]:
                st.write("•", i)

        st.subheader("💡 Recommendations")
        for r in result["recommendations"]:
            st.write("•", r)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Built with ❤️ for Smart Agriculture")
