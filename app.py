import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from utils import *

st.set_page_config(page_title="Smart Farming AI", layout="centered")

st.title("🌱 Smart Farming Anomaly Detection")
st.markdown("AI-powered crop monitoring using sensor data")

# ===============================
# LOAD DATA + MODELS
# ===============================
@st.cache_resource
def load_all():

    df = pd.read_csv("data/dataset.csv")

    if_models = joblib.load("model/if_models.pkl")
    lof_models = joblib.load("model/lof_models.pkl")
    scalers = joblib.load("model/scalers.pkl")

    scaler_if = joblib.load("model/scaler_if.pkl")
    scaler_lof = joblib.load("model/scaler_lof.pkl")
    scaler_ae = joblib.load("model/scaler_ae.pkl")
    scaler_rule = joblib.load("model/scaler_rule.pkl")

    weights = joblib.load("model/weights.pkl")
    best_thresh = joblib.load("model/threshold.pkl")
    encoders = joblib.load("model/encoders.pkl")

    autoencoders = {}
    for key in if_models.keys():
        crop, region = key
        path = f"model/ae_models/ae_{crop}_{region}.h5"
        autoencoders[key] = tf.keras.models.load_model(path)

    return df, if_models, lof_models, autoencoders, scalers, scaler_if, scaler_lof, scaler_ae, scaler_rule, weights, best_thresh, encoders


(df, if_models, lof_models, autoencoders, scalers,
 scaler_if, scaler_lof, scaler_ae, scaler_rule,
 weights, best_thresh, encoders) = load_all()

# ===============================
# INPUT UI
# ===============================
region = st.selectbox("Region", encoders['region'].classes_)
crop = st.selectbox("Crop Type", encoders['crop_type'].classes_)

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
        autoencoders,
        scalers,
        scaler_if,
        scaler_lof,
        scaler_ae,
        scaler_rule,
        weights,
        best_thresh,
        encoders,
        df
    )

    if "error" in result:
        st.error(result["error"])
    else:
        st.subheader("📊 Scores")
        st.json(result["scores"])

        st.subheader("🔥 Final Score")
        st.write(result["final_score"])

        if result["prediction"] == "ANOMALY":
            st.error("⚠️ Anomaly Detected!")
        elif result["prediction"] == "TENDENCY":
            st.warning("🚨 Tendency towards anomaly")
        else:
            st.success("✅ Conditions Normal")

        # Explainability
        st.subheader("🔍 Analysis")

        if result["parameter_issues"]:
            st.write("📊 Parameter Issues:")
            for i in result["parameter_issues"]:
                st.write("-", i)

        if result["sensor_issues"]:
            st.write("🔗 Sensor Issues:")
            for i in result["sensor_issues"]:
                st.write("-", i)
