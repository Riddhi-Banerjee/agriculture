import streamlit as st
import numpy as np
import pandas as pd
import joblib
from utils import *

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Smart Farming AI 🌱",
    layout="wide",
    page_icon="🌱"
)

# ===============================
# CUSTOM CSS (🔥 PREMIUM UI)
# ===============================
st.markdown("""
<style>

body {
    background-color: #0e1117;
}

.main {
    background-color: #0e1117;
}

h1, h2, h3 {
    color: #00FFAA;
}

.stButton>button {
    background: linear-gradient(90deg, #00FFAA, #00C2FF);
    color: black;
    border-radius: 10px;
    font-weight: bold;
    height: 3em;
    width: 100%;
}

.block-container {
    padding-top: 2rem;
}

.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,255,170,0.2);
    margin-bottom: 20px;
}

.metric-box {
    background: linear-gradient(135deg, #00FFAA20, #00C2FF20);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# TITLE
# ===============================
st.title("🌱 Smart Farming AI Dashboard")
st.caption("AI-powered anomaly detection + intelligent recommendations")

# ===============================
# LOAD DATA
# ===============================
@st.cache_resource
def load_all():

    df = pd.read_csv("data/Smart_Farming_Crop_Yield_2024.csv")
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
# INPUT SECTION
# ===============================
st.markdown("## 📥 Enter Farm Data")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    region = st.selectbox("🌍 Region", encoders['region'].classes_)
    crop = st.selectbox("🌾 Crop Type", encoders['crop_type'].classes_)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    sm = st.slider("Soil Moisture (%)", 0, 100, 40)
    ph = st.slider("Soil pH", 0.0, 14.0, 6.5)
    temp = st.slider("Temperature (°C)", -10, 60, 25)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    rain = st.slider("Rainfall (mm)", 0, 500, 50)
    hum = st.slider("Humidity (%)", 0, 100, 60)
    ndvi = st.slider("NDVI Index", 0.0, 1.0, 0.5)
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# BUTTON
# ===============================
if st.button("🔍 Analyze Farm Conditions"):

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
    # RESULT SECTION
    # ===============================
    st.markdown("## 📊 Result")

    if result["prediction"] == "ANOMALY":
        st.error("⚠️ High Risk Detected")
    elif result["prediction"] == "TENDENCY":
        st.warning("⚠️ Moderate Risk")
    else:
        st.success("✅ Healthy Conditions")

    # ===============================
    # ANALYSIS SECTION
    # ===============================
    colA, colB = st.columns(2)

    with colA:
        st.markdown("### 📊 Parameter Issues")
        if result["parameter_issues"]:
            for i in result["parameter_issues"]:
                st.markdown(f"- {i}")
        else:
            st.success("No parameter issues detected")

    with colB:
        st.markdown("### 🔗 Sensor Issues")
        if result["sensor_issues"]:
            for i in result["sensor_issues"]:
                st.markdown(f"- {i}")
        else:
            st.success("Sensors look consistent")

    # ===============================
    # RECOMMENDATIONS
    # ===============================
    st.markdown("### 💡 Smart Recommendations")

    for r in result["recommendations"]:
        st.markdown(f"👉 {r}")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("🚀 Built with AI for Smart Agriculture | Premium Dashboard UI")
