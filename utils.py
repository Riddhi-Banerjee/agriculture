import numpy as np

# ===============================
# UI → MODEL COLUMN MAPPING
# ===============================
COLUMN_MAP = {
    'Soil Moisture (%)': 'soil_moisture_%',
    'Soil pH': 'soil_pH',
    'Temperature(C)': 'temperature_C',
    'Rainfall (mm)': 'rainfall_mm',
    'Humidity (%)': 'humidity_%',
    'NDVI_index': 'NDVI_index'
}

# ===============================
# ENCODING
# ===============================
def encode_input(value, encoder):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return None

# ===============================
# MODEL SCORES
# ===============================
def get_if_score(model, X):
    return -model.decision_function(X)[0]

def get_lof_score(model, X):
    return -model.decision_function(X)[0]

# ===============================
# RULE-BASED ANOMALY
# ===============================
def detect_rule_anomalies(row):
    issues = 0

    if row['Rainfall (mm)'] > 100 and row['Soil Moisture (%)'] < 30:
        issues += 1

    if row['NDVI_index'] > 0.7 and row['Soil Moisture (%)'] < 25:
        issues += 1

    if row['Temperature(C)'] > 35 and row['Humidity (%)'] > 80:
        issues += 1

    return issues * 2

# ===============================
# PARAMETER ANALYSIS (FIXED)
# ===============================
def detect_problems_dynamic(user_data, df, encoders):

    crop = encode_input(user_data['Crop Type'], encoders['crop_type'])
    region = encode_input(user_data['Region'], encoders['region'])

    key_df = df[(df['crop_type'] == crop) & (df['region'] == region)]

    if len(key_df) == 0:
        return []

    issues = []

    for ui_col, model_col in COLUMN_MAP.items():

        if model_col not in key_df.columns:
            continue

        low = key_df[model_col].quantile(0.10)
        high = key_df[model_col].quantile(0.90)
        val = user_data[ui_col]

        if val < low:
            issues.append(f"{ui_col} LOW (expected > {round(low,2)})")
        elif val > high:
            issues.append(f"{ui_col} HIGH (expected < {round(high,2)})")

    return issues

# ===============================
# SENSOR ISSUES
# ===============================
def detect_sensor_fusion_anomalies(data):

    issues = []

    if data['Rainfall (mm)'] > 100 and data['Soil Moisture (%)'] < 30:
        issues.append("High rainfall but low soil moisture")

    if data['NDVI_index'] > 0.7 and data['Soil Moisture (%)'] < 25:
        issues.append("High NDVI but low soil moisture")

    if data['Temperature(C)'] > 35 and data['Humidity (%)'] > 80:
        issues.append("High temperature with very high humidity")

    if data['Temperature(C)'] > 40 and data['NDVI_index'] > 0.7:
        issues.append("Extreme temperature but high NDVI")

    if data['Rainfall (mm)'] < 10 and data['NDVI_index'] > 0.8:
        issues.append("Low rainfall but high NDVI")

    return issues

# ===============================
# RECOMMENDATIONS ENGINE
# ===============================
def generate_recommendations(user_data):

    recs = []

    if user_data['Soil Moisture (%)'] < 30:
        recs.append("Increase irrigation — soil moisture is low")

    if user_data['Soil pH'] < 5.5:
        recs.append("Add lime to increase soil pH")

    if user_data['Soil pH'] > 7.5:
        recs.append("Use sulfur or organic matter to reduce pH")

    if user_data['Temperature(C)'] > 35:
        recs.append("Use mulching or shade nets to reduce heat stress")

    if user_data['Humidity (%)'] > 80:
        recs.append("Improve ventilation to prevent fungal diseases")

    if user_data['NDVI_index'] < 0.3:
        recs.append("Crop health is poor — check fertilizers or pests")

    if not recs:
        recs.append("Maintain current conditions — everything looks optimal")

    return recs

# ===============================
# MAIN PREDICTION FUNCTION
# ===============================
def predict_user_input(
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
):

    # Encode (FIXED KEYS)
    crop = encode_input(user_data['Crop Type'], encoders['crop_type'])
    region = encode_input(user_data['Region'], encoders['region'])

    if crop is None or region is None:
        return {"error": "Invalid crop or region"}

    key = (crop, region)

    if key not in if_models:
        return {"error": "No trained model for this crop-region"}

    # Convert UI → model format
    X = np.array([[user_data[col] for col in COLUMN_MAP.keys()]])

    # Scores
    if_s = scaler_if.transform([[get_if_score(if_models[key], X)]])[0][0]
    lof_s = scaler_lof.transform([[get_lof_score(lof_models[key], X)]])[0][0]
    rule_s = scaler_rule.transform([[detect_rule_anomalies(user_data)]])[0][0]

    # Final score
    final_score = (
        weights["if"] * if_s +
        weights["lof"] * lof_s +
        weights["rule"] * rule_s
    )

    # Decision
    if final_score >= 0.35:
        prediction = "ANOMALY"
    elif final_score >= 0.2:
        prediction = "TENDENCY"
    else:
        prediction = "NORMAL"

    return {
        "prediction": prediction,
        "parameter_issues": detect_problems_dynamic(user_data, df, encoders),
        "sensor_issues": detect_sensor_fusion_anomalies(user_data),
        "recommendations": generate_recommendations(user_data)
    }
