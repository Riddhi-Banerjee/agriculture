import numpy as np

# =========================================
# FEATURE LIST (IMPORTANT - SAME AS TRAINING)
# =========================================
features = [
    'soil_moisture_%',
    'soil_pH',
    'temperature_C',
    'rainfall_mm',
    'humidity_%',
    'NDVI_index'
]

# =========================================
# ENCODING
# =========================================
def encode_input(value, encoder):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return None

# =========================================
# MODEL SCORE FUNCTIONS
# =========================================
def get_if_score(model, X):
    return -model.decision_function(X)[0]

def get_lof_score(model, X):
    return -model.decision_function(X)[0]

def get_ae_score(autoencoder, scaler, X):
    X_scaled = scaler.transform(X)
    recon = autoencoder.predict(X_scaled, verbose=0)
    return np.mean((X_scaled - recon) ** 2)

# =========================================
# RULE-BASED ANOMALY
# =========================================
def detect_rule_anomalies(row):
    issues = 0

    if row['rainfall_mm'] > 100 and row['soil_moisture_%'] < 30:
        issues += 1

    if row['NDVI_index'] > 0.7 and row['soil_moisture_%'] < 25:
        issues += 1

    if row['temperature_C'] > 35 and row['humidity_%'] > 80:
        issues += 1

    return issues * 2  # weighted

# =========================================
# PARAMETER-LEVEL ANALYSIS
# =========================================
def detect_problems_dynamic(user_data, df, encoders):

    crop = encode_input(user_data['crop_type'], encoders['crop_type'])
    region = encode_input(user_data['region'], encoders['region'])

    key_df = df[(df['crop_type'] == crop) & (df['region'] == region)]

    if len(key_df) == 0:
        return []

    issues = []

    for f in features:
        low = key_df[f].quantile(0.10)
        high = key_df[f].quantile(0.90)
        val = user_data[f]

        if val < low:
            issues.append(f"{f} too LOW (expected > {round(low,2)})")
        elif val > high:
            issues.append(f"{f} too HIGH (expected < {round(high,2)})")

    return issues

# =========================================
# SENSOR FUSION LOGIC
# =========================================
def detect_sensor_fusion_anomalies(data):

    issues = []

    if data['rainfall_mm'] > 100 and data['soil_moisture_%'] < 30:
        issues.append("High rainfall but low soil moisture (sensor issue)")

    if data['NDVI_index'] > 0.7 and data['soil_moisture_%'] < 25:
        issues.append("High NDVI but low soil moisture")

    if data['temperature_C'] > 35 and data['humidity_%'] > 80:
        issues.append("High temperature with very high humidity")

    if data['temperature_C'] > 40 and data['NDVI_index'] > 0.7:
        issues.append("Extreme temperature but high NDVI")

    if data['rainfall_mm'] < 10 and data['NDVI_index'] > 0.8:
        issues.append("Low rainfall but high NDVI")

    return issues

# =========================================
# MAIN PREDICTION FUNCTION (CORE)
# =========================================
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

    # Encode
    crop = encode_input(user_data['crop_type'], encoders['crop_type'])
    region = encode_input(user_data['region'], encoders['region'])

    if crop is None or region is None:
        return {"error": "Invalid crop or region"}

    key = (crop, region)

    if key not in if_models:
        return {"error": "No trained model for this crop-region"}

    # Prepare input
    X = np.array([[user_data[f] for f in features]])

    # Scores (hidden internally)
    if_raw = get_if_score(if_models[key], X)
    lof_raw = get_lof_score(lof_models[key], X)
    ae_raw = 0
    rule_raw = detect_rule_anomalies(user_data)

    # Normalize
    if_s = scaler_if.transform([[if_raw]])[0][0]
    lof_s = scaler_lof.transform([[lof_raw]])[0][0]
    ae_s = 0
    rule_s = scaler_rule.transform([[rule_raw]])[0][0]

    # Final score
    final_score = (
        weights["if"] * if_s +
        weights["lof"] * lof_s +
        weights["rule"] * rule_s
    )

    # Decision
    if final_score > best_thresh:
        prediction = "ANOMALY"
    elif final_score >= 0.5:
        prediction = "TENDENCY"
    else:
        prediction = "NORMAL"

    # ✅ FIXED PARAMETER ISSUES (important 🔥)
    param_issues = detect_problems_dynamic(user_data, df, encoders)
    fusion_issues = detect_sensor_fusion_anomalies(user_data)

    # Remove useless message
    if param_issues == ["No reference data available"]:
        param_issues = []

    return {
        "final_score": round(final_score, 4),
        "prediction": prediction,
        "parameter_issues": param_issues,
        "sensor_issues": fusion_issues
    }
