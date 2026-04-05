import numpy as np

features = [
    'soil_moisture_%',
    'soil_pH',
    'temperature_C',
    'rainfall_mm',
    'humidity_%',
    'NDVI_index'
]

def encode_input(value, encoder):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return None

def get_if_score(model, X):
    return -model.decision_function(X)[0]

def get_lof_score(model, X):
    return -model.decision_function(X)[0]

def detect_rule_anomalies(row):
    issues = 0

    if row['rainfall_mm'] > 100 and row['soil_moisture_%'] < 30:
        issues += 1
    if row['NDVI_index'] > 0.7 and row['soil_moisture_%'] < 25:
        issues += 1
    if row['temperature_C'] > 35 and row['humidity_%'] > 80:
        issues += 1

    return issues * 2

# ===============================
# PARAMETER ANALYSIS
# ===============================
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

# ===============================
# SENSOR ISSUES
# ===============================
def detect_sensor_fusion_anomalies(data):

    issues = []

    if data['rainfall_mm'] > 100 and data['soil_moisture_%'] < 30:
        issues.append("High rainfall but low soil moisture")

    if data['NDVI_index'] > 0.7 and data['soil_moisture_%'] < 25:
        issues.append("High NDVI but low soil moisture")

    if data['temperature_C'] > 35 and data['humidity_%'] > 80:
        issues.append("High temperature with very high humidity")

    return issues

# ===============================
# 🔥 RECOMMENDATION ENGINE
# ===============================
def generate_recommendations(user_data):

    recs = []

    if user_data['soil_moisture_%'] < 30:
        recs.append("Increase irrigation — soil moisture is low")

    if user_data['soil_pH'] < 5.5:
        recs.append("Add lime to increase soil pH")

    if user_data['soil_pH'] > 7.5:
        recs.append("Use sulfur or organic matter to reduce pH")

    if user_data['temperature_C'] > 35:
        recs.append("Use mulching or shade nets to reduce heat stress")

    if user_data['humidity_%'] > 80:
        recs.append("Improve ventilation to prevent fungal diseases")

    if user_data['NDVI_index'] < 0.3:
        recs.append("Crop health is poor — consider fertilizers or pest check")

    if not recs:
        recs.append("Maintain current conditions — everything looks optimal")

    return recs

# ===============================
# MAIN FUNCTION
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

    crop = encode_input(user_data['crop_type'], encoders['crop_type'])
    region = encode_input(user_data['region'], encoders['region'])

    if crop is None or region is None:
        return {"error": "Invalid crop or region"}

    key = (crop, region)

    if key not in if_models:
        return {"error": "No trained model for this crop-region"}

    X = np.array([[user_data[f] for f in features]])

    if_s = scaler_if.transform([[get_if_score(if_models[key], X)]])[0][0]
    lof_s = scaler_lof.transform([[get_lof_score(lof_models[key], X)]])[0][0]
    rule_s = scaler_rule.transform([[detect_rule_anomalies(user_data)]])[0][0]

    final_score = (
        weights["if"] * if_s +
        weights["lof"] * lof_s +
        weights["rule"] * rule_s
    )

    if final_score > = 0.35:
        prediction = "ANOMALY"
    elif final_score >= 0.2 and final_score < 0.35 :
        prediction = "TENDENCY"
    else:
        prediction = "NORMAL"

    return {
        "prediction": prediction,
        "parameter_issues": detect_problems_dynamic(user_data, df, encoders),
        "sensor_issues": detect_sensor_fusion_anomalies(user_data),
        "recommendations": generate_recommendations(user_data)
    }
