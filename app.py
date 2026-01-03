from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# -----------------------------
# Load saved artifacts
# -----------------------------
MODEL_PATH = "Models/xgb_gcv_model.joblib"
SCALER_PATH = "Scaling/robust_scaler_refined.joblib"
TRANSFORMERS_PATH = "Transformers/yeojohnson_transformers.pkl"
TRANSFORM_MAP_PATH = "Transformers/transform_map.pkl"
SHIFTS_PATH = "Transformers/log_shifts.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
transformers = joblib.load(TRANSFORMERS_PATH)
transform_map = joblib.load(TRANSFORM_MAP_PATH)
shifts = joblib.load(SHIFTS_PATH)

COAL_IMAGES = {
    "Anthracite": "Anthracite.jpg",
    "Bituminous": "Bituminous.JPG",      
    "Sub-Bituminous": "Sub-Bituminous.jpg",
    "Lignite": "Lignite.jpeg",
}


target_col = "GCV_kcal"

FEATURE_COLS = [
    "Moisture",
    "Volatile_matter",
    "Fixed_Carbon",
    "Std.Ash",
    "Hydrogen",
    "Nitrogen",
    "Sulfur"
]

# -----------------------------
# Grade classifier  (grade + broad coal type)
# -----------------------------
def classify_grade(gcv: float):
    # returns (grade_label, coal_type)

    if gcv >= 7001:
        return "G1 – Anthracite", "Anthracite"
    elif 6701 <= gcv <= 7000:
        return "G2 – Anthracite", "Anthracite"
    elif 6401 <= gcv <= 6700:
        return "G3 – Bituminous", "Bituminous"
    elif 6101 <= gcv <= 6400:
        return "G4 – Bituminous", "Bituminous"
    elif 5801 <= gcv <= 6100:
        return "G5 – Bituminous", "Bituminous"
    elif 5501 <= gcv <= 5800:
        return "G6 – Bituminous", "Bituminous"
    elif 5201 <= gcv <= 5500:
        return "G7 – Bituminous", "Bituminous"
    elif 4901 <= gcv <= 5200:
        return "G8 – Bituminous", "Bituminous"
    elif 4601 <= gcv <= 4900:
        return "G9 – Sub-bituminous", "Sub-Bituminous"
    elif 4301 <= gcv <= 4600:
        return "G10 – Sub-bituminous", "Sub-Bituminous"
    elif 4001 <= gcv <= 4300:
        return "G11 – Sub-bituminous", "Sub-Bituminous"
    elif 3701 <= gcv <= 4000:
        return "G12 – Sub-bituminous", "Sub-Bituminous"
    elif 3401 <= gcv <= 3700:
        return "G13 – Sub-bituminous", "Sub-Bituminous"
    elif 3101 <= gcv <= 3400:
        return "G14 – Sub-bituminous", "Sub-Bituminous"
    elif 2801 <= gcv <= 3100:
        return "G15 – Lignite", "Lignite"
    elif 2501 <= gcv <= 2800:
        return "G16 – Lignite", "Lignite"
    elif 2201 <= gcv <= 2500:
        return "G17 – Lignite", "Lignite"
    else:
        return "Unclassified – Below G17", "Unclassified"



# -----------------------------
# Helpers: forward & inverse transforms
# -----------------------------
def forward_transform_column(col_name, series):
    t = transform_map.get(col_name, "none")
    s = series.astype(float).copy()
    mask = s.notna()

    if t == "log1p":
        s[mask] = np.log1p(s[mask])
    elif t == "log1p_shift":
        shift = shifts[col_name]
        s[mask] = np.log1p(s[mask] + shift)
    elif t == "sqrt":
        s[mask] = np.sqrt(s[mask])
    elif t == "cuberoot":
        s[mask] = np.cbrt(s[mask])
    elif t == "yeojohnson":
        pt = transformers[col_name]
        s[mask] = pt.transform(s[mask].values.reshape(-1, 1)).ravel()
    return s


def inverse_transform_target(series):
    t = transform_map[target_col]
    s = series.astype(float).copy()
    mask = s.notna()

    if t == "log1p":
        s[mask] = np.expm1(s[mask])
    elif t == "log1p_shift":
        shift = shifts[target_col]
        s[mask] = np.expm1(s[mask]) - shift
    elif t == "sqrt":
        s[mask] = s[mask] ** 2
    elif t == "cuberoot":
        s[mask] = s[mask] ** 3
    elif t == "yeojohnson":
        pt = transformers[target_col]
        s[mask] = pt.inverse_transform(s[mask].values.reshape(-1, 1)).ravel()
    return s


def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict], columns=FEATURE_COLS)
    for col in FEATURE_COLS:
        df[col] = forward_transform_column(col, df[col])
    X_scaled = scaler.transform(df[FEATURE_COLS])
    return X_scaled


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    # just render the form
    return render_template(
        "index.html",
        prediction=None,
        grade=None,
        feature_cols=FEATURE_COLS
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("index"))

    prediction = None
    grade = None
    coal_type = None
    coal_image = None

    try:
        input_data = {}
        for col in FEATURE_COLS:
            val_str = request.form.get(col)
            if val_str is None or val_str.strip() == "":
                raise ValueError(f"Missing value for {col}")
            input_data[col] = float(val_str)

        X_processed = preprocess_input(input_data)

        # model outputs transformed GCV_kcal
        y_pred_trans = model.predict(X_processed)

        # inverse transform to original kcal/kg
        y_pred_real = inverse_transform_target(pd.Series(y_pred_trans)).iloc[0]
        prediction = round(float(y_pred_real), 2)

        grade, coal_type = classify_grade(prediction)

        if coal_type in COAL_IMAGES:
            coal_image = COAL_IMAGES[coal_type]

    except Exception as e:
        prediction = f"Error: {e}"
        grade = None
        coal_type = None
        coal_image = None

    return render_template(
        "result.html",
        prediction=prediction,
        grade=grade,
        coal_type=coal_type,
        coal_image=coal_image,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
