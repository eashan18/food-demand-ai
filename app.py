import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ===============================
# LOAD MODEL
# ===============================
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open(model_path, "rb"))

# ===============================
# LOAD FEATURE COLUMNS (IMPORTANT)
# ===============================
cols_path = os.path.join(os.path.dirname(__file__), "columns.pkl")
feature_cols = pickle.load(open(cols_path, "rb"))

# ===============================
# UI
# ===============================
st.set_page_config(page_title="PredictPlate", layout="centered")

st.title("🍽️ PredictPlate AI")
st.subheader("Smart Food Demand Prediction")

# ===============================
# USER INPUTS
# ===============================
week = st.slider("Week", 1, 52, 10)
base_price = st.number_input("Base Price", value=250)
checkout_price = st.number_input("Checkout Price", value=200)

# ===============================
# FEATURE ENGINEERING (SAME AS TRAINING)
# ===============================
price_diff = base_price - checkout_price
discount = price_diff / base_price if base_price != 0 else 0

input_dict = {
    "week": week,
    "checkout_price": checkout_price,
    "base_price": base_price,
    "price_diff": price_diff,
    "discount": discount
}

# ===============================
# CREATE DATAFRAME + FIX FEATURES
# ===============================
df = pd.DataFrame([input_dict])

# Add missing columns
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0

# Reorder columns
df = df[feature_cols]

# ===============================
# PREDICT
# ===============================
if st.button("Predict Demand"):

    pred_log = model.predict(df)
    pred = float(np.expm1(pred_log)[0])

    # ===============================
    # DEMAND LOGIC
    # ===============================
    if pred < 100:
        level = "LOW-MODERATE"
        rec_min = int(pred * 1.05)
        rec_max = int(pred * 1.25)

    elif pred < 300:
        level = "MODERATE"
        rec_min = int(pred * 1.00)
        rec_max = int(pred * 1.20)

    else:
        level = "HIGH"
        rec_min = int(pred * 1.00)
        rec_max = int(pred * 1.15)

    # ===============================
    # OUTPUT UI
    # ===============================
    st.success(f"📊 Predicted Orders: {round(pred, 2)}")

    st.info(f"📈 Demand Level: {level}")

    st.warning(f"📦 Recommendation: Prepare {rec_min} – {rec_max} orders")

    st.write("### 🧠 Explanation")
    st.write(
        f"Demand is {level.lower()} based on pricing and weekly trends. "
        f"The discount level is {round(discount, 2)}, influencing customer orders."
    )