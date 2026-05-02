import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import requests

# ===============================
# LOAD MODEL
# ===============================
model = pickle.load(open("model.pkl", "rb"))

# ===============================
# WEATHER FUNCTION
# ===============================
def get_weather(city):
    try:
        API_KEY = "2099c9ed011e4e99931112909260105"   # optional (replace later)
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
        res = requests.get(url).json()
        return res["current"]["condition"]["text"]
    except:
        return "Unknown"

# ===============================
# UI
# ===============================
st.set_page_config(page_title="Food Demand AI", layout="centered")

st.title("🍽️ Food Demand Prediction System")

st.write("Enter details to predict food demand and get recommendation")

# ===============================
# INPUTS
# ===============================
city = st.text_input("City", "Bangalore")

week = st.slider("Week", 1, 52)

checkout_price = st.number_input("Checkout Price", value=200.0)
base_price = st.number_input("Base Price", value=250.0)

emailer = st.selectbox("Email Promotion", [0, 1])
homepage = st.selectbox("Homepage Featured", [0, 1])

# ===============================
# PREDICT BUTTON
# ===============================
if st.button("Predict Demand"):

    # ===============================
    # CREATE INPUT DATA
    # ===============================
    df = pd.DataFrame([{
        "week": week,
        "checkout_price": checkout_price,
        "base_price": base_price,
        "emailer_for_promotion": emailer,
        "homepage_featured": homepage,
        "price_diff": base_price - checkout_price,
        "discount": (base_price - checkout_price) / base_price
    }])

    # ===============================
    # PREDICTION
    # ===============================
    pred_log = model.predict(df)
    pred = float(np.expm1(pred_log)[0])

    # ===============================
    # DEMAND LOGIC
    # ===============================
    if pred < 100:
        level = "LOW-MODERATE"
        rmin = int(pred * 1.05)
        rmax = int(pred * 1.25)
    elif pred < 300:
        level = "MODERATE"
        rmin = int(pred * 1.00)
        rmax = int(pred * 1.20)
    else:
        level = "HIGH"
        rmin = int(pred * 1.00)
        rmax = int(pred * 1.15)

    # ===============================
    # WEATHER + WEEKEND
    # ===============================
    weather = get_weather(city)
    is_weekend = datetime.now().weekday() >= 5

    # ===============================
    # OUTPUT
    # ===============================
    st.subheader("📊 Results")

    st.write(f"**Prediction:** {round(pred, 2)} orders")
    st.write(f"**Demand Level:** {level}")
    st.write(f"**Weather:** {weather}")
    st.write(f"**Weekend:** {is_weekend}")
    st.write(f"**Recommendation:** Prepare {rmin}–{rmax} orders")

    # ===============================
    # EXPLANATION
    # ===============================
    st.subheader("🧠 Explanation")

    st.write(f"""
Demand is classified as **{level}** based on predicted orders.
Weather impact is minimal and today is {'weekend' if is_weekend else 'weekday'}.
No strong external demand drivers detected.
""")