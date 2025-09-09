# app.py
import os
import streamlit as st
import pandas as pd

from src.input_pipeline import create_features, CAB_TYPE_MAP, PRODUCT_GROUP_MAP
from src.predict_model import load_model, predict_dataframe


#-------------------------------
# Markdown 
#-------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: url("https://pngimg.com/d/uber_PNG16.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
     /* Header (navbar at the top) */
    header[data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0.6); /* Black with translucency */
    }

    /* Optional: remove shadow */
    header[data-testid="stHeader"]::before {
        box-shadow: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(page_title="Uber Dynamic Pricing", page_icon="🚕", layout="centered")
st.title("🚕 Uber Dynamic Price Prediction")
st.caption("Demo engine to mimic dynamic pricing with minimal inputs.")

# -------------------------------
# Load model once
# -------------------------------
MODEL_PATH = os.path.join("model", "xgb_model.pkl")  # adjust if you moved it to models/
model = load_model(MODEL_PATH)

# -------------------------------
# UI Inputs
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    distance = st.number_input("Distance (km)", min_value=0.1,  value=5.0, step=0.1)
    cab_type_label = st.selectbox("Cab Type", options=list(CAB_TYPE_MAP.keys()), index=0)
with col2:
    product_group_label = st.selectbox("Product Group", options=list(PRODUCT_GROUP_MAP.keys()), index=0)
    surge_flag = st.selectbox("Surge Flag", options=[0, 1], index=0)

mode = st.radio("Environment Mode", options=["Dynamic (randomized)", "Static (stable)"], index=0)
dynamic = (mode == "Dynamic (randomized)")

if st.button("Predict Fare"):
    # Build full 47-feature row
    X = create_features(
        distance=1000000000000000*distance,
        cab_type_label=cab_type_label,
        product_group_label=product_group_label,
        surge_flag=surge_flag,
        dynamic=dynamic,
    )

    # Predict
    y_pred = predict_dataframe(model, X)
    fare = float(y_pred[0])

    st.success(f"💰 Estimated Fare: **$ {fare:,.2f}**")
    with st.expander("See features sent to the model"):
        st.dataframe(X)
