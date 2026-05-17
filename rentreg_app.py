import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input

# 1. Manually Re-build the Model Architecture (Matches your training)
def load_manual_model():
    model = Sequential([
        Input(shape=(119,)), 
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear') 
    ])
    # Load ONLY the weights from your .h5 file
    model.load_weights('basak_adana_rent_model.h5')
    return model

# 2. Page Setup
st.set_page_config(page_title="Adana Rent Predictor", page_icon="🏠")

# 3. Load Assets
@st.cache_resource
def load_assets():
    model = load_manual_model()
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    return model, scaler, features

try:
    model, scaler, features = load_assets()
    st.sidebar.success("✅ Model Weights Loaded Successfully")
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# 4. Extract Category names for dropdowns
districts = sorted([c.replace('District_', '') for c in features if c.startswith('District_')])
neighborhoods = sorted([c.replace('Neighborhood_', '') for c in features if c.startswith('Neighborhood_')])

# 5. User Interface
st.title("🏠 Adana Rent Prediction AI")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        m2 = st.number_input("Square Meters (m²)", min_value=10, value=120)
        rooms = st.number_input("Number of Rooms", min_value=1, value=3)
        sel_dist = st.selectbox("District", districts)
    with col2:
        age = st.number_input("Building Age", min_value=0, value=5)
        floor = st.number_input("Floor Level", min_value=0, value=3)
        sel_neigh = st.selectbox("Neighborhood", neighborhoods)
    
    submit = st.form_submit_button("Predict Rent", use_container_width=True)

# 6. Prediction Logic
if submit:
    # Create the 119-feature template
    input_df = pd.DataFrame(0, index=[0], columns=features)
    
    # Map numeric values (Verify names against features.pkl)
    mapping = {'m2': m2, 'Bina_Yasi': age, 'Oda_Sayisi': rooms, 'Bulundugu_Kat': floor}
    for col, val in mapping.items():
        if col in input_df.columns:
            input_df[col] = val
    
    # One-Hot encoding for District and Neighborhood
    if f"District_{sel_dist}" in input_df.columns:
        input_df[f"District_{sel_dist}"] = 1
    if f"Neighborhood_{sel_neigh}" in input_df.columns:
        input_df[f"Neighborhood_{sel_neigh}"] = 1

    try:
        scaled = scaler.transform(input_df)
        pred_log = model.predict(scaled, verbose=0)
        price = np.expm1(pred_log[0][0]) 
        
        st.success(f"### Estimated Monthly Rent: {price:,.2f} TL")
        st.balloons()
    except Exception as e:
        st.error(f"Prediction Error: {e}")