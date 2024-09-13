import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import joblib
from PIL import Image

# Load the saved XGBoost model and scalers
best_model = joblib.load('xgb_model_filtered2.pkl')  # Load XGBoost model
scaler_y = joblib.load('y_scaler_filtered2.pkl')  # Load y-scaler
scaler_X = joblib.load('X_scaler_filtered2.pkl')  # Load X-scaler

# Function to convert SMILES to MACCS fingerprints
def smiles_to_maccs(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        st.error("Invalid SMILES string.")
        return np.zeros(166, dtype=int)  # Return a default array of 166 bits if SMILES is invalid
    maccs_fingerprint = MACCSkeys.GenMACCSKeys(molecule)
    return np.array(list(maccs_fingerprint)[1:], dtype=int)  # Discard the first bit

# Function to predict pIC50 and IC50
def predict_ic50(smiles):
    fingerprints = np.array([smiles_to_maccs(smiles)])
    if fingerprints is None:
        return None, None
    
    # Scale the fingerprints
    X_new_scaled = scaler_X.transform(fingerprints)
    
    # Predict transformed pIC50 and convert it back
    y_pred_transformed = best_model.predict(X_new_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()
    
    predicted_pic50 = y_pred[0]
    predicted_ic50 = 10 ** (-predicted_pic50) * 1000000  # Convert pIC50 to IC50 in µM
    
    return predicted_ic50, predicted_pic50

# Home Page
def home_page():
    st.title("IC50 Prediction for COVID-19 Compounds")

    # Display an image on the homepage
    image = Image.open('assets/virus.jpg')  # Replace with your image path
    st.image(image, use_column_width=True)

    st.write("""
        **IC50** measures the effectiveness of a substance in inhibiting a specific biological or biochemical function. 
        Predicting the IC50 value of compounds for COVID-19 can aid in understanding their therapeutic potential.
    """)

    # Add an animation or other media
    st.markdown("""
        <div style="text-align:center;">
            <img src="https://assets10.lottiefiles.com/packages/lf20_u4yrau.json" alt="Virus Animation" height="300">
        </div>
    """, unsafe_allow_html=True)

    # Navigation button
    if st.button("Start Prediction"):
        st.session_state.page = "predict_page"

# Prediction Page
def predict_page():
    st.title("Predict IC50 Value")

    smiles_input = st.text_input("Enter SMILES String", "")

    if st.button("Predict"):
        predicted_ic50, predicted_pic50 = predict_ic50(smiles_input)
        if predicted_ic50 is not None:
            # st.success(f"Predicted pIC50: {predicted_pic50:.4f}")
            st.success(f"Predicted IC50: {predicted_ic50:.4f} µM")
        else:
            st.error("Unable to generate prediction. Please check the SMILES string.")

    # Navigation buttons for another prediction or returning home
    if st.button("Another Prediction"):
        st.session_state.smiles_input = ""

    if st.button("Return to Home"):
        st.session_state.page = "home_page"

# Initialize the session state for page routing
if 'page' not in st.session_state:
    st.session_state.page = 'home_page'

# Page routing logic
if st.session_state.page == 'home_page':
    home_page()
elif st.session_state.page == 'predict_page':
    predict_page()
