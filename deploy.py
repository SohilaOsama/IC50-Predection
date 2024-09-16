import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw
import joblib
from PIL import Image

# Load the saved XGBoost model and scalers
best_model = joblib.load('xgb_model_filtered2.pkl')  # Load XGBoost model
scaler_y = joblib.load('y_scaler_filtered2.pkl')  # Load y-scaler
scaler_X = joblib.load('X_scaler_filtered2.pkl')  # Load X-scaler

# Custom CSS for styling
def add_custom_css():
    st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-color: #f7f8fa;
        padding: 2rem;
    }
    .main-title {
        text-align: center;
        color: #2c3e50;
        font-size: 3rem;
        font-weight: bold;
    }
    .subheader {
        text-align: center;
        color: #34495e;
        font-size: 1.5rem;
        margin-bottom: 2rem;
    }
    .centered-button {
        text-align: center;
        margin-top: 20px;
    }
    .feature-table {
        margin-top: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #ecf0f1;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transition: background-color 0.3s ease;
    }
    .footer {
        text-align: center;
        padding: 20px 0;
        margin-top: 50px;
        font-size: 14px;
        color: #7f8c8d;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to convert SMILES to MACCS fingerprints
def smiles_to_maccs(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        st.error("Invalid SMILES string.")
        return np.zeros(166, dtype=int)  # Return a default array of 166 bits if SMILES is invalid
    maccs_fingerprint = MACCSkeys.GenMACCSKeys(molecule)
    return np.array(list(maccs_fingerprint)[1:], dtype=int)  # Discard the first bit

# Function to extract features from SMILES
def extract_features(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        st.error("Invalid SMILES string.")
        return None
    
    # Extract molecular descriptors
    features = {
        'MolecularWeight': Descriptors.MolWt(molecule),
        'LogP': Descriptors.MolLogP(molecule),
        'HydrogenBondDonors': Descriptors.NumHDonors(molecule),
        'HydrogenBondAcceptors': Descriptors.NumHAcceptors(molecule),
        'TopologicalPolarSurfaceArea': Descriptors.TPSA(molecule),
        'NumberofRotatableBonds': Descriptors.NumRotatableBonds(molecule),
        'NumberofValenceElectrons': Descriptors.NumValenceElectrons(molecule),
        'NumberofAromaticRings': rdMolDescriptors.CalcNumAromaticRings(molecule),
        'Fractionofsp3Carbons': rdMolDescriptors.CalcFractionCSP3(molecule)
    }
    
    return features

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

# Function to generate a 2D image from SMILES
def generate_2d_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string for 2D drawing.")
        return None
    return Draw.MolToImage(mol)

# Home Page
def home_page():
    st.markdown("<h1 class='main-title'>IC50 Prediction & Feature Extraction</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='subheader'>Predict IC50 value and extract molecular features from SMILES</h2>", unsafe_allow_html=True)

    # Display an image on the homepage
    image = Image.open('assets/virus.jpg')  # Replace with your image path
    st.image(image, use_column_width=True)

    st.write("""
        IC50 measures the effectiveness of a substance in inhibiting a specific biological or biochemical function. 
        Predicting the IC50 value of compounds for COVID-19 can help in understanding their therapeutic potential.
    """)

    # Navigation button
    st.markdown("<div class='centered-button'>", unsafe_allow_html=True)
    if st.button("Start Prediction and Extraction"):
        st.session_state.page = "predict_page"
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction Page
def predict_page():
    st.markdown("<h1 class='main-title'>Predict & Extract Features</h1>", unsafe_allow_html=True)

    smiles_input = st.text_input("Enter SMILES String", "")

    if st.button("Predict and Extract Features"):
        predicted_ic50, predicted_pic50 = predict_ic50(smiles_input)
        features = extract_features(smiles_input)

        if predicted_ic50 is not None and features is not None:
            # Display IC50 prediction
            st.success(f"Predicted IC50: {predicted_ic50:.4f} µM")
            
            # Display extracted features in a table
            st.write("### Extracted Features:")
            feature_df = pd.DataFrame([features])
            st.write(feature_df)

            # Button to download features as CSV
            csv = feature_df.to_csv(index=False)
            st.download_button(
                label="Download features as CSV",
                data=csv,
                file_name='compound_features.csv',
                mime='text/csv'
            )

            # 2D Molecule Image
            st.write("### 2D Structure")
            img = generate_2d_image(smiles_input)
            if img:
                st.image(img, caption="2D Structure", use_column_width=True)
        else:
            st.error("Unable to generate prediction or extract features. Please check the SMILES string.")

    # Navigation buttons for another prediction or returning home
    st.markdown("<div class='centered-button'>", unsafe_allow_html=True)
    if st.button("Another Prediction"):
        st.session_state.smiles_input = ""
    if st.button("Return to Home"):
        st.session_state.page = "home_page"
    st.markdown("</div>", unsafe_allow_html=True)

# Initialize the session state for page routing
if 'page' not in st.session_state:
    st.session_state.page = 'home_page'

# Page routing logic
if st.session_state.page == 'home_page':
    home_page()
elif st.session_state.page == 'predict_page':
    predict_page()

# Add custom CSS
add_custom_css()

# Add footer
st.markdown("<div class='footer'>© 2024 | National HPC Grid</div>", unsafe_allow_html=True)
