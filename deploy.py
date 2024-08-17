import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from PIL import Image

# Load the trained model and scaler
model = joblib.load('xgboost_model_pIC50_optimized.pkl')
scaler = joblib.load('scaler_X_pIC50.pkl')

# Function to generate molecular descriptors from SMILES
def generate_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string.")
        return None
    descriptor_names = ['MolecularWeight', 'LogP', 'HydrogenBondDonors',
                        'HydrogenBondAcceptors', 'TopologicalPolarSurfaceArea',
                        'NumberofRotatableBonds', 'NumberofValenceElectrons',
                        'NumberofAromaticRings', 'Fractionofsp3Carbons',
                        'Asphericity', 'Eccentricity', 'NPR1', 'NPR2',
                        'PMI1', 'PMI2', 'PMI3', 'RadiusofGyration',
                        'InertialShapeFactor', 'SpherocityIndex']
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    features = np.array(calc.CalcDescriptors(mol)).reshape(1, -1)
    return pd.DataFrame(features, columns=descriptor_names)

# Function to predict IC50
def predict_ic50(smiles):
    features = generate_features(smiles)
    if features is not None:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return prediction[0], features
    return None, None

# Home Page
def home_page():
    st.title("IC50 Prediction for COVID-19")
    
    # Add an image to the home page
    st.write("""
        **IC50** is a measure of the effectiveness of a substance in inhibiting a specific biological or biochemical function. 
        For COVID-19, predicting the IC50 value of compounds can help in understanding their potential as therapeutic agents.
    """) 
    
    image = Image.open('assets/virus.jpg')  # Replace with your image path
    st.image(image, use_column_width=True)
    
    
    
    # Add an animation (Lottie animation embedded)
    # st.markdown(
    #     """
    #     <div style="text-align:center;">
    #         <img src="https://assets10.lottiefiles.com/packages/lf20_u4yrau.json" alt="Virus Animation" height="300">
    #     </div>
    #     """, unsafe_allow_html=True
    # )
    
    if st.button("Start Prediction"):
        st.session_state.page = "predict_page"

# Prediction Page
def predict_page():
    st.title("Predict IC50 Value")
    
    smiles_input = st.text_input("Enter SMILES String", "")
    
    if st.button("Predict"):
        prediction, features = predict_ic50(smiles_input)
        if prediction is not None:
            st.success(f"Predicted IC50: {prediction:.4f}")
            st.write("**Generated Features:**")
            st.dataframe(features)
        else:
            st.error("Unable to generate features. Please check the SMILES string.")
    
    # Add buttons for navigation
    if st.button("Another Prediction"):
        st.session_state.smiles_input = ""
    
    if st.button("Return to Home"):
        st.session_state.page = "home_page"

# Initialize navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home_page'

# Page routing
if st.session_state.page == 'home_page':
    home_page()
elif st.session_state.page == 'predict_page':
    predict_page()
