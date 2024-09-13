import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Descriptors3D
import joblib
from PIL import Image

# Load the trained model and scaler
model = joblib.load('xgboost_model_pIC50_optimized.pkl')
scaler = joblib.load('scaler_X_pIC50.pkl')

# Function to compute 2D and 3D molecular descriptors from SMILES
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string.")
        return None

    # 2D descriptors
    descriptors = {
        'MolecularWeight': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HydrogenBondDonors': Descriptors.NumHDonors(mol),
        'HydrogenBondAcceptors': Descriptors.NumHAcceptors(mol),
        'TopologicalPolarSurfaceArea': Descriptors.TPSA(mol),
        'NumberofRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumberofValenceElectrons': Descriptors.NumValenceElectrons(mol),
        'NumberofAromaticRings': Descriptors.NumAromaticRings(mol),
        'Fractionofsp3Carbons': Descriptors.FractionCSP3(mol)
    }

    # Generate 3D molecule and compute 3D descriptors
    mol_3d = generate_3d_molecule(mol)
    if mol_3d:
        descriptors.update(compute_3d_descriptors(mol_3d))

    return descriptors

def generate_3d_molecule(mol):
    mol = Chem.AddHs(mol)
    best_mol = None
    best_energy = float('inf')

    for _ in range(100):
        mol_temp = Chem.Mol(mol)  # Ensure fresh copy each iteration
        AllChem.EmbedMolecule(mol_temp)
        AllChem.UFFOptimizeMolecule(mol_temp)
        energy = AllChem.UFFGetMoleculeForceField(mol_temp).CalcEnergy()
        
        if energy < best_energy:
            best_energy = energy
            best_mol = mol_temp
    
    return best_mol

def compute_3d_descriptors(mol):
    def safe_descriptor_calculation(func):
        try:
            return func(mol)
        except Exception:
            return np.nan

    return {
        'Asphericity': safe_descriptor_calculation(Descriptors3D.Asphericity),
        'Eccentricity': safe_descriptor_calculation(Descriptors3D.Eccentricity),
        'NPR1': safe_descriptor_calculation(Descriptors3D.NPR1),
        'NPR2': safe_descriptor_calculation(Descriptors3D.NPR2),
        'PMI1': safe_descriptor_calculation(Descriptors3D.PMI1),
        'PMI2': safe_descriptor_calculation(Descriptors3D.PMI2),
        'PMI3': safe_descriptor_calculation(Descriptors3D.PMI3),
        'RadiusofGyration': safe_descriptor_calculation(Descriptors3D.RadiusOfGyration),
        'InertialShapeFactor': safe_descriptor_calculation(Descriptors3D.InertialShapeFactor),
        'SpherocityIndex': safe_descriptor_calculation(Descriptors3D.SpherocityIndex)
    }

# Function to generate features (descriptors) from SMILES
def generate_features(smiles):
    descriptors = compute_descriptors(smiles)
    if descriptors is None:
        return None
    
    # Convert descriptors to a DataFrame
    original_feature_names = ['MolecularWeight', 'LogP', 'HydrogenBondDonors',
                              'HydrogenBondAcceptors', 'TopologicalPolarSurfaceArea',
                              'NumberofRotatableBonds', 'NumberofValenceElectrons',
                              'NumberofAromaticRings', 'Fractionofsp3Carbons',
                              'Asphericity', 'Eccentricity', 'NPR1', 'NPR2',
                              'PMI1', 'PMI2', 'PMI3', 'RadiusofGyration',
                              'InertialShapeFactor', 'SpherocityIndex']
    descriptors_df = pd.DataFrame([descriptors], columns=original_feature_names)
    return descriptors_df

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
    image = Image.open('assets/virus.jpg')  # Replace with your image path
    st.image(image, use_column_width=True)
    
    st.write("""
        **IC50** is a measure of the effectiveness of a substance in inhibiting a specific biological or biochemical function. 
        For COVID-19, predicting the IC50 value of compounds can help in understanding their potential as therapeutic agents.
    """)
    
    # Add an animation (Lottie animation embedded)
    st.markdown(
        """
        <div style="text-align:center;">
            <img src="https://assets10.lottiefiles.com/packages/lf20_u4yrau.json" alt="Virus Animation" height="300">
        </div>
        """, unsafe_allow_html=True
    )
    
    if st.button("Start Prediction"):
        st.session_state.page = "predict_page"

# Prediction Page
def predict_page():
    st.title("Predict IC50 Value")
    
    smiles_input = st.text_input("Enter SMILES String", "")
    
    if st.button("Predict"):
        prediction, features = predict_ic50(smiles_input)
        if prediction is not None:
            st.success(f"Predicted IC50: {prediction:.4f} µM")
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
