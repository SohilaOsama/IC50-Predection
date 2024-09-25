import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import MACCSkeys
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
import warnings
import os

# Suppress Deprecation Warnings and TensorFlow Info/Warn Messages
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages

# ============================
# Model and Scaler Loading
# ============================

@st.cache_resource
def load_models():
    """
    Load all necessary models and scalers.
    This function is cached to prevent reloading on every interaction.
    """
    try:
        # Load XGBoost model and scalers for IC50 prediction
        xgb = joblib.load('xgb_model_filtered2.pkl')
        scaler_y = joblib.load('y_scaler_filtered2.pkl')
        scaler_X = joblib.load('X_scaler_filtered2.pkl')

        # Load CNN model and scaler for classification
        cnn = load_model('cnn_maccs_model_class.h5')
        scaler_cls = joblib.load('scaler_class.pkl')

        return xgb, scaler_y, scaler_X, cnn, scaler_cls
    except FileNotFoundError as e:
        st.error(f"Model or scaler file not found: {e}")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading models/scalers: {e}")
        return None, None, None, None, None

# Load models and scalers
xgb_model, scaler_y, scaler_X, cnn_model, classification_scaler = load_models()

# ============================
# Define Category Labels
# ============================

category_labels = {0: 'Inactive', 1: 'Active', 2: 'Semi-Active'}

# ============================
# SMILES Processing Functions
# ============================

# Function to convert SMILES to MACCS fingerprints for XGBoost (166 bits)
def smiles_to_maccs_xgb(smiles):
    """
    Convert a SMILES string to MACCS fingerprints.
    Returns a numpy array of fingerprint bits or None if invalid.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        st.error("Invalid SMILES string. Please enter a correct SMILES format.")
        return None  # Return None if SMILES is invalid
    maccs_fingerprint = MACCSkeys.GenMACCSKeys(molecule)
    return np.array(list(maccs_fingerprint)[1:], dtype=int)  # Discard the first bit (166 bits)

# Function to convert SMILES to MACCS fingerprints for CNN (167 bits)
def smiles_to_maccs_cnn(smiles):
    """
    Convert a SMILES string to MACCS fingerprints.
    Returns a numpy array of fingerprint bits or None if invalid.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        st.error("Invalid SMILES string. Please enter a correct SMILES format.")
        return None  # Return None if SMILES is invalid
    maccs_fingerprint = MACCSkeys.GenMACCSKeys(molecule)
    return np.array(list(maccs_fingerprint), dtype=int)  # Retain all 167 bits

def extract_features(smiles):
    """
    Extract molecular descriptors from a SMILES string.
    Returns a dictionary of features or None if invalid.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        st.error("Invalid SMILES string. Cannot extract features.")
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

# ============================
# Prediction Functions
# ============================

def predict_ic50(smiles):
    """
    Predict IC50 and pIC50 values using the XGBoost model.
    Returns predicted IC50 (µM) and pIC50.
    """
    if None in (xgb_model, scaler_X, scaler_y):
        st.error("IC50 prediction model or scalers not loaded.")
        return None, None

    fingerprints = smiles_to_maccs_xgb(smiles)
    if fingerprints is None:
        return None, None

    try:
        # Scale the fingerprints
        X_new_scaled = scaler_X.transform([fingerprints])

        # Predict transformed pIC50 and convert it back
        y_pred_transformed = xgb_model.predict(X_new_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()

        predicted_pic50 = y_pred[0]
        predicted_ic50 = 10 ** (-predicted_pic50) * 1_000_000  # Convert pIC50 to IC50 in µM

        return predicted_ic50, predicted_pic50
    except Exception as e:
        st.error(f"Error during IC50 prediction: {e}")
        return None, None

def classify_compound(smiles):
    """
    Classify the compound into Active, Inactive, or Semi-active using the CNN model.
    Returns the category and confidence score.
    """
    if None in (cnn_model, classification_scaler):
        st.error("Classification model or scaler not loaded.")
        return None

    fingerprints = smiles_to_maccs_cnn(smiles)
    if fingerprints is None:
        return None

    try:
        # Scale the fingerprints using the classification scaler
        X_scaled = classification_scaler.transform([fingerprints])

        # Reshape for CNN input (samples, features, channels)
        X_cnn = np.expand_dims(X_scaled, axis=-1)

        # Predict probabilities
        probabilities = cnn_model.predict(X_cnn)

        # Get the class with highest probability
        predicted_class = np.argmax(probabilities, axis=-1)[0]

        # Get the corresponding label
        category = category_labels.get(predicted_class, "Unknown")

        # Get the confidence
        confidence = probabilities[0][predicted_class]

        return category, confidence
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None

# ============================
# Streamlit Page Definitions
# ============================

def home_page():
    """
    Display the home page with an introduction and navigation button.
    """
    st.title("IC50 Prediction and Compound Classification for COVID-19 Compounds")

    # Display an image on the homepage
    try:
        image = Image.open('assets/virus.jpg')  # Ensure this path is correct
        st.image(image, use_column_width=True)
    except FileNotFoundError:
        st.warning("Image 'assets/virus.jpg' not found. Please ensure the image path is correct.")

    st.write(""" 
        **IC50** measures the effectiveness of a substance in inhibiting a specific biological or biochemical function. 
        Predicting the IC50 value of compounds for COVID-19 can aid in understanding their therapeutic potential.
        
        Additionally, the compound can be classified into categories (**Active**, **Inactive**, **Semi-active**) based on its properties.
    """)

    # Navigation button
    if st.button("Start Prediction and Classification"):
        st.session_state.page = "predict_page"

def predict_page():
    """
    Display the prediction and classification page where users can input SMILES strings.
    """
    st.title("Predict IC50 Value and Classify Compound")

    # Instruction Steps
    st.write("""
    ## Instructions:
    1. To convert your compound to a Simplified Molecular Input Line Entry System (SMILES), please visit this website: [decimer.ai](https://decimer.ai/)
    """)

    # Input SMILES string
    smiles_input = st.text_input("Enter SMILES String", value="", placeholder="e.g., CCO")

    if st.button("Predict and Classify"):
        if not smiles_input.strip():
            st.error("Please enter a valid SMILES string.")
        else:
            # Display a spinner while processing
            with st.spinner('Processing...'):
                # IC50 Prediction
                predicted_ic50, predicted_pic50 = predict_ic50(smiles_input)

                # Classification
                classification_result = classify_compound(smiles_input)

                # Extracted Features
                features = extract_features(smiles_input)

            # Check if all results are available
            if (predicted_ic50 is not None) and (classification_result is not None) and (features is not None):
                category, _ = classification_result  # Ignore confidence

                # Display IC50 prediction
                st.success(f"**Predicted IC50**: {predicted_ic50:.4f} µM")

                # Display Classification
                st.success(f"**Classification**: {category}")

                # Display molecular features
                st.subheader("Extracted Features:")
                st.write(pd.DataFrame([features]).T)
            else:
                st.error("An error occurred during prediction or classification.")

    # Navigation buttons for another prediction or returning home
    st.write("---")  # Horizontal line for separation
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Another Prediction"):
            st.session_state.page = "predict_page"
            st.experimental_set_query_params(rerun="true")  # Simulate app rerun

    with col2:
        if st.button("Return to Home"):
            st.session_state.page = "home_page"

# ============================
# Main Application
# ============================

def main():
    """
    Main function to control navigation between pages.
    """
    if 'page' not in st.session_state:
        st.session_state.page = "home_page"  # Default to home page

    if st.session_state.page == "home_page":
        home_page()
    elif st.session_state.page == "predict_page":
        predict_page()

if __name__ == "__main__":
    main()
