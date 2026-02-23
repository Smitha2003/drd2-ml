import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import matplotlib.pyplot as plt
import shap
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw

from src.inference import DRD2Predictor
from src.features import smiles_to_fingerprint


# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="DRD2 Affinity Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        /* Main background and text styling */
        .main {
            padding: 2rem;
        }
        
        /* Title styling */
        h1 {
            color: #00d9ff !important;
            text-align: center;
            font-size: 3em;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }
        
        /* Subtitle styling */
        .subtitle {
            text-align: center;
            color: #ffffff !important;
            font-size: 1.1em;
            margin-bottom: 2rem;
            line-height: 1.6;
            font-weight: 500;
        }
        
        /* Section headers */
        h2 {
            color: #00d9ff !important;
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.5rem;
            font-size: 1.8em;
        }
        
        h3 {
            color: #ffffff !important;
            font-size: 1.3em;
            font-weight: 600;
        }
        
        /* Metric card styling */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        /* Input styling */
        .stTextInput input {
            border-radius: 8px;
            padding: 0.75rem;
            border: 2px solid #667eea;
            font-size: 1rem;
            background-color: #1e1e2e;
        }
        
        .stTextInput input::placeholder {
            color: #888;
        }
        
        .stTextInput input:focus {
            border: 2px solid #00d9ff;
            box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.2);
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 700;
            font-size: 1em;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.6);
            transform: translateY(-2px);
        }
        
        /* Divider */
        hr {
            margin: 2rem 0;
            border: none;
            border-top: 2px solid #333;
        }
        
        /* Info boxes */
        .info-box {
            background-color: #1e3a5f;
            border-left: 4px solid #00d9ff;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)


# -------------------------
# Load Predictor (cached)
# -------------------------
@st.cache_resource
def load_predictor():
    return DRD2Predictor(
        model_path="models/rf_morgan_model.pkl",
        config_path="models/model_config.pkl",
        training_fp_path="models/training_fingerprints.pkl"
    )

predictor = load_predictor()


# -------------------------
# Load SHAP Explainer (cached)
# -------------------------
@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_shap_explainer(predictor.model)


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown("<h3 style='color: #00d9ff; font-weight: 700;'>📊 Model Information</h3>", unsafe_allow_html=True)
    
    # Model info in a more organized way with explicit styling
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p style='color: #ffffff; font-weight: 700; margin-bottom: 0;'><b>Model</b></p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #e0e0e0; margin-top: 0;'>Random Forest</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("<p style='color: #ffffff; font-weight: 700; margin-bottom: 0;'><b>Target</b></p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #e0e0e0; margin-top: 0;'>pIC50 (DRD2)</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p style='color: #ffffff; font-weight: 700; margin-bottom: 0;'><b>Fingerprint</b></p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #e0e0e0; margin-top: 0;'>Morgan (ECFP)</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("<p style='color: #ffffff; font-weight: 700; margin-bottom: 0;'><b>Bits</b></p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #e0e0e0; margin-top: 0;'>990</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p style='color: #ffffff; font-weight: 700; margin-bottom: 0;'><b>Radius</b></p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #e0e0e0; margin-top: 0;'>2</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("<p style='color: #ffffff; font-weight: 700; margin-bottom: 0;'><b>Dataset</b></p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #e0e0e0; margin-top: 0;'>ChEMBL</p>", unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown(
        "<div class='info-box'>"
        "Trained on curated ChEMBL DRD2 IC50 dataset with advanced feature engineering."
        "</div>",
        unsafe_allow_html=True
    )
    
    st.divider()
    
    st.markdown("<h3 style='color: #00d9ff; font-weight: 700;'>🔍 Feature Importance</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #ffffff; font-weight: 500;'>View which molecular features most influence predictions</p>", unsafe_allow_html=True)

    if st.button("📈 Show Global SHAP Summary", use_container_width=True):
        with st.spinner("Computing global SHAP values..."):

            shap_values_global = explainer.shap_values(predictor.training_fps)

            fig = plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values_global,
                predictor.training_fps,
                show=False
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# -------------------------
# Main Title
# -------------------------
st.markdown(
    "<h1>🧬 DRD2 Binding Affinity Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>"
    "Predict dopamine D2 receptor binding affinity (pIC50) from molecular SMILES "
    "<br> using an explainable Random Forest model"
    "</div>",
    unsafe_allow_html=True
)

st.divider()


# -------------------------
# SMILES Input
# -------------------------
st.markdown("### 🧪 Enter Molecular SMILES")
st.write("Provide a SMILES string to predict DRD2 binding affinity")

col_input, col_btn = st.columns([4, 1])

with col_input:
    smiles_input = st.text_input(
        "SMILES Input",
        placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O",
        label_visibility="collapsed"
    )

with col_btn:
    st.write("")  # Spacer for alignment
    calculate_clicked = st.button("🔬 Calculate", use_container_width=True, key="calc_btn")

# -------------------------
# Prediction Logic
# -------------------------
if smiles_input and calculate_clicked:

    result = predictor.predict(smiles_input)

    if "error" in result:
        st.error(f"❌ {result['error']}")

    else:
        st.success("✓ Prediction successful!")
        st.divider()
        
        col_mol, col_pred = st.columns([1, 1], gap="large")

        # LEFT COLUMN → Molecule
        with col_mol:
            st.markdown("### 🔬 Molecular Structure")
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                img = Draw.MolToImage(mol, size=(400, 400))
                st.image(img, use_column_width=True, caption="2D Structure")
            else:
                st.warning("⚠️ Invalid molecule structure.")

        # RIGHT COLUMN → Prediction Panel
        with col_pred:
            st.markdown("### 📊 Prediction Results")
            
            # Prediction metric with custom styling
            st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0; color: white;'>Predicted pIC50</h3>
                    <h1 style='margin: 0.5rem 0; color: white; font-size: 3em;'>{result['prediction']:.2f}</h1>
                </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Results breakdown
            col_sim, col_conf = st.columns(2)
            
            with col_sim:
                st.markdown("**Similarity Score**")
                similarity = result['similarity']
                st.markdown(f"<h2 style='color: #667eea; margin: 0;'>{similarity:.2f}</h2>", unsafe_allow_html=True)
                st.caption("Training set similarity")
            
            with col_conf:
                st.markdown("**Confidence Level**")
                confidence = result["confidence"]
                
                # Color-coded confidence
                if confidence == "High":
                    conf_color = "#28a745"
                    conf_icon = "✅"
                elif confidence == "Moderate":
                    conf_color = "#ffc107"
                    conf_icon = "⚠️"
                else:  # Low
                    conf_color = "#dc3545"
                    conf_icon = "⛔"
                
                st.markdown(f"<h2 style='color: {conf_color}; margin: 0;'>{conf_icon} {confidence}</h2>", unsafe_allow_html=True)

        st.divider()
        
        # -------------------------
        # Per-Molecule SHAP
        # -------------------------
        st.markdown("### 🔍 Prediction Explanation (SHAP)")
        st.write("Feature contribution analysis - how atomic properties influence the prediction")
        
        with st.expander("📈 View SHAP Waterfall Plot", expanded=False):

            mol_obj, features = smiles_to_fingerprint(
                smiles_input,
                predictor.generator,
                predictor.n_bits
            )

            if features is not None:

                shap_values = explainer.shap_values(
                    np.array(features).reshape(1, -1)
                )

                # For sklearn regression, shap_values is array
                shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

                explanation = shap.Explanation(
                    values=shap_vals,
                    base_values=explainer.expected_value,
                    data=features
                )

                fig = plt.figure(figsize=(12, 6))
                shap.plots.waterfall(explanation, show=False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)