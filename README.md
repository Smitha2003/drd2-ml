# DRD2 Binding Affinity Prediction — Explainable ML Regression

An end-to-end regression pipeline for predicting Dopamine D2 receptor (DRD2) binding affinity (pIC50) from molecular structure representations.

This project focuses on applied machine learning practices including data curation, feature engineering, cross-validated model training, interpretability, and clean deployment.

---

## Objective

To model the relationship between molecular structure and experimentally measured binding affinity using interpretable tree-based regression.

The emphasis is on:

* Working with noisy real-world scientific data
* Evaluating model generalization properly
* Inspecting model behavior through structured error analysis
* Providing transparent predictions via feature attribution

---

## Approach Overview

### Data Preparation

* Bioactivity data retrieved from ChEMBL
* Filtered to consistent IC50 measurements
* Converted to pIC50 to stabilize distribution
* Duplicate and invalid structures removed

### Molecular Representation

* Circular fingerprints (Morgan/ECFP)
* Structured feature generation pipeline
* Deterministic preprocessing for reproducibility

### Modeling Strategy

* Tree-based ensemble regression
* K-fold cross-validation
* Hyperparameter optimization
* Learning curve inspection

### Evaluation

* RMSE, MAE, R²
* Cross-fold performance consistency
* Residual analysis to assess prediction behavior

---

## Model Interpretability

Model predictions are analyzed using SHAP to provide:

* Global feature importance insights
* Local, per-molecule contribution breakdown

This allows inspection of how structural patterns influence predictions.

---

## Interactive Inference App

A lightweight Streamlit interface enables:

* SMILES input
* Molecular structure visualization
* Predicted pIC50 output
* Confidence indicator based on structural similarity
* Expandable SHAP explanation panel

Training and inference pipelines are fully separated.

---

## Project Structure

```
drd2-ml/
├── notebooks/      # Experimentation & analysis
├── src/            # Feature + model pipeline
├── models/         # Serialized model artifacts
├── app/            # Streamlit interface
└── requirements.txt
```

Design principles:

* Modular architecture
* No retraining during inference
* Reproducible preprocessing
* Clear separation of concerns

---

## Setup

Python 3.10 recommended.

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/streamlit_app.py
```

---

## Additional Notes

* Experimental bioactivity measurements contain inherent variability.
* Predictions outside the training distribution are flagged using structural similarity thresholds.
* This project is intended as a modeling study and not as a replacement for experimental validation.
