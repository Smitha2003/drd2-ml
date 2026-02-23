import joblib
from .features import (
    initialize_generator,
    smiles_to_fingerprint,
    numpy_to_bitvect
)
from .similarity import compute_max_similarity


class DRD2Predictor:

    def __init__(self, model_path, config_path, training_fp_path):
        """
        Load model, configuration, and training fingerprints.
        Initialize fingerprint generator.
        """
        self.model = joblib.load(model_path)
        self.config = joblib.load(config_path)
        self.training_fps = joblib.load(training_fp_path)

        self.radius = self.config["radius"]
        self.n_bits = self.config["n_bits"]

        self.generator = initialize_generator(self.radius, self.n_bits)

        # Convert training fingerprints once to RDKit bitvectors
        self.training_bitvectors = [
            numpy_to_bitvect(fp) for fp in self.training_fps
        ]

    def predict(self, smiles: str):
        """
        Predict pIC50 and compute similarity-based confidence.
        """

        mol, features = smiles_to_fingerprint(
            smiles,
            self.generator,
            self.n_bits
        )

        if mol is None:
            return {"error": "Invalid SMILES"}

        prediction = self.model.predict([features])[0]

        input_bv = numpy_to_bitvect(features)

        max_similarity = compute_max_similarity(
            input_bv,
            self.training_bitvectors
        )

        # Confidence logic
        if max_similarity > 0.7:
            confidence = "High"
        elif max_similarity > 0.5:
            confidence = "Moderate"
        elif max_similarity > 0.3:
            confidence = "Low"
        else:
            confidence = "Very Low"

        return {
            "prediction": float(prediction),
            "similarity": float(max_similarity),
            "confidence": confidence
        }