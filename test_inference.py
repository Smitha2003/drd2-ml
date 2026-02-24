from src.inference import DRD2Predictor

predictor = DRD2Predictor(
    model_path="models/rf_morgan_model.pkl",
    config_path="models/model_config.pkl",
    training_fp_path="models/training_fingerprints.pkl"
)

result = predictor.predict("CCN(CC)CC")
print(result)