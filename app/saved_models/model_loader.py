import joblib
import os
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../../models")

loaded_models: Dict[str, Any] = {}

def load_model(model_name: str):
    """
    Load a specific model and its vectorizer by name.
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name}_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, f"{model_name}_vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Model or vectorizer for '{model_name}' not found")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


def get_model(model_name: str):
    """
    Retrieve the model and vectorizer from memory,
    or load them from disk if not already loaded.
    """
    if model_name not in loaded_models:
        model, vectorizer = load_model(model_name)
        loaded_models[model_name] = (model, vectorizer)
    return loaded_models[model_name]
