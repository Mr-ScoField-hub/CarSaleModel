import joblib
import pickle
from pathlib import Path

def load_model(model_path):
    """
    Load a pre-trained classification model from a pickle or joblib file.
    
    Args:
        model_path (str): Path to the saved model file (.pkl or .joblib)
    
    Returns:
        object: Loaded classification model
    
    Raises:
        FileNotFoundError: If the model file doesn't exist
        Exception: If there's an error loading the model
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    try:
        model = joblib.load(model_path)
        if not hasattr(model, 'predict'):
            raise AttributeError("Loaded object does not appear to be a valid machine learning model")
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def main():
    try:
        model_path = './models/trained_model.pkl'
        model = load_model(model_path)
        print(f"Model type: {type(model).__name__}")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")

if __name__ == '__main__':
    main()