import pandas as pd
import numpy as np
from pathlib import Path
from preprocess import load_and_preprocess_test_data
from model_loader import load_model

def predict_probabilities(test_data, model):
    """
    Predict probability of VehicleSold class for each test sample.
    
    Args:
        test_data (pd.DataFrame): Preprocessed test dataset
        model: Loaded classification model with predict_proba method
    
    Returns:
        np.ndarray: Probability predictions for VehicleSold class
    """
    try:
        if not hasattr(model, 'predict_proba'):
            raise AttributeError("Model does not support probability predictions")
        probabilities = model.predict_proba(test_data)
        if probabilities.shape[1] == 2:
            return probabilities[:, 1]
        else:
            raise ValueError("Model output does not match expected binary classification")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

def save_predictions(predictions, output_path, original_ids=None):
    """
    Save predictions to a CSV file.
    
    Args:
        predictions (np.ndarray): Array of probability predictions
        output_path (str): Path to save the predictions
        original_ids (pd.Series, optional): Original IDs or index from test data
    """
    try:
        results = pd.DataFrame({
            'ID': original_ids if original_ids is not None else range(len(predictions)),
            'VehicleSold_Probability': predictions
        })
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    except Exception as e:
        print(f"Error saving predictions: {str(e)}")
        raise

def main():
    try:
        test_data_path = '../Data/TestData.csv'
        model_path = './models/trained_model.pkl'
        preprocessor_path = './preprocessors'
        output_path = './predictions/test_predictions.csv'
        test_data = load_and_preprocess_test_data(test_data_path, preprocessor_path)
        original_ids = test_data['ID'] if 'ID' in test_data.columns else None
        model = load_model(model_path)
        predictions = predict_probabilities(test_data, model)
        save_predictions(predictions, output_path, original_ids)
        print("Prediction process completed successfully!")
    except Exception as e:
        print(f"Prediction pipeline failed: {str(e)}")

if __name__ == '__main__':
    main()