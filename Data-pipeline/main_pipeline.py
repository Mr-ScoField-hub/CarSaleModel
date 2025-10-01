import os
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from preprocess import load_and_preprocess_test_data
from model_loader import load_model
from predict import predict_probabilities
from priority_classifier import add_priority_levels
from save_final_predictions import merge_and_save_predictions

# Configure logging
def setup_logging(log_dir='./logs'):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'prediction_pipeline_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )




def run_prediction_pipeline(config):
    """
    Execute the complete prediction pipeline.
    
    Args:
        config (dict): Configuration containing all necessary paths and parameters
    """
    try:
        logging.info("Starting prediction pipeline")
        
        # Step 1:
        logging.info("Loading and preprocessing test data")
        processed_data = load_and_preprocess_test_data(
            config['test_data_path'],
            config['preprocessor_path']
        )
        logging.info(f"Preprocessed data shape: {processed_data.shape}")

    
        
        # Step 2:
        logging.info("Loading trained model")
        model = load_model(config['model_path'])
        logging.info(f"Loaded model type: {type(model).__name__}")
        
        # Step 3:
        logging.info("Generating probability predictions")
        probabilities = predict_probabilities(processed_data, model)
        predictions_df = pd.DataFrame({'VehicleSold_Probability': probabilities})
        
        # Step 4:
        logging.info("Classifying priority levels")
        predictions_df = add_priority_levels(predictions_df)
        
        # Step 5:
        logging.info("Saving final results")
        merge_and_save_predictions(
            config['test_data_path'],
            predictions_df,
            config['output_path']
        )
        
        logging.info("Pipeline completed successfully!")
        return True
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise
  

def main():
    # Define configuration
    config = {
        'test_data_path': '../Data/TestData.csv',
        'model_path': './models/trained_model.pkl',
        'preprocessor_path': './preprocessors',
        'output_path': '../Data/scored_leads.csv'
    }
    
    # Create necessary directories
    for path in ['./models', './preprocessors', './logs']:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging()
    
    try:
        # Run the pipeline
        success = run_prediction_pipeline(config)
        
        if success:
            logging.info("Pipeline execution completed successfully")
            print("\nPrediction pipeline completed! Check the logs for details.")
            print(f"Final results saved to: {config['output_path']}")
    
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        print(f"\nError: Pipeline execution failed. Check the logs for details.")

if __name__ == '__main__':
    main()