import pandas as pd
from pathlib import Path
from preprocess import load_and_preprocess_test_data
from predict import predict_probabilities
from priority_classifier import add_priority_levels
from model_loader import load_model

def merge_and_save_predictions(original_data_path, predictions_df, output_path):
    """
    Merge original test data with predictions and save to CSV.
    
    Args:
        original_data_path (str): Path to original TestData.csv
        predictions_df (pd.DataFrame): DataFrame with predictions and priority levels
        output_path (str): Path to save the final results
    """
    try:
        original_data = pd.read_csv(original_data_path)
        
        if len(original_data) != len(predictions_df):
            raise ValueError("Number of predictions doesn't match original data")
        
        final_df = pd.concat([original_data, 
                            predictions_df[['VehicleSold_Probability', 'Priority_Level']]], 
                           axis=1)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        
        print(f"\nFinal Dataset Summary:")
        print(f"Total records: {len(final_df)}")
        print(f"Columns: {', '.join(final_df.columns)}")
        print(f"\nPriority Level Distribution:")
        priority_counts = final_df['Priority_Level'].value_counts()
        for level in ['High', 'Medium', 'Low']:
            count = priority_counts.get(level, 0)
            percentage = (count / len(final_df)) * 100
            print(f"{level}: {count} ({percentage:.1f}%)")
            
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving final predictions: {str(e)}")
        raise

def main():
    try:
        test_data_path = '../Data/TestData.csv'
        model_path = './models/trained_model.pkl'
        preprocessor_path = './preprocessors'
        final_output_path = '../Data/scored_leads.csv'
        
        processed_data = load_and_preprocess_test_data(test_data_path, preprocessor_path)
        
        model = load_model(model_path)
        probabilities = predict_probabilities(processed_data, model)
        
        predictions_df = pd.DataFrame({
            'VehicleSold_Probability': probabilities
        })
        
        predictions_df = add_priority_levels(predictions_df)
        
        merge_and_save_predictions(test_data_path, predictions_df, final_output_path)
        
        print("Complete prediction pipeline executed successfully!")
        
    except Exception as e:
        print(f"Failed to generate final predictions: {str(e)}")

if __name__ == '__main__':
    main()