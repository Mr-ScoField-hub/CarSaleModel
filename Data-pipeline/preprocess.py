import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def load_and_preprocess_test_data(test_data_path, preprocessor_path):
    """
    Load and preprocess test data using saved preprocessing objects.
    
    Args:
        test_data_path (str): Path to the test data CSV file
        preprocessor_path (str): Path to the directory containing saved preprocessor objects
    
    Returns:
        pd.DataFrame: Preprocessed test data
    """
    test_data = pd.read_csv(test_data_path)
    
    try:
        preprocessor_dir = Path(preprocessor_path)
        encoder = joblib.load(preprocessor_dir / 'encoder.joblib')
        expected_columns = joblib.load(preprocessor_dir / 'expected_columns.joblib')
        processed_data = test_data.copy()

        drop_cols = ['LeadID', 'CustomerID', 'DTLeadCreated', 'DTLeadAllocated', 'OBSFullName', 'OBSEmail', 'Domain']
        processed_data.drop(columns=[col for col in drop_cols if col in processed_data.columns], inplace=True, errors='ignore')

        if 'DTLeadCreated' in test_data.columns:
            processed_data['DTLeadCreated'] = pd.to_datetime(test_data['DTLeadCreated'])
        if 'DTLeadAllocated' in test_data.columns:
            processed_data['DTLeadAllocated'] = pd.to_datetime(test_data['DTLeadAllocated'])

        if 'DTLeadCreated' in processed_data.columns:
            processed_data['IsWeekendEnquiry'] = processed_data['DTLeadCreated'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
        def time_of_day(hour):
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:
                return 'Night'
        if 'HourOfEnquiry' in processed_data.columns:
            processed_data['TimeOfDayCategory'] = processed_data['HourOfEnquiry'].apply(time_of_day)
        if 'FinanceApplied' in processed_data.columns and 'FinanceApproved' in processed_data.columns:
            processed_data['IsFinanceAppliedAndApproved'] = processed_data.apply(
                lambda row: 1 if row['FinanceApplied'] == 1 and row['FinanceApproved'] == 1 else 0, axis=1
            )

        cat_cols = processed_data.select_dtypes(include='object').columns.tolist()
        processed_data[cat_cols] = encoder.transform(processed_data[cat_cols].astype(str))

        processed_data = processed_data.reindex(columns=expected_columns, fill_value=0)
        return processed_data
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise
def main():
    test_data_path = '../Data/TestData.csv'
    preprocessor_path = './preprocessors'
    try:
        processed_test_data = load_and_preprocess_test_data(test_data_path, preprocessor_path)
        print("Test data preprocessing completed successfully!")
        print(f"Processed data shape: {processed_test_data.shape}")
    except Exception as e:
        print(f"Failed to preprocess test data: {str(e)}")

if __name__ == '__main__':
    main()