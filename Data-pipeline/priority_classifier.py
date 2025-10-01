import pandas as pd
import numpy as np

def classify_priority(probabilities):
    """
    Convert predicted probabilities into priority levels.
    
    Args:
        probabilities (np.ndarray or pd.Series): Predicted probabilities of vehicle sales
    
    Returns:
        pd.Series: Priority levels (High, Medium, Low)
    """
    prob_array = np.array(probabilities)
    priorities = np.empty(len(prob_array), dtype=object)
    priorities[prob_array >= 0.8] = 'High'
    priorities[(prob_array >= 0.5) & (prob_array < 0.8)] = 'Medium'
    priorities[prob_array < 0.5] = 'Low'
    return pd.Series(priorities, name='Priority_Level')

def add_priority_levels(predictions_df):
    """
    Add priority levels to the predictions DataFrame.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing probability predictions
                                      (must have 'VehicleSold_Probability' column)
    
    Returns:
        pd.DataFrame: DataFrame with added Priority_Level column
    """
    try:
        if 'VehicleSold_Probability' not in predictions_df.columns:
            raise ValueError("Predictions DataFrame must contain 'VehicleSold_Probability' column")
        priority_levels = classify_priority(predictions_df['VehicleSold_Probability'])
        result_df = predictions_df.copy()
        result_df['Priority_Level'] = priority_levels
        priority_counts = result_df['Priority_Level'].value_counts()
        print("\nPriority Level Distribution:")
        for level in ['High', 'Medium', 'Low']:
            count = priority_counts.get(level, 0)
            percentage = (count / len(result_df)) * 100
            print(f"{level}: {count} ({percentage:.1f}%)")
        return result_df
    except Exception as e:
        print(f"Error adding priority levels: {str(e)}")
        raise

def main():
    try:
        predictions_path = './predictions/test_predictions.csv'
        output_path = './predictions/test_predictions_with_priority.csv'
        predictions_df = pd.read_csv(predictions_path)
        result_df = add_priority_levels(predictions_df)
        result_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    except Exception as e:
        print(f"Priority classification failed: {str(e)}")

if __name__ == '__main__':
    main()