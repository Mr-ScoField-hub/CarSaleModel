# Vehicle Sales Prediction Pipeline

This pipeline processes test data to predict vehicle sales probabilities and assigns priority levels to potential leads.

## Project Structure

```
Data-pipeline/
├── main_pipeline.py      # Main orchestration script
├── preprocess.py         # Data preprocessing module
├── model_loader.py       # Model loading utilities
├── predict.py            # Prediction generation
├── priority_classifier.py # Priority level classification
├── save_final_predictions.py # Final results saving
├── logs/                 # Pipeline execution logs
├── models/               # Trained model files
├── preprocessors/        # Saved preprocessor objects
└── README.md            # This documentation
```

## Pipeline Components

1. **Data Preprocessing** (`preprocess.py`)
   - Loads test data
   - Applies same preprocessing steps as training
   - Handles missing values, encoding, and scaling

2. **Model Management** (`model_loader.py`)
   - Loads pre-trained classification model
   - Supports both .pkl and .joblib formats
   - Includes model validation

3. **Prediction Generation** (`predict.py`)
   - Generates probability predictions
   - Includes error handling and validation

4. **Priority Classification** (`priority_classifier.py`)
   - Converts probabilities to priority levels
   - High: prob ≥ 0.8
   - Medium: 0.5 ≤ prob < 0.8
   - Low: prob < 0.5

5. **Results Management** (`save_final_predictions.py`)
   - Merges predictions with original data
   - Preserves all original features
   - Saves final results

6. **Pipeline Orchestration** (`main_pipeline.py`)
   - Coordinates all pipeline components
   - Includes logging and error handling
   - Provides execution tracking

## Usage

1. Ensure required files are in place:
   - TestData.csv in the Data directory
   - Trained model in models/
   - Preprocessor objects in preprocessors/

2. Run the pipeline:
   ```bash
   python main_pipeline.py
   ```

3. Check results:
   - Final predictions in scored_leads.csv
   - Execution logs in logs/

## Output

The pipeline generates:
- scored_leads.csv with:
  - Original test data features
  - VehicleSold probability predictions
  - Priority level classifications
- Detailed execution logs

## Error Handling

The pipeline includes comprehensive error handling:
- Input validation
- Data consistency checks
- Detailed error logging
- Execution tracking

## Logging

Detailed logs are generated for each run:
- Timestamp-based log files
- Progress tracking
- Error reporting
- Performance metrics