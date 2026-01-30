from pathlib import Path
from src.model import FarmerIncomePredictor
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pandas as pd

def main():
    """Main execution function"""
    
    print("="*80)
    print("FARMER INCOME PREDICTION MODEL")
    print("="*80)
    
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"

    TRAIN_PATH  = DATA_DIR / "train_data.csv"
    TEST_PATH = DATA_DIR / "test_data.csv"
    
    # Initialize predictor
    predictor = FarmerIncomePredictor(random_state=42)
    
    # Load data
    predictor.load_data(TRAIN_PATH, TEST_PATH)
    
    # Separate target variable
    y_train = predictor.train_df['Target_Variable/Total Income']
    X_train_raw = predictor.train_df.drop(['Target_Variable/Total Income', 'FarmerID'], axis=1)
    X_test_raw = predictor.test_df.drop(['FarmerID'], axis=1)
    
    # Feature engineering
    X_train_engineered = predictor.engineer_features(X_train_raw, is_train=True)
    X_test_engineered = predictor.engineer_features(X_test_raw, is_train=False)
    
    # Preprocessing
    X_train_processed = predictor.preprocess_data(X_train_engineered, is_train=True)
    X_test_processed = predictor.preprocess_data(X_test_engineered, is_train=False)
    
    # Align columns
    common_cols = X_train_processed.columns.intersection(X_test_processed.columns)
    X_train_aligned = X_train_processed[common_cols]
    X_test_aligned = X_test_processed[common_cols]
    
    # Feature selection
    X_train_selected, X_test_selected, selected_features = predictor.select_features(
        X_train_aligned, y_train, X_test_aligned
    )
    predictor.selected_features = selected_features
    
    # Split for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_selected, y_train, test_size=0.2, random_state=42
    )
    
    # Train models
    predictor.train_models(X_train_split, y_train_split)
    
    # Evaluate on validation set
    val_predictions = predictor.predict_ensemble(X_val)
    print("\n" + "="*80)
    print("VALIDATION SET PERFORMANCE")
    print("="*80)
    metrics = predictor.evaluate(y_val, val_predictions)
    
    #train on full training data
    print("\n" + "="*80)
    print("TRAINING ON FULL DATASET")
    print("="*80)
    predictor.train_models(X_train_selected, y_train)
    
    # Make predictions on test set
    print("\n" + "="*80)
    print("GENERATING TEST PREDICTIONS")
    print("="*80)
    test_predictions = predictor.predict_ensemble(X_test_selected)
    
    #create submission file
    submission = pd.DataFrame({
        'FarmerID': predictor.test_farmer_ids,
        'Target_Variable/Total Income': test_predictions
    })
    
    submission_filename = f'farmer_income_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    submission.to_csv(submission_filename, index=False)
    print(f"\nPredictions saved to: {submission_filename}")
    
    #save model
    predictor.save_model('farmer_income_model.pkl')
    
    #save feature importance
    if predictor.feature_importance is not None:
        predictor.feature_importance.to_csv('feature_importance.csv', index=False)
        print("\nTop 20 Most Important Features:")
        print(predictor.feature_importance.head(20))
    
    print("\n" + "="*80)
    print("PREDICTION COMPLETE!")
    print("="*80)
    
    return predictor, submission, metrics

if __name__ == "__main__":
    #install required packages if not already installed
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 
        'xgboost', 'lightgbm'
    ]
    
    print("Checking required packages...")
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"pip install {package} --break-system-packages -q")
    
    # Run main prediction pipeline
    predictor, submission, metrics = main()
