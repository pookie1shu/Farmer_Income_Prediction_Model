# Farmer_Income_Prediction_Model


## Overview
This project implements a machine learning solution to predict farmer income using demographic, agricultural, climatic, and socio-economic features. The solution employs an ensemble of state-of-the-art gradient boosting algorithms to achieve robust predictions.

## Problem Statement
**Objective:** Predict farmer income using 105 features spanning multiple data categories

**Key Challenges:**
- Complex interplay of factors affecting agricultural income
- High-dimensional feature space requiring careful engineering
- Temporal variations in climate and agricultural performance (2020-2022)
- Regional disparities in infrastructure and market access
- Missing and inconsistent data patterns

**Evaluation Metric:** Mean Absolute Percentage Error (MAPE)

[View Project Presentation on Canva](https://www.canva.com/design/DAG7p3j2K9U/1r7EwVgjWYYSgqTXqKCAUw/edit?utm_content=DAG7p3j2K9U&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)


## Dataset Description

### Feature Categories (105 total features)

1. **Demographics (13 features)**
   - Location identifiers: State, Region, District, Village, City, Zipcode
   - Personal: Sex, Marital Status, Address Type, Ownership

2. **Financial (3 features)**
   - Active loans in bureau
   - Average disbursement amount
   - Non-agricultural income

3. **Land Holdings (1 feature)**
   - Total land for agriculture

4. **Village Infrastructure & Socio-Economic (14 features)**
   - Village categories (agricultural and socio-economic)
   - Proximity to mandi and railway
   - Village scores (0-100 scale)
   - Housing quality indicators
   - Sanitation and electricity access

5. **Climate Data - Multi-Year (12 features)**
   - Seasonal rainfall (mm) for Kharif and Rabi seasons
   - Ambient temperature ranges (min & max) for 2020, 2021, 2022

6. **Agricultural Performance - Multi-Year (60+ features)**
   - Irrigated area for both seasons across years
   - Cropping density
   - Agricultural performance scores
   - Agricultural scores (composite metrics)
   - Soil types
   - Water bodies data
   - Agro-ecological sub-zones
   - Groundwater thickness and replenishment rates

7. **Infrastructure & Development (3 features)**
   - Night light index
   - Land holding index
   - Road density (km/sq km)

### Temporal Coverage
- **Years:** 2020, 2021, 2022
- **Seasons:** Kharif (monsoon) and Rabi (winter) 
- **Total Temporal Points:** 6 season-year combinations per farmer

## Methodology

### 1. Feature Engineering

Our feature engineering strategy creates 30+ derived features:

#### Temperature Features
- Extract min and max from temperature range strings
- Calculate temperature range (max - min)
- Compute seasonal average temperatures

#### Agricultural Aggregation
- **Multi-year averages:** Mean agricultural scores across all seasons and years
- **Performance trends:** Score changes from 2020 to 2022
- **Consistency metrics:** Standard deviation of agricultural metrics
- **Season-specific scores:** Separate averages for Kharif and Rabi seasons

#### Water Resources
- **Irrigation metrics:** Average irrigated area across years
- **Irrigation consistency:** Standard deviation to measure reliability
- **Groundwater features:** Average thickness and replenishment rates
- **Water availability index:** Combined metric of irrigation and groundwater

#### Climate Analysis
- **Rainfall statistics:** Mean, max, min across all seasons
- **Rainfall variability:** Standard deviation to capture unpredictability
- **Temperature patterns:** Average and range across years

#### Land Productivity
- **Income per land:** Non-agricultural income divided by land area
- **Land utilization:** Agricultural area as percentage of total geographical area
- **Productivity potential:** Agricultural score multiplied by land area

#### Market Access
- **Market accessibility:** Inverse distance to nearest mandi
- **Railway accessibility:** Inverse distance to nearest railway
- **Combined accessibility:** Weighted combination of market and transport access

#### Infrastructure Development
- **Infrastructure index:** Road density multiplied by night light index
- **Development score:** Combined measure of infrastructure quality

#### Financial Metrics
- **Total credit exposure:** Number of loans times average disbursement
- **Credit per land:** Average disbursement divided by agricultural land
- **Credit accessibility:** Relative credit availability metrics

#### Socio-Economic Features
- **Housing quality index:** Average of housing-related indicators
- **Socio-economic categories:** Binned village scores (Low/Medium/High)

#### Interaction Features
- **Land-rainfall interaction:** Captures how land size interacts with rainfall
- **Productivity potential:** Agricultural score times land area
- **Season-specific interactions:** Kharif and Rabi performance combinations

### 2. Data Preprocessing

#### Missing Value Handling
- **Numerical features:** Median imputation to handle outliers
- **Categorical features:** Label encoding with unknown category support

#### Feature Encoding
- **Label Encoding:** All categorical variables converted to numeric codes
- **Unknown Category Handling:** New categories in test set mapped to dedicated "Unknown" class
- **Encoding Preservation:** All encoders saved for consistent test set transformation

#### Feature Scaling
- **Robust Scaler:** Used to handle outliers effectively
- **Per-feature scaling:** Each feature scaled independently
- **Preservation:** Scaler saved for test set transformation

#### Feature Selection
- **Random Forest Importance:** Quick feature ranking using RF
- **Top-N Selection:** Select top 100 features by importance
- **Dimensionality Reduction:** Reduce noise while retaining predictive power

### 3. Model Architecture: Ensemble Approach

We employ a weighted ensemble of six algorithms:

#### Primary Models (75% weight)

**LightGBM (25% weight)**
- Leaf-wise tree growth for better accuracy
- Fast training on large datasets
- Excellent handling of categorical features
- Hyperparameters:
  - n_estimators: 1000
  - learning_rate: 0.05
  - max_depth: 8
  - num_leaves: 31
  - subsample: 0.8
  - colsample_bytree: 0.8

**XGBoost (25% weight)**
- Depth-wise tree growth for balanced trees
- Strong regularization capabilities
- Excellent generalization
- Hyperparameters:
  - n_estimators: 1000
  - learning_rate: 0.05
  - max_depth: 8
  - subsample: 0.8
  - colsample_bytree: 0.8

**CatBoost (25% weight)**
- Superior categorical feature handling
- Ordered boosting reduces overfitting
- Robust to parameter choices
- Hyperparameters:
  - iterations: 1000
  - learning_rate: 0.05
  - depth: 8

#### Supporting Models (25% weight)

**Random Forest (10% weight)**
- Bagging ensemble reduces variance
- Robust to outliers and noise
- Parallel training capability
- Hyperparameters:
  - n_estimators: 500
  - max_depth: 15
  - min_samples_split: 5
  - min_samples_leaf: 2

**Extra Trees (10% weight)**
- Higher randomization than Random Forest
- Captures different patterns
- Additional variance reduction
- Hyperparameters: Same as Random Forest

**Histogram Gradient Boosting (5% weight)**
- Fast training with histogram binning
- Good baseline performance
- Hyperparameters:
  - max_iter: 1000
  - learning_rate: 0.05
  - max_depth: 8

#### Ensemble Strategy
- **Weighted averaging:** Each model contributes based on expected performance
- **Diversity maximization:** Different algorithms capture different patterns
- **Bias-variance tradeoff:** Combination reduces both bias and variance
- **Robustness:** Ensemble mitigates individual model failures

### 4. Training Strategy

#### Validation Approach
1. **Initial Split:** 80% training, 20% validation
2. **Performance Evaluation:** Measure MAPE, MAE, RMSE, R² on validation set
3. **Model Comparison:** Assess individual model contributions
4. **Final Training:** Retrain on full dataset for final predictions

#### Hyperparameter Rationale
- **1000 estimators:** Sufficient for convergence without excessive overfitting
- **0.05 learning rate:** Balance between training speed and optimization
- **Max depth 8:** Prevent overfitting while capturing interactions
- **Subsample 0.8:** Stochastic training for better generalization
- **Column sampling 0.8:** Feature randomness prevents overfitting

### 5. Prediction Generation

1. Load and preprocess test data using saved encoders
2. Apply feature engineering pipeline
3. Transform using saved scaler
4. Select same features as training
5. Generate predictions from each model
6. Apply weighted ensemble
7. Output predictions in required format

## Implementation Details

### Technical Stack
- **Language:** Python 3.8+
- **Core Libraries:** pandas, numpy, scikit-learn
- **ML Frameworks:** LightGBM, XGBoost, CatBoost
- **Development:** Modular class-based design

### Code Structure

```
FarmerIncomePredictor/
│
├── __init__()              # Initialize with random seed
├── load_data()             # Load train and test CSV files
├── engineer_features()     # Create 30+ derived features
├── preprocess_data()       # Handle missing values and encoding
├── select_features()       # RF-based feature selection
├── train_models()          # Train all six models
├── predict_ensemble()      # Weighted ensemble prediction
├── evaluate()              # Calculate MAPE, MAE, RMSE, R²
├── save_model()            # Serialize trained models
└── load_model()            # Load saved models
```

### Key Features
- **Reproducibility:** Fixed random seeds (seed=42)
- **Modularity:** Each step is a separate method
- **Extensibility:** Easy to add new models or features
- **Serialization:** Save/load trained models for reuse
- **Error Handling:** Robust handling of edge cases

## Usage Instructions

### Installation

```bash
# Install required packages
pip install pandas numpy scikit-learn xgboost lightgbm catboost --break-system-packages
```

### Running the Model

```python
# Import and initialize
from farmer_income_prediction import FarmerIncomePredictor
predictor = FarmerIncomePredictor(random_state=42)

# Load data
predictor.load_data('train_data.csv', 'test_data.csv')

# Run complete pipeline
predictor, submission, metrics = main()

# Outputs:
# - farmer_income_predictions_YYYYMMDD_HHMMSS.csv (predictions)
# - farmer_income_model.pkl (saved model)
# - feature_importance.csv (feature rankings)
```

### Command Line Execution

```bash
python farmer_income_prediction.py
```

### Expected Outputs

1. **Predictions File:** CSV with FarmerID and predicted income
2. **Model File:** Serialized trained model (.pkl)
3. **Feature Importance:** CSV ranking features by importance
4. **Console Output:** Training progress and performance metrics

## Performance Metrics

### Evaluation Metrics

- **MAPE (Primary):** Mean Absolute Percentage Error - percentage deviation from actual
- **MAE:** Mean Absolute Error - average absolute difference
- **RMSE:** Root Mean Squared Error - penalizes large errors
- **R²:** Coefficient of determination - variance explained

### Expected Performance
- Models typically achieve competitive MAPE scores
- Ensemble approach reduces variance compared to single models
- Feature engineering contributes significantly to performance

## Key Insights from Analysis

### Primary Income Drivers
1. **Total Agricultural Land:** Strongest predictor of income potential
2. **Multi-year Agricultural Scores:** Consistent performance indicates stability
3. **Irrigation Access:** Critical for reliable crop production
4. **Market Proximity:** Shorter distance to mandi improves price realization

### Regional Factors
- Village socio-economic scores capture local development levels
- Infrastructure (roads, electricity) correlates with market integration
- Climate patterns show strong seasonal and annual variation

### Financial Indicators
- Active loan counts suggest investment capacity
- Non-agricultural income provides diversification
- Credit accessibility varies by region and land size

## Future Enhancements

### Model Improvements
1. **Deep Learning:** Neural networks for complex pattern recognition
2. **Time Series Models:** LSTM/GRU for temporal dependencies
3. **Hyperparameter Optimization:** Bayesian or grid search
4. **Stacking:** Meta-learner on base model predictions

### Data Enrichment
1. **Satellite Imagery:** Crop health via NDVI indices
2. **Market Prices:** Historical commodity price integration
3. **Weather Forecasts:** Forward-looking climate predictions
4. **Government Schemes:** Policy intervention impact analysis
5. **Social Networks:** Cooperative participation metrics

### Deployment
1. **API Development:** Real-time prediction service
2. **Dashboard:** Interactive visualization for stakeholders
3. **Mobile App:** Field-level access for extension workers
4. **Automated Retraining:** Periodic model updates with new data

## Reproducibility

### Random Seeds
- All models use `random_state=42`
- Ensures consistent results across runs
- Important for model comparison and debugging

### Data Splits
- Train-validation split: 80-20
- Stratification: Not applicable for regression
- Final training: Full dataset after validation

### Model Serialization
- All trained models saved to `.pkl` file
- Label encoders and scalers included
- Feature selection preserved
- Reproducible predictions on new data

## License
This project is developed for agricultural income prediction research and applications.

---
**Last Updated:** December 2025
**Version:** 1.0
