import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#from pathlib import Path

# Machine Learning Libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                               ExtraTreesRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Utilities
import pickle
from datetime import datetime


class FarmerIncomePredictor:
    """
    A comprehensive class for predicting farmer income using ensemble methods
    and advanced feature engineering.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = RobustScaler()
        self.models = {}
        self.feature_importance = None
        self.categorical_features = []
        self.numerical_features = []
        
    def load_data(self, TRAIN_PATH, TEST_PATH):
        """Load training and test datasets"""
        print("Loading datasets...")
        self.train_df = pd.read_csv(TRAIN_PATH)
        self.test_df = pd.read_csv(TEST_PATH)
        
        print(f"Training data shape: {self.train_df.shape}")
        print(f"Test data shape: {self.test_df.shape}")
        
        # Store FarmerID for final submission
        self.test_farmer_ids = self.test_df['FarmerID'].copy()
        
        return self
    
    def extract_temperature_features(self, df, temp_column):
        """Extract min and max temperature from temperature range string"""
        if temp_column not in df.columns:
            return df
        
        try:
            # Handle format like "15-25" or "15 to 25"
            df[f'{temp_column}_min'] = df[temp_column].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
            df[f'{temp_column}_max'] = df[temp_column].astype(str).str.extract(r'[-–to]+\s*(\d+\.?\d*)')[0].astype(float)
            df[f'{temp_column}_range'] = df[f'{temp_column}_max'] - df[f'{temp_column}_min']
            df[f'{temp_column}_avg'] = (df[f'{temp_column}_max'] + df[f'{temp_column}_min']) / 2
        except:
            pass
        
        return df
    
    def engineer_features(self, df, is_train=True):
        """Comprehensive feature engineering"""
        print("Engineering features...")
        df = df.copy()
        
        # 1. Temperature Feature Extraction
        temp_columns = [col for col in df.columns if 'temperature' in col.lower()]
        for temp_col in temp_columns:
            df = self.extract_temperature_features(df, temp_col)
        
        # 2. Agricultural Performance Features
        # Average agricultural scores across years and seasons
        agri_score_cols = [col for col in df.columns if 'Agricultural Score' in col]
        if agri_score_cols:
            df['avg_agri_score'] = df[agri_score_cols].mean(axis=1)
            df['max_agri_score'] = df[agri_score_cols].max(axis=1)
            df['min_agri_score'] = df[agri_score_cols].min(axis=1)
            df['agri_score_trend'] = df[agri_score_cols].iloc[:, -1] - df[agri_score_cols].iloc[:, 0] if len(agri_score_cols) > 1 else 0
        
        # 3. Irrigation and Water Features
        irrigated_cols = [col for col in df.columns if 'Irrigated area' in col]
        if irrigated_cols:
            df['avg_irrigated_area'] = df[irrigated_cols].mean(axis=1)
            df['irrigation_consistency'] = df[irrigated_cols].std(axis=1)
        
        # Groundwater features
        groundwater_thickness_cols = [col for col in df.columns if 'groundwater thickness' in col.lower()]
        if groundwater_thickness_cols:
            df['avg_groundwater_thickness'] = df[groundwater_thickness_cols].mean(axis=1)
        
        groundwater_replenishment_cols = [col for col in df.columns if 'groundwater replenishment' in col.lower()]
        if groundwater_replenishment_cols:
            df['avg_groundwater_replenishment'] = df[groundwater_replenishment_cols].mean(axis=1)
        
        # 4. Rainfall Features
        rainfall_cols = [col for col in df.columns if 'Rainfall' in col]
        if rainfall_cols:
            df['avg_rainfall'] = df[rainfall_cols].mean(axis=1)
            df['max_rainfall'] = df[rainfall_cols].max(axis=1)
            df['min_rainfall'] = df[rainfall_cols].min(axis=1)
            df['rainfall_variability'] = df[rainfall_cols].std(axis=1)
        
        # 5. Cropping Density Features
        cropping_density_cols = [col for col in df.columns if 'Cropping density' in col]
        if cropping_density_cols:
            df['avg_cropping_density'] = df[cropping_density_cols].mean(axis=1)
            df['cropping_intensity_change'] = df[cropping_density_cols].iloc[:, -1] - df[cropping_density_cols].iloc[:, 0] if len(cropping_density_cols) > 1 else 0
        
        # 6. Land and Agricultural Area Features
        if 'Total_Land_For_Agriculture' in df.columns:
            df['land_productivity'] = df['Non_Agriculture_Income'] / (df['Total_Land_For_Agriculture'] + 1)
        
        if 'K022-Net Agri area (in Ha)-' in df.columns and 'K022-Total Geographical Area (in Hectares)-' in df.columns:
            df['agri_land_utilization'] = df['K022-Net Agri area (in Ha)-'] / (df['K022-Total Geographical Area (in Hectares)-'] + 1)
        
        # 7. Socio-Economic Features
        if 'KO22-Village score based on socio-economic parameters (0 to 100)' in df.columns:
            df['socioeconomic_category'] = pd.cut(df['KO22-Village score based on socio-economic parameters (0 to 100)'], 
                                                   bins=[0, 33, 66, 100], 
                                                   labels=['Low', 'Medium', 'High'])
        
        # 8. Market Access Features
        if 'K022-Proximity to nearest mandi (Km)' in df.columns:
            df['market_accessibility'] = 1 / (df['K022-Proximity to nearest mandi (Km)'] + 1)
        
        if 'K022-Proximity to nearest railway (Km)' in df.columns:
            df['railway_accessibility'] = 1 / (df['K022-Proximity to nearest railway (Km)'] + 1)
        
        # 9. Infrastructure Features
        if 'Road density (Km/ SqKm)' in df.columns:
            df['infrastructure_index'] = df['Road density (Km/ SqKm)'] * df.get('Night light index', 1)
        
        # 10. Credit and Financial Features
        if 'No_of_Active_Loan_In_Bureau' in df.columns and 'Avg_Disbursement_Amount_Bureau' in df.columns:
            df['total_credit_exposure'] = df['No_of_Active_Loan_In_Bureau'] * df['Avg_Disbursement_Amount_Bureau']
            df['credit_per_land'] = df['Avg_Disbursement_Amount_Bureau'] / (df.get('Total_Land_For_Agriculture', 1) + 1)
        
        # 11. Housing Quality Index
        housing_features = [
            'Perc_of_house_with_6plus_room',
            'perc_Households_with_Pucca_House_That_Has_More_Than_3_Rooms',
            'Households_with_improved_Sanitation_Facility'
        ]
        available_housing = [f for f in housing_features if f in df.columns]
        if available_housing:
            df['housing_quality_index'] = df[available_housing].mean(axis=1)
        
        # 12. Agricultural Performance Trend
        perf_cols = [col for col in df.columns if 'Agricultural performance' in col]
        if len(perf_cols) >= 2:
            # Create trend from oldest to newest
            df['agri_performance_trend'] = df[perf_cols].iloc[:, -1].astype(float) - df[perf_cols].iloc[:, 0].astype(float)
        
        # 13. Season-specific features aggregation
        kharif_cols = [col for col in df.columns if 'Kharif' in col and 'Score' in col]
        rabi_cols = [col for col in df.columns if 'Rabi' in col and 'Score' in col]
        
        if kharif_cols:
            df['kharif_avg_score'] = df[kharif_cols].mean(axis=1)
        if rabi_cols:
            df['rabi_avg_score'] = df[rabi_cols].mean(axis=1)
        
        # 14. Interaction Features
        if 'Total_Land_For_Agriculture' in df.columns and 'avg_rainfall' in df.columns:
            df['land_rainfall_interaction'] = df['Total_Land_For_Agriculture'] * df['avg_rainfall']
        
        if 'avg_agri_score' in df.columns and 'Total_Land_For_Agriculture' in df.columns:
            df['productivity_potential'] = df['avg_agri_score'] * df['Total_Land_For_Agriculture']
        
        return df
    
    def preprocess_data(self, df, is_train=True):
        """Preprocess and encode categorical variables"""
        print("Preprocessing data...")
        df = df.copy()
        
        # Identify categorical and numerical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove ID and target if present
        if 'FarmerID' in categorical_cols:
            categorical_cols.remove('FarmerID')
        if 'Target_Variable/Total Income' in categorical_cols:
            categorical_cols.remove('Target_Variable/Total Income')
        
        # Store for later use
        if is_train:
            self.categorical_features = categorical_cols
        
        # Label encode categorical variables
        for col in categorical_cols:
            if col in df.columns:
                if is_train:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        df[col] = df[col].astype(str)
                        df[col] = df[col].apply(lambda x: x if x in self.label_encoders[col].classes_ else 'Unknown')
                        
                        # Add 'Unknown' to classes if not present
                        if 'Unknown' not in self.label_encoders[col].classes_:
                            self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'Unknown')
                        
                        df[col] = self.label_encoders[col].transform(df[col])
        
        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        return df
    
    def select_features(self, X_train, y_train, X_test):
        """Feature selection based on importance"""
        print("Selecting features...")
        
        # Quick feature importance using RandomForest
        rf_selector = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        rf_selector.fit(X_train, y_train)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance
        
        # Select top features (you can adjust this threshold)
        top_n = min(100, len(feature_importance))  # Use top 100 features or all if less
        selected_features = feature_importance.head(top_n)['feature'].tolist()
        
        print(f"Selected {len(selected_features)} features")
        
        return X_train[selected_features], X_test[selected_features], selected_features
    
    def train_models(self, X_train, y_train):
        """Train multiple models and create ensemble"""
        print("Training models...")
        
        # 1. LightGBM
        print("Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        self.models['lightgbm'] = lgb_model
        
        # 2. XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        
        # 4. Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # 5. Extra Trees
        print("Training Extra Trees...")
        et_model = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        et_model.fit(X_train, y_train)
        self.models['extra_trees'] = et_model
        
        # 6. Gradient Boosting
        print("Training Gradient Boosting...")
        gb_model = HistGradientBoostingRegressor(
            max_iter=1000,
            learning_rate=0.05,
            max_depth=8,
            random_state=self.random_state
        )
        gb_model.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb_model
        
        print(f"Trained {len(self.models)} models successfully!")
        
        return self
    
    def predict_ensemble(self, X):
        """Make predictions using weighted ensemble"""
        predictions = []
        
        # Weights for each model (based on typical performance)
        weights = {
            'lightgbm': 0.25,
            'xgboost': 0.25,
            'catboost': 0.25,
            'random_forest': 0.10,
            'extra_trees': 0.10,
            'gradient_boosting': 0.05
        }
        
        for model_name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred * weights.get(model_name, 0.1))
        
        # Ensemble prediction
        ensemble_pred = np.sum(predictions, axis=0)
        
        return ensemble_pred
    
    def evaluate(self, y_true, y_pred):
        """Evaluate model performance"""
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print(f"\nModel Performance:")
        print(f"MAPE: {mape:.2f}%")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        return {'MAPE': mape, 'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    def save_model(self, filepath='farmer_income_model.pkl'):
        """Save the trained model"""
        model_data = {
            'models': self.models,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'selected_features': self.selected_features if hasattr(self, 'selected_features') else None,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='farmer_income_model.pkl'):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.selected_features = model_data.get('selected_features')
        self.feature_importance = model_data.get('feature_importance')
        
        print(f"Model loaded from {filepath}")
        return self