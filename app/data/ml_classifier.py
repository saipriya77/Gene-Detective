# app/data/ml_classifier.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MLClassifier:
    """Class for machine learning-based classification of gene expression data."""
    
    def __init__(self):
        """Initialize the ML classifier."""
        self.model = None
        self.feature_importances = None
    
    def train_xgboost(self, expression_df, labels, n_features=None):
        """Train an XGBoost classifier on gene expression data."""
        try:
            # Prepare data
            X = expression_df.values
            y = labels
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Train model
            model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Get feature importances
            feature_importances = pd.DataFrame({
                "gene_id": expression_df.columns,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            
            # Store model and feature importances
            self.model = model
            self.feature_importances = feature_importances
            
            return feature_importances
            
        except Exception as e:
            raise Exception(f"XGBoost training failed: {str(e)}")
