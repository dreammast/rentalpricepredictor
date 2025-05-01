from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import logging
from typing import Dict, Any, Tuple

class RentalPredictor:
    def __init__(self, model_type: str = 'linear'):
        """
        Initialize the RentalPredictor.
        
        Args:
            model_type (str): Type of model to use ('linear', 'ridge', 'lasso', 'rf', 'gbm')
        """
        self.model_type = model_type
        self.model = self._get_model()
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Set up logging configuration."""
        logger = logging.getLogger('RentalPredictor')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _get_model(self):
        """Get the specified model instance."""
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        return models.get(self.model_type, LinearRegression())
    
    def train(self, X_train, y_train):
        """
        Train the model on the given data.
        
        Args:
            X_train: Training features
            y_train: Training target values
        """
        self.logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed")
        
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            numpy.ndarray: Predicted values
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate the model's performance.
        
        Args:
            X_test: Test features
            y_test: True target values
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        predictions = self.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions)
        }
        
        self.logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, path: str):
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path to save the model
        """
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load a trained model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from {path}") 