from sklearn.ensemble import VotingRegressor, GradientBoostingRegressor, RandomForestRegressor
import joblib
import logging
from pathlib import Path

class RentalEnsemble:
    def __init__(self, model_path: str = None):
        """
        Initialize the RentalEnsemble.
        
        Args:
            model_path (str): Path to load/save the model
        """
        self.model_path = model_path
        self.logger = self._setup_logger()
        self.model = self._create_ensemble()
        
    def _setup_logger(self):
        """Set up logging configuration."""
        logger = logging.getLogger('RentalEnsemble')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _create_ensemble(self):
        """Create the ensemble model."""
        gbm = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        return VotingRegressor([
            ('gbm', gbm),
            ('rf', rf)
        ], weights=[0.6, 0.4])  # Give more weight to GBM based on previous results
    
    def train(self, X_train, y_train):
        """Train the ensemble model."""
        self.logger.info("Training ensemble model...")
        self.model.fit(X_train, y_train)
        self.logger.info("Ensemble model training completed")
        
        if self.model_path:
            self.save_model()
    
    def predict(self, X):
        """Make predictions using the ensemble model."""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the ensemble model."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        
        predictions = self.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions)
        }
        
        self.logger.info(f"Ensemble model evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self):
        """Save the ensemble model to disk."""
        if self.model_path:
            joblib.dump(self.model, self.model_path)
            self.logger.info(f"Ensemble model saved to {self.model_path}")
    
    def load_model(self):
        """Load the ensemble model from disk."""
        if self.model_path and Path(self.model_path).exists():
            self.model = joblib.load(self.model_path)
            self.logger.info(f"Ensemble model loaded from {self.model_path}")
        else:
            self.logger.warning("No saved model found, using newly created ensemble") 