import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

class RentalPlotter:
    def __init__(self):
        """Initialize the RentalPlotter with styling and logging."""
        self.logger = self._setup_logger()
        self._set_style()
    
    def _setup_logger(self):
        """Set up logging configuration."""
        logger = logging.getLogger('RentalPlotter')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _set_style(self):
        """Set the style for all plots."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_predictions(self, actual: np.ndarray, predicted: np.ndarray, 
                        title: str = "Actual vs Predicted Rental Prices",
                        save_path: Optional[str] = None):
        """
        Plot actual vs predicted values with enhanced styling.
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        plt.scatter(actual, predicted, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Add labels and title
        plt.xlabel("Actual Rental Prices (₹)")
        plt.ylabel("Predicted Rental Prices (₹)")
        plt.title(title)
        
        # Add correlation coefficient
        correlation = np.corrcoef(actual, predicted)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', 
                transform=plt.gca().transAxes)
        
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        return plt

    def plot_rental_trends(self, df: pd.DataFrame, cities: List[str],
                          save_path: Optional[str] = None):
        """
        Plot rental trends for different cities with enhanced styling.
        
        Args:
            df: DataFrame containing rental data
            cities: List of cities to plot
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(12, 6))
        
        for city in cities:
            city_data = df[df['City'] == city]
            monthly_avg = city_data.groupby(pd.Grouper(key='Posted On', freq='M'))['Rent'].mean()
            plt.plot(monthly_avg.index, monthly_avg.values, label=city, marker='o')
        
        plt.xlabel("Date")
        plt.ylabel("Average Rental Price (₹)")
        plt.title("Monthly Average Rental Price Trends by City")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        return plt

    def plot_feature_importance(self, model, feature_names: List[str],
                              save_path: Optional[str] = None):
        """
        Plot feature importance for models that support it.
        
        Args:
            model: Trained model instance
            feature_names: List of feature names
            save_path: Path to save the plot (optional)
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            self.logger.warning("Model doesn't support feature importance visualization")
            return None
        
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance in Rental Price Prediction')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        return plt 