import pandas as pd
import logging
from pathlib import Path
from data.data_processor import RentalDataProcessor
from models.ensemble import RentalEnsemble
from visualization.plotter import RentalPlotter

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    
    # Initialize paths
    data_path = Path("data/House_Rent_Dataset.csv")
    output_path = Path("output")
    output_path.mkdir(exist_ok=True)
    
    try:
        # Initialize components
        processor = RentalDataProcessor(data_path)
        ensemble = RentalEnsemble(model_path=str(output_path / "ensemble_model.joblib"))
        plotter = RentalPlotter()
        
        # Prepare data
        logger.info("Preparing data...")
        X_train, X_test, y_train, y_test = processor.prepare_data()
        
        # Train ensemble model
        logger.info("Training ensemble model...")
        ensemble.train(X_train, y_train)
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = ensemble.evaluate(X_test, y_test)
        
        # Make predictions
        predictions = ensemble.predict(X_test)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        
        # Plot predictions
        plotter.plot_predictions(
            y_test, predictions,
            title=f"Actual vs Predicted Rental Prices (Ensemble Model: {metrics['r2']:.3f} R²)",
            save_path=output_path / "predictions.png"
        )
        
        # Plot feature importance
        plotter.plot_feature_importance(
            ensemble.model.estimators_[0],  # Plot GBM feature importance
            processor.get_feature_names(),
            save_path=output_path / "feature_importance.png"
        )
        
        logger.info("Process completed successfully!")
        logger.info(f"Model metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 