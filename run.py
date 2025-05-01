import logging
import sys
from pathlib import Path
import yaml

# Add the src directory to the Python path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

from src.main import main

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    logger = setup_logging()
    config = load_config()
    
    try:
        logger.info("Starting Rental Trend Predictor...")
        main()
        logger.info("Process completed successfully!")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1) 