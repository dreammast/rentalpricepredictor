import sys
from pathlib import Path

# Add the directory containing 'src' to sys.path
sys.path.append(str(Path(__file__).parent.resolve()))

import logging
import yaml
from flask import Flask, jsonify

# Add the src directory to the Python path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

from src.main import main

app = Flask(__name__)

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

@app.route('/')
def home():
    return jsonify({"status": "running", "message": "Rental Trend Predictor API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger = setup_logging()
        config = load_config()
        logger.info("Starting prediction process...")
        main()
        return jsonify({"status": "success", "message": "Prediction completed successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

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
