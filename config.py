import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Model and data paths
MODEL_PATH = BASE_DIR / 'output' / 'ensemble_model.joblib'
DATA_PATH = BASE_DIR / 'data' / 'House_Rent_Dataset.csv'

# Flask settings
class Config:
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Application settings
    DEBUG = False
    TESTING = False
    
    # Database settings (if needed in future)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///rental_predictor.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Cache settings (if needed in future)
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300

class ProductionConfig(Config):
    # Production-specific settings
    DEBUG = False
    TESTING = False
    
    # Security settings for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'
    
    # Logging configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

class DevelopmentConfig(Config):
    # Development-specific settings
    DEBUG = True
    TESTING = False
    
    # Security settings for development
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Logging configuration
    LOG_LEVEL = 'DEBUG'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

class TestingConfig(Config):
    # Testing-specific settings
    DEBUG = False
    TESTING = True
    
    # Security settings for testing
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Use in-memory database for testing
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

# Choose the appropriate configuration
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 