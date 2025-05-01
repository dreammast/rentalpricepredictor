from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from wtforms import Form, FloatField, IntegerField, SelectField, validators
import logging
from config import config

# Add the parent directory to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data.data_processor import RentalDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # Load the trained model
    model_path = Path(__file__).parent.parent.parent / 'output' / 'ensemble_model.joblib'
    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # Load the data processor for feature engineering
    data_path = Path(__file__).parent.parent.parent / 'data' / 'House_Rent_Dataset.csv'
    logger.info(f"Loading data from: {data_path}")
    processor = RentalDataProcessor(str(data_path))
    _, _, _, _ = processor.prepare_data()  # This sets up the feature engineering pipeline

    class PredictionForm(Form):
        bhk = IntegerField('Number of BHK', [validators.NumberRange(min=1, max=10)])
        size = FloatField('Size (sq ft)', [validators.NumberRange(min=100, max=10000)])
        floor = SelectField('Floor', choices=[
            ('Ground', 'Ground Floor'),
            ('1', '1st Floor'),
            ('2', '2nd Floor'),
            ('3', '3rd Floor'),
            ('4', '4th Floor'),
            ('5', '5th Floor and above')
        ])
        area_type = SelectField('Area Type', choices=[
            ('Super Area', 'Super Area'),
            ('Carpet Area', 'Carpet Area'),
            ('Built Area', 'Built Area')
        ])
        furnishing_status = SelectField('Furnishing Status', choices=[
            ('Unfurnished', 'Unfurnished'),
            ('Semi-Furnished', 'Semi-Furnished'),
            ('Furnished', 'Furnished')
        ])
        tenant_preferred = SelectField('Tenant Preferred', choices=[
            ('Bachelors/Family', 'Bachelors/Family'),
            ('Bachelors', 'Bachelors'),
            ('Family', 'Family')
        ])
        bathroom = IntegerField('Number of Bathrooms', [validators.NumberRange(min=1, max=10)])
        city = SelectField('City', choices=[
            ('Mumbai', 'Mumbai'),
            ('Chennai', 'Chennai'),
            ('Bangalore', 'Bangalore'),
            ('Hyderabad', 'Hyderabad'),
            ('Delhi', 'Delhi'),
            ('Kolkata', 'Kolkata')
        ])

    def prepare_features(form_data):
        """Prepare features for prediction using the same preprocessing as training."""
        # Create a DataFrame with the input data
        df = pd.DataFrame([{
            'BHK': form_data['bhk'],
            'Size': form_data['size'],
            'Floor': form_data['floor'],
            'Area Type': form_data['area_type'],
            'Furnishing Status': form_data['furnishing_status'],
            'Tenant Preferred': form_data['tenant_preferred'],
            'Bathroom': form_data['bathroom'],
            'City': form_data['city'],
            'Posted On': pd.to_datetime(datetime.now().strftime('%Y-%m-%d')),
            'Point of Contact': 'Contact Owner'  # Default value for prediction
        }])
        
        # Apply the same feature engineering as in training
        df = processor.create_features(df)
        
        # Select the same features used in training
        features = processor.get_feature_names()
        X = df[features]
        
        # Scale the features
        X_scaled = processor.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=features)

    @app.route('/', methods=['GET', 'POST'])
    def predict():
        form = PredictionForm(request.form)
        prediction = None
        
        if request.method == 'POST' and form.validate():
            # Prepare features
            features = prepare_features(form.data)
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Format the prediction
            prediction = f"₹{prediction:,.2f}"
        
        return render_template('index.html', form=form, prediction=prediction)

    @app.route('/api/predict', methods=['POST'])
    def api_predict():
        try:
            data = request.get_json()
            features = prepare_features(data)
            prediction = model.predict(features)[0]
            return jsonify({
                'prediction': float(prediction),
                'formatted_prediction': f"₹{prediction:,.2f}"
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    return app

if __name__ == '__main__':
    app = create_app('development')
    app.run(host='0.0.0.0', port=5000, debug=True) 