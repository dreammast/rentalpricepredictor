url of application:https://f100f12a-cac0-4dc7-9a02-253a1a94597a-00-1y2ph3v5oer4q.picard.replit.dev/
# Rental Trend Predictor

## Overview
This project implements a machine learning solution to predict rental prices using historical data. The system employs various regression techniques to forecast rental prices, providing valuable insights for real estate market analysis.

## Features
- Historical rental price analysis
- Multiple model options (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting)
- Advanced data preprocessing and feature engineering
- Model performance evaluation metrics
- Interactive visualizations of rental trends
- Rental price forecasting

## Technical Stack
- Python 3.8+
- scikit-learn for machine learning
- pandas for data manipulation
- matplotlib/seaborn for visualization
- pytest for testing

## Project Structure
```
rental_trend_predictor/
├── src/
│   ├── data/
│   │   └── data_processor.py
│   ├── models/
│   │   └── predictor.py
│   ├── visualization/
│   │   └── plotter.py
│   └── main.py
├── tests/
├── output/
├── config.yaml
└── requirements.txt
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/rental-trend-predictor.git
cd rental-trend-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your rental data in the `data` directory
2. Run the main script:
```bash
python src/main.py
```

The script will:
- Process the data
- Train the model
- Generate visualizations
- Save the trained model
- Display evaluation metrics

## Model Evaluation
The model's performance is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score

## Data Format
The input data should be a CSV file with the following columns:
- Posted On: Date of posting
- BHK: Number of bedrooms, hall, kitchen
- Rent: Rental price
- Size: Size in square feet
- Floor: Floor information
- Area Type: Type of area measurement
- Area Locality: Locality name
- City: City name
- Furnishing Status: Furnishing status
- Tenant Preferred: Preferred tenant type
- Bathroom: Number of bathrooms
- Point of Contact: Contact information

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
Your Name - naledushyanth@gmail.com
Project Link: https://github.com/dreammast/rental-trend-predictor 
