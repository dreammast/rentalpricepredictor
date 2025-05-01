import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import logging

class RentalDataProcessor:
    def __init__(self, data_path: str):
        """
        Initialize the RentalDataProcessor.
        
        Args:
            data_path (str): Path to the CSV data file
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.logger = self._setup_logger()
        self.feature_columns = None  # Initialize feature_columns attribute
        
    def _setup_logger(self):
        """Set up logging configuration."""
        logger = logging.getLogger('RentalDataProcessor')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_data(self) -> pd.DataFrame:
        """
        Load and perform initial cleaning of the data.
        
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        try:
            df = pd.read_csv(self.data_path)
            self.logger.info(f"Successfully loaded data with shape: {df.shape}")
            
            # Convert date column
            df['Posted On'] = pd.to_datetime(df['Posted On'])
            
            # Clean rent values (remove any non-numeric values and convert to float)
            df['Rent'] = pd.to_numeric(df['Rent'], errors='coerce')
            
            # Clean size values
            df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
            
            # Fill missing values
            numeric_columns = ['Rent', 'Size', 'BHK']
            for col in numeric_columns:
                df[col] = df[col].fillna(df[col].median())
            
            # Fill categorical missing values with mode
            categorical_columns = ['Area Type', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']
            for col in categorical_columns:
                df[col] = df[col].fillna(df[col].mode()[0])
            
            self.logger.info(f"Data cleaning completed. New shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for the model.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        # Temporal features
        df['Year'] = df['Posted On'].dt.year
        df['Month'] = df['Posted On'].dt.month
        df['Quarter'] = df['Posted On'].dt.quarter
        df['DayOfWeek'] = df['Posted On'].dt.dayofweek
        
        # Extract floor information
        df['Floor_Num'] = df['Floor'].apply(lambda x: str(x).split(' out')[0].strip())
        df['Total_Floors'] = df['Floor'].apply(lambda x: str(x).split('out of ')[-1] if 'out of' in str(x) else '0')
        
        # Convert floor information to numeric
        df['Floor_Num'] = pd.to_numeric(df['Floor_Num'].replace('Ground', '0'), errors='coerce')
        df['Total_Floors'] = pd.to_numeric(df['Total_Floors'], errors='coerce')
        
        # Fill missing floor information with medians
        df['Floor_Num'] = df['Floor_Num'].fillna(df['Floor_Num'].median())
        df['Total_Floors'] = df['Total_Floors'].fillna(df['Total_Floors'].median())
        
        # Create derived features - only if Rent column exists
        if 'Rent' in df.columns:
            df['Price_Per_Sqft'] = df['Rent'] / df['Size']
            df['Room_Price'] = df['Rent'] / df['BHK']
            
            # Handle infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            df['Price_Per_Sqft'] = df['Price_Per_Sqft'].fillna(df['Price_Per_Sqft'].median())
            df['Room_Price'] = df['Room_Price'].fillna(df['Room_Price'].median())
            
            # Create city-wise average prices
            city_avg_price = df.groupby('City')['Price_Per_Sqft'].transform('mean')
            df['Price_Ratio_To_City_Avg'] = df['Price_Per_Sqft'] / city_avg_price
        else:
            # For prediction, use default values
            df['Price_Per_Sqft'] = df['Size'].median()  # Use median size as default
            df['Room_Price'] = df['BHK'].median()  # Use median BHK as default
            df['Price_Ratio_To_City_Avg'] = 1.0  # Default ratio
        
        # Encode categorical variables
        categorical_columns = ['Area Type', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[f'{col}_Encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        self.logger.info("Feature engineering completed successfully")
        return df

    def prepare_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Prepare data for modeling.
        
        Args:
            test_size (float): Proportion of dataset to include in the test split
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Load and process data
        df = self.load_data()
        df = self.create_features(df)
        
        # Select features for modeling
        self.feature_columns = [
            'BHK', 'Size', 'Floor_Num', 'Total_Floors',
            'Price_Per_Sqft', 'Room_Price', 'Year', 'Month',
            'Quarter', 'DayOfWeek', 'Area Type_Encoded',
            'Furnishing Status_Encoded', 'Tenant Preferred_Encoded',
            'Point of Contact_Encoded', 'Price_Ratio_To_City_Avg'
        ]
        
        X = df[self.feature_columns]
        y = df['Rent']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        self.logger.info(f"Data preparation completed. Training set shape: {X_train.shape}")
        return X_train, X_test, y_train, y_test

    def get_feature_names(self) -> list:
        """Return the list of feature names used in the model."""
        if self.feature_columns is None:
            raise ValueError("Feature columns not set. Call prepare_data() first.")
        return self.feature_columns 