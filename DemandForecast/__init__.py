# __init__.py - Neural Network Demand Forecasting
# This is the main Azure Function that handles forecasting requests

import logging
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime, timedelta
import azure.functions as func

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

class DemandForecaster:
    """
    This class handles the neural network forecasting logic
    It's like a smart brain that learns from your sales data
    """
    
    def __init__(self):
        self.model = None                # Will store the trained model
        self.scaler = None              # Will store the data scaler
        self.label_encoders = {}        # Will store text-to-number converters
        self.feature_columns = []       # Will store which data columns to use
        
    def prepare_data(self, df):
        """
        Clean and prepare Excel data for machine learning
        Converts dates, text, and creates useful features
        """
        logging.info("Preparing data for machine learning...")
        
        # Make a copy so we don't change the original data
        data = df.copy()
        
        # Make sure dates are in the right format
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)
        
        # Convert text columns to numbers (ML models need numbers)
        text_columns = ['Product_ID', 'Weather', 'Day_of_Week']
        
        for col in text_columns:
            if col in data.columns:
                # Create a converter for this column
                converter = LabelEncoder()
                data[col + '_encoded'] = converter.fit_transform(data[col].astype(str))
                # Save converter for later use
                self.label_encoders[col] = converter
        
        # Create time-based features (help model understand patterns)
        data['Month'] = data['Date'].dt.month
        data['Quarter'] = data['Date'].dt.quarter
        data['DayOfYear'] = data['Date'].dt.dayofyear
        data['WeekOfYear'] = data['Date'].dt.isocalendar().week
        
        # Create lag features (yesterday's sales affect today's)
        for days_back in [1, 7, 30]:  # 1 day, 1 week, 1 month ago
            data[f'Demand_lag_{days_back}'] = data['Demand'].shift(days_back)
        
        # Create moving averages (smooth out random fluctuations)
        for window_size in [7, 30]:  # 7-day and 30-day averages
            data[f'Demand_avg_{window_size}'] = data['Demand'].rolling(window=window_size).mean()
        
        # Remove rows with missing values (created by lag features)
        data = data.dropna()
        
        # Define which columns the model will use as inputs
        self.feature_columns = [
            'Product_ID_encoded', 'Price', 'Promotion', 'Weather_encoded',
            'Day_of_Week_encoded', 'Month', 'Quarter', 'DayOfYear', 'WeekOfYear',
            'Demand_lag_1', 'Demand_lag_7', 'Demand_lag_30',
            'Demand_avg_7', 'Demand_avg_30'
        ]
        
        # Only keep columns that actually exist in our data
        self.feature_columns = [col for col in self.feature_columns if col in data.columns]
        
        logging.info(f"Data prepared. Using {len(self.feature_columns)} features: {self.feature_columns}")
        return data
    
    def train_model(self, data):
        """
        Train the forecasting model using your historical data
        This teaches the AI to recognize patterns in your sales
        """
        logging.info("Training forecasting model...")
        
        # Prepare the data for training
        X = data[self.feature_columns].values  # Input features
        y = data['Demand'].values              # What we want to predict
        
        # Split data into training and testing portions
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the data (makes training more stable)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train the model
        self.model = RandomForestRegressor(
            n_estimators=100,    # Use 100 decision trees
            max_depth=10,        # Limit tree depth to prevent overfitting
            random_state=42,     # For consistent results
            n_jobs=-1           # Use all available CPU cores
        )
        
        # Actually train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Test how well it learned
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logging.info(f"Model training complete!")
        logging.info(f"Training accuracy: {train_score:.3f}")
        logging.info(f"Testing accuracy: {test_score:.3f}")
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'features_used': len(self.feature_columns),
            'training_samples': len(X_train)
        }
    
    def predict(self, input_data, days_ahead=30):
        """
        Make demand forecasts using the trained model
        Returns predictions for each product
        """
        if self.model is None:
            raise ValueError("Model hasn't been trained yet! Please train first.")
        
        logging.info(f"Making predictions for {days_ahead} days ahead...")
        
        # Prepare the input data the same way as training data
        prepared_data = self.prepare_data(input_data)
        predictions = []
        
        # Make predictions for each product separately
        for product_id in prepared_data['Product_ID_encoded'].unique():
            # Get data for this specific product
            product_data = prepared_data[
                prepared_data['Product_ID_encoded'] == product_id
            ].tail(1)  # Use the most recent data point
            
            if len(product_data) == 0:
                logging.warning(f"No data found for product ID {product_id}")
                continue
            
            # Extract features for prediction
            features = product_data[self.feature_columns].values
            
            # Scale features the same way as training
            features_scaled = self.scaler.transform(features)
            
            # Make the prediction
            predicted_demand = self.model.predict(features_scaled)[0]
            
            # Get the original product name
            original_product_id = product_data['Product_ID'].iloc[0]
            
            # Determine confidence level based on data quality
            data_points = len(prepared_data[prepared_data['Product_ID_encoded'] == product_id])
            if data_points >= 90:
                confidence = 'High'
            elif data_points >= 30:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            predictions.append({
                'Product_ID': original_product_id,
                'Predicted_Demand': round(float(predicted_demand), 2),
                'Confidence': confidence,
                'Historical_Data_Points': data_points
            })
        
        logging.info(f"Generated predictions for {len(predictions)} products")
        return predictions

# Global variable to store the trained model (persists between API calls)
global_forecaster = None

def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Main Azure Function entry point - this handles all API requests
    Excel and Power BI will call this function
    """
    global global_forecaster
    
    logging.info('Demand forecasting API called')
    
    try:
        # Parse the incoming request
        req_body = req.get_json()
        
        if not req_body:
            return func.HttpResponse(
                json.dumps({"error": "Request body is required"}),
                status_code=400,
                headers={"Content-Type": "application/json"}
            )
        
        action = req_body.get('action', '')
        
        # Handle different types of requests
        if action == 'train':
            # TRAINING MODE: Train the model with historical data
            if 'data' not in req_body:
                return func.HttpResponse(
                    json.dumps({"error": "Training data is required"}),
                    status_code=400,
                    headers={"Content-Type": "application/json"}
                )
            
            # Convert request data to DataFrame
            try:
                df = pd.DataFrame(req_body['data'])
                logging.info(f"Received {len(df)} rows of training data")
            except Exception as e:
                return func.HttpResponse(
                    json.dumps({"error": f"Invalid data format: {str(e)}"}),
                    status_code=400,
                    headers={"Content-Type": "application/json"}
                )
            
            # Validate minimum data requirements
            if len(df) < 30:
                return func.HttpResponse(
                    json.dumps({"error": "Need at least 30 rows of historical data for training"}),
                    status_code=400,
                    headers={"Content-Type": "application/json"}
                )
            
            # Initialize and train the forecaster
            global_forecaster = DemandForecaster()
            prepared_data = global_forecaster.prepare_data(df)
            
            if len(prepared_data) < 10:
                return func.HttpResponse(
                    json.dumps({"error": "After data preparation, need at least 10 valid rows"}),
                    status_code=400,
                    headers={"Content-Type": "application/json"}
                )
            
            # Train the model
            training_results = global_forecaster.train_model(prepared_data)
            
            return func.HttpResponse(
                json.dumps({
                    "status": "success",
                    "message": "Model trained successfully!",
                    "training_accuracy": round(training_results['test_score'], 3),
                    "features_used": training_results['features_used'],
                    "training_samples": training_results['training_samples'],
                    "trained_at": datetime.now().isoformat()
                }),
                status_code=200,
                headers={"Content-Type": "application/json"}
            )
        
        elif action == 'predict':
            # PREDICTION MODE: Generate forecasts
            if global_forecaster is None:
                return func.HttpResponse(
                    json.dumps({"error": "Model not trained yet. Please train the model first."}),
                    status_code=400,
                    headers={"Content-Type": "application/json"}
                )
            
            if 'data' not in req_body:
                return func.HttpResponse(
                    json.dumps({"error": "Input data is required for predictions"}),
                    status_code=400,
                    headers={"Content-Type": "application/json"}
                )
            
            # Get prediction parameters
            days_ahead = req_body.get('days_ahead', 30)
            
            # Convert request data to DataFrame
            try:
                input_df = pd.DataFrame(req_body['data'])
                logging.info(f"Received {len(input_df)} rows for prediction")
            except Exception as e:
                return func.HttpResponse(
                    json.dumps({"error": f"Invalid input data format: {str(e)}"}),
                    status_code=400,
                    headers={"Content-Type": "application/json"}
                )
            
            # Generate predictions
            predictions = global_forecaster.predict(input_df, days_ahead)
            
            return func.HttpResponse(
                json.dumps({
                    "status": "success",
                    "predictions": predictions,
                    "forecast_horizon_days": days_ahead,
                    "generated_at": datetime.now().isoformat(),
                    "total_products": len(predictions)
                }),
                status_code=200,
                headers={"Content-Type": "application/json"}
            )
        
        elif action == 'status':
            # STATUS CHECK: See if model is ready
            model_status = {
                "model_trained": global_forecaster is not None,
                "timestamp": datetime.now().isoformat(),
                "api_version": "1.0"
            }
            
            if global_forecaster is not None:
                model_status["features_available"] = global_forecaster.feature_columns
                model_status["model_ready"] = True
            else:
                model_status["model_ready"] = False
                model_status["message"] = "Please train the model first"
            
            return func.HttpResponse(
                json.dumps(model_status),
                status_code=200,
                headers={"Content-Type": "application/json"}
            )
        
        else:
            return func.HttpResponse(
                json.dumps({
                    "error": "Invalid action",
                    "valid_actions": ["train", "predict", "status"],
                    "example": {"action": "status"}
                }),
                status_code=400,
                headers={"Content-Type": "application/json"}
            )
    
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": "Internal server error",
                "details": str(e)
            }),
            status_code=500,
            headers={"Content-Type": "application/json"}
        )
