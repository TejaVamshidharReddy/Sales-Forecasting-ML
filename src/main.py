#!/usr/bin/env python3
"""
Sales Forecasting ML - Main Prediction Script
This script loads historical sales data, trains a machine learning model,
and generates future sales predictions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath='data/sample_sales_data.csv'):
    """
    Load sales data from CSV file
    """
    print("Loading sales data...")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df

def feature_engineering(df):
    """
    Create features from raw data
    """
    print("Engineering features...")
    df = df.copy()
    
    # Time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Lag features
    df = df.sort_values('date')
    df['revenue_lag_1'] = df.groupby('product_id')['revenue'].shift(1)
    df['revenue_lag_7'] = df.groupby('product_id')['revenue'].shift(7)
    
    # Rolling statistics
    df['revenue_rolling_mean_7'] = df.groupby('product_id')['revenue'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    return df

def train_model(df):
    """
    Train the sales prediction model
    """
    print("Training model...")
    
    # Prepare features and target
    feature_cols = ['day_of_week', 'day_of_month', 'month', 'week_of_year',
                    'quantity', 'price', 'revenue_lag_1', 'revenue_lag_7',
                    'revenue_rolling_mean_7']
    
    # Remove rows with NaN values
    df_clean = df.dropna()
    
    X = df_clean[feature_cols]
    y = df_clean['revenue']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, r2, mae, rmse

def predict_future_sales(model, df, days=7):
    """
    Predict sales for the next N days
    """
    print(f"Generating predictions for next {days} days...")
    
    # Get the last date in the dataset
    last_date = df['date'].max()
    
    # Create future dates
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    
    predictions = []
    
    for future_date in future_dates:
        # Create features for prediction
        # (In a real scenario, you'd use more sophisticated methods)
        last_revenue = df['revenue'].tail(1).values[0]
        avg_quantity = df['quantity'].mean()
        avg_price = df['price'].mean()
        
        features = {
            'day_of_week': future_date.dayofweek,
            'day_of_month': future_date.day,
            'month': future_date.month,
            'week_of_year': future_date.isocalendar()[1],
            'quantity': avg_quantity,
            'price': avg_price,
            'revenue_lag_1': last_revenue,
            'revenue_lag_7': df['revenue'].tail(7).mean(),
            'revenue_rolling_mean_7': df['revenue'].tail(7).mean()
        }
        
        # Make prediction
        X_pred = pd.DataFrame([features])
        pred_revenue = model.predict(X_pred)[0]
        
        predictions.append({
            'date': future_date,
            'predicted_revenue': pred_revenue
        })
    
    return pd.DataFrame(predictions)

def main():
    """
    Main execution function
    """
    print("=" * 50)
    print("Sales Forecasting ML - Prediction System")
    print("=" * 50)
    print()
    
    # Load and prepare data
    df = load_data()
    df = feature_engineering(df)
    
    # Train model
    model, r2, mae, rmse = train_model(df)
    
    # Display model performance
    print("\nModel Performance:")
    print(f"R² Score: {r2:.2f}")
    print(f"Mean Absolute Error: ${mae:.2f}")
    print(f"Root Mean Squared Error: ${rmse:.2f}")
    
    # Generate predictions
    predictions = predict_future_sales(model, df, days=7)
    
    # Display predictions
    print("\nSales Forecast for Next 7 Days:")
    print("-" * 40)
    for idx, row in predictions.iterrows():
        print(f"Day {idx + 1} ({row['date'].strftime('%Y-%m-%d')}): ${row['predicted_revenue']:.2f}")
    
    print("\n" + "=" * 50)
    print("Prediction complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
