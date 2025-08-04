#!/usr/bin/env python3
"""
Test script for the Optimized House Price Predictor
Demonstrates the functionality with a simple example.
"""

import pandas as pd
import numpy as np
from house_price_predictor import OptimizedHousePricePredictor
import time

def test_predictor():
    """Test the house price predictor with sample data."""
    print("=" * 50)
    print("TESTING HOUSE PRICE PREDICTOR")
    print("=" * 50)
    
    # Initialize predictor
    predictor = OptimizedHousePricePredictor(random_state=42)
    
    # Load and preprocess data
    print("1. Loading and preprocessing data...")
    start_time = time.time()
    X, y = predictor.load_and_preprocess('house_data.csv')
    load_time = time.time() - start_time
    print(f"   Data loaded in {load_time:.4f} seconds")
    
    # Split data
    print("2. Splitting data...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Normalize features
    print("3. Normalizing features...")
    X_train_scaled, X_test_scaled = predictor.normalize_features(X_train, X_test)
    
    # Train model
    print("4. Training model...")
    training_time = predictor.train_model(X_train_scaled, y_train)
    
    # Evaluate model
    print("5. Evaluating model...")
    y_pred, r2, mse, rmse = predictor.evaluate_model(X_test_scaled, y_test)
    
    # Test predictions
    print("6. Testing predictions...")
    test_houses = [
        (1500, 2, 1.5),  # Small house
        (2500, 3, 2.0),  # Medium house
        (4000, 4, 3.0),  # Large house
    ]
    
    print("\nPrediction Examples:")
    for sqft, beds, baths in test_houses:
        predicted_price = predictor.predict_new_house(sqft, beds, baths)
        print(f"   {sqft} sqft, {beds} bed, {baths} bath: ${predicted_price:,.2f}")
    
    # Performance summary
    total_time = load_time + training_time
    print(f"\nPerformance Summary:")
    print(f"   Total time: {total_time:.4f} seconds")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: ${rmse:,.2f}")
    
    if total_time < 1.0:
        print("   ✅ Performance target achieved!")
    else:
        print("   ⚠️  Performance target exceeded")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_predictor() 