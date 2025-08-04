#!/usr/bin/env python3
"""
Interactive House Price Predictor (INR)
Simple script for user input and house price prediction in Indian Rupees.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import time

def train_model():
    """Train the house price prediction model."""
    print("🏠 Training House Price Prediction Model...")
    print("=" * 50)
    
    # Load and preprocess data
    df = pd.read_csv('house_data.csv')
    
    # Convert USD to INR (1 USD ≈ 83 INR)
    df['price'] = df['price'] * 83
    
    # Drop missing values
    df = df.dropna()
    
    # Prepare features and target
    X = df[['square_footage', 'bedrooms', 'bathrooms']].values
    y = df['price'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    r2 = model.score(X_test_scaled, y_test)
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Model accuracy: {r2:.1%}")
    print(f"📈 Training completed with {len(X_train)} samples")
    
    return model, scaler, r2

def get_user_input():
    """Get house details from user."""
    print("\n" + "=" * 50)
    print("🏠 ENTER HOUSE DETAILS")
    print("=" * 50)
    
    while True:
        try:
            square_footage = float(input("📏 Square footage (sq ft): "))
            if square_footage <= 0:
                print("❌ Square footage must be positive!")
                continue
                
            bedrooms = float(input("🛏️  Number of bedrooms: "))
            if bedrooms <= 0:
                print("❌ Number of bedrooms must be positive!")
                continue
                
            bathrooms = float(input("🚿 Number of bathrooms: "))
            if bathrooms <= 0:
                print("❌ Number of bathrooms must be positive!")
                continue
                
            return square_footage, bedrooms, bathrooms
            
        except ValueError:
            print("❌ Please enter valid numbers!")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            exit()

def predict_price(model, scaler, square_footage, bedrooms, bathrooms):
    """Predict house price in INR."""
    # Create feature array
    features = np.array([[square_footage, bedrooms, bathrooms]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    predicted_price = model.predict(features_scaled)[0]
    
    return predicted_price

def format_price_inr(price):
    """Format price in Indian context (lakhs and crores)."""
    if price >= 10000000:  # 1 crore
        crores = price / 10000000
        return f"₹{price:,.2f} (₹{crores:.2f} crores)"
    elif price >= 100000:  # 1 lakh
        lakhs = price / 100000
        return f"₹{price:,.2f} (₹{lakhs:.2f} lakhs)"
    else:
        return f"₹{price:,.2f}"

def main():
    """Main interactive function."""
    print("🏠 HOUSE PRICE PREDICTOR (INR)")
    print("=" * 60)
    print("🇮🇳 All prices in Indian Rupees (INR)")
    print("=" * 60)
    
    # Train model
    model, scaler, accuracy = train_model()
    
    print(f"\n🎯 Ready for predictions! Model accuracy: {accuracy:.1%}")
    
    while True:
        try:
            # Get user input
            square_footage, bedrooms, bathrooms = get_user_input()
            
            # Make prediction
            predicted_price = predict_price(model, scaler, square_footage, bedrooms, bathrooms)
            
            # Display results
            print("\n" + "=" * 50)
            print("🏠 PREDICTION RESULT")
            print("=" * 50)
            print(f"📏 House Details:")
            print(f"   Square Footage: {square_footage:,.0f} sq ft")
            print(f"   Bedrooms: {bedrooms}")
            print(f"   Bathrooms: {bathrooms}")
            print(f"\n💰 Predicted Price: {format_price_inr(predicted_price)}")
            print(f"\n⚡ Model Performance:")
            print(f"   Accuracy: {accuracy:.1%}")
            
            # Ask if user wants to predict another house
            print("\n" + "-" * 50)
            another = input("🔍 Predict another house? (y/n): ").lower().strip()
            if another not in ['y', 'yes', 'Y', 'YES']:
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
    
    print("\n" + "=" * 60)
    print("🎉 Thank you for using the House Price Predictor!")
    print("🇮🇳 All predictions in Indian Rupees (INR)")
    print("=" * 60)

if __name__ == "__main__":
    main() 