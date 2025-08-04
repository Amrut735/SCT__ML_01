#!/usr/bin/env python3
"""
Generate sample house data for testing the house price predictor.
Creates realistic house data with square footage, bedrooms, bathrooms, and prices.
"""

import pandas as pd
import numpy as np

def generate_house_data(n_samples=1000):
    """
    Generate realistic house data for testing.
    
    Args:
        n_samples (int): Number of houses to generate
        
    Returns:
        pd.DataFrame: DataFrame with house data
    """
    np.random.seed(42)
    
    # Generate realistic house features
    square_footage = np.random.normal(2000, 800, n_samples)
    square_footage = np.clip(square_footage, 800, 5000)  # Realistic range
    
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.25, 0.35, 0.25, 0.05])
    
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5], n_samples, p=[0.2, 0.25, 0.3, 0.15, 0.08, 0.02])
    
    # Create realistic price based on features with some noise
    base_price = 150000  # Base price
    price_per_sqft = 150  # Price per square foot
    bedroom_bonus = 25000  # Bonus per bedroom
    bathroom_bonus = 15000  # Bonus per bathroom
    
    # Calculate price with realistic relationships
    price = (base_price + 
             square_footage * price_per_sqft + 
             bedrooms * bedroom_bonus + 
             bathrooms * bathroom_bonus)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 50000, n_samples)
    price += noise
    
    # Ensure prices are positive
    price = np.maximum(price, 100000)
    
    # Add some missing values for testing
    missing_mask = np.random.random(n_samples) < 0.05  # 5% missing values
    square_footage = square_footage.astype(float)
    square_footage[missing_mask] = np.nan
    
    missing_mask = np.random.random(n_samples) < 0.03  # 3% missing values
    bedrooms = bedrooms.astype(float)
    bedrooms[missing_mask] = np.nan
    
    missing_mask = np.random.random(n_samples) < 0.03  # 3% missing values
    bathrooms = bathrooms.astype(float)
    bathrooms[missing_mask] = np.nan
    
    missing_mask = np.random.random(n_samples) < 0.02  # 2% missing values
    price[missing_mask] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'square_footage': square_footage,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'price': price
    })
    
    return df

def main():
    """Generate and save sample house data."""
    print("Generating sample house data...")
    
    # Generate data
    df = generate_house_data(n_samples=1000)
    
    # Save to CSV
    df.to_csv('house_data.csv', index=False)
    
    print(f"Generated {len(df)} house records")
    print("Data saved to 'house_data.csv'")
    
    # Display sample statistics
    print("\nSample data statistics:")
    print(df.describe())
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nSample data (first 5 rows):")
    print(df.head())

if __name__ == "__main__":
    main() 