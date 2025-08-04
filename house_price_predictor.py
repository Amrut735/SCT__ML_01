#!/usr/bin/env python3
"""
Optimized House Price Predictor using Linear Regression
Time Complexity: O(n * d) where n = samples, d = features
Modified for Indian Rupees (INR) with user input functionality
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

class OptimizedHousePricePredictor:
    """
    Optimized house price predictor with minimal time complexity.
    Uses vectorized operations and efficient preprocessing.
    Modified for Indian Rupees (INR).
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        self.model = LinearRegression()
        self.feature_names = None
        
    def load_and_preprocess(self, file_path):
        """
        Load and preprocess data with minimal operations.
        Time Complexity: O(n) where n = number of rows
        """
        print("Loading and preprocessing data...")
        start_time = time.time()
        
        # Load data with only required columns for efficiency
        required_columns = ['square_footage', 'bedrooms', 'bathrooms', 'price']
        df = pd.read_csv(file_path, usecols=required_columns)
        
        # Convert prices to Indian Rupees (assuming original data is in USD)
        # 1 USD ≈ 83 INR (approximate conversion rate)
        df['price'] = df['price'] * 83
        
        # Drop missing values (no imputation to reduce runtime)
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        if initial_rows != final_rows:
            print(f"Dropped {initial_rows - final_rows} rows with missing values")
        
        # Separate features and target
        X = df[['square_footage', 'bedrooms', 'bathrooms']].values
        y = df['price'].values
        
        self.feature_names = ['square_footage', 'bedrooms', 'bathrooms']
        
        preprocessing_time = time.time() - start_time
        print(f"Preprocessing completed in {preprocessing_time:.4f} seconds")
        print(f"Dataset shape: {X.shape}")
        
        return X, y
    
    def normalize_features(self, X_train, X_test):
        """
        Normalize features using MinMaxScaler (vectorized operation).
        Time Complexity: O(n * d)
        """
        print("Normalizing features...")
        start_time = time.time()
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        normalization_time = time.time() - start_time
        print(f"Normalization completed in {normalization_time:.4f} seconds")
        
        return X_train_scaled, X_test_scaled
    
    def train_model(self, X_train, y_train):
        """
        Train linear regression model.
        Time Complexity: O(n * d) for training
        """
        print("Training linear regression model...")
        start_time = time.time()
        
        # Train model (sklearn uses efficient algorithms)
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.4f} seconds")
        
        # Print model coefficients
        coefficients = dict(zip(self.feature_names, self.model.coef_))
        print(f"Model coefficients (INR): {coefficients}")
        print(f"Intercept: ₹{self.model.intercept_:,.2f}")
        
        return training_time
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance using vectorized operations.
        Time Complexity: O(n) for predictions and metrics
        """
        print("Evaluating model...")
        start_time = time.time()
        
        # Make predictions (vectorized)
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        evaluation_time = time.time() - start_time
        
        print(f"Model Performance:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Mean Squared Error: ₹{mse:,.2f}")
        print(f"  Root Mean Squared Error: ₹{rmse:,.2f}")
        print(f"Evaluation completed in {evaluation_time:.4f} seconds")
        
        return y_pred, r2, mse, rmse
    
    def plot_results(self, y_test, y_pred, max_points=100):
        """
        Plot actual vs predicted prices (limited to max_points for efficiency).
        Time Complexity: O(max_points)
        """
        print("Creating visualization...")
        start_time = time.time()
        
        # Limit points for efficient plotting
        if len(y_test) > max_points:
            indices = np.random.choice(len(y_test), max_points, replace=False)
            y_test_plot = y_test[indices]
            y_pred_plot = y_pred[indices]
        else:
            y_test_plot = y_test
            y_pred_plot = y_pred
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_plot, y_pred_plot, alpha=0.6, color='blue')
        
        # Add perfect prediction line
        min_val = min(y_test_plot.min(), y_pred_plot.min())
        max_val = max(y_test_plot.max(), y_pred_plot.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Price (₹)')
        plt.ylabel('Predicted Price (₹)')
        plt.title('House Price Prediction: Actual vs Predicted (INR)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² score to plot
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('house_price_prediction_inr.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        plotting_time = time.time() - start_time
        print(f"Visualization completed in {plotting_time:.4f} seconds")
        print("Plot saved as 'house_price_prediction_inr.png'")
    
    def predict_new_house(self, square_footage, bedrooms, bathrooms):
        """
        Predict price for a new house in Indian Rupees.
        Time Complexity: O(1)
        """
        # Create feature array
        features = np.array([[square_footage, bedrooms, bathrooms]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        predicted_price = self.model.predict(features_scaled)[0]
        
        return predicted_price

def get_user_input():
    """
    Get house details from user input.
    """
    print("\n" + "=" * 50)
    print("🏠 HOUSE PRICE PREDICTION - USER INPUT")
    print("=" * 50)
    
    while True:
        try:
            print("\nPlease enter the house details:")
            square_footage = float(input("📏 Square footage: "))
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

def main():
    """
    Main execution function with performance monitoring and user input.
    """
    print("=" * 60)
    print("🏠 OPTIMIZED HOUSE PRICE PREDICTOR (INR)")
    print("=" * 60)
    
    # Initialize predictor
    predictor = OptimizedHousePricePredictor(random_state=42)
    
    # Load and preprocess data
    try:
        X, y = predictor.load_and_preprocess('house_data.csv')
    except FileNotFoundError:
        print("Error: house_data.csv not found!")
        print("Please ensure the CSV file exists with columns: square_footage, bedrooms, bathrooms, price")
        return
    
    # Split data (80/20)
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Normalize features
    X_train_scaled, X_test_scaled = predictor.normalize_features(X_train, X_test)
    
    # Train model
    training_time = predictor.train_model(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred, r2, mse, rmse = predictor.evaluate_model(X_test_scaled, y_test)
    
    # Create visualization
    predictor.plot_results(y_test, y_pred)
    
    # Get user input for prediction
    while True:
        square_footage, bedrooms, bathrooms = get_user_input()
        
        # Make prediction
        predicted_price = predictor.predict_new_house(square_footage, bedrooms, bathrooms)
        
        print("\n" + "=" * 50)
        print("🏠 PREDICTION RESULTS")
        print("=" * 50)
        print(f"📏 House Details:")
        print(f"   Square Footage: {square_footage:,.0f} sq ft")
        print(f"   Bedrooms: {bedrooms}")
        print(f"   Bathrooms: {bathrooms}")
        print(f"\n💰 Predicted Price: ₹{predicted_price:,.2f}")
        
        # Convert to lakhs and crores for Indian context
        if predicted_price >= 10000000:  # 1 crore
            crores = predicted_price / 10000000
            print(f"   (₹{crores:.2f} crores)")
        elif predicted_price >= 100000:  # 1 lakh
            lakhs = predicted_price / 100000
            print(f"   (₹{lakhs:.2f} lakhs)")
        
        # Performance summary
        print(f"\n⚡ Performance Summary:")
        print(f"   Training time: {training_time:.4f} seconds")
        print(f"   R² Score: {r2:.4f} ({r2*100:.1f}% accuracy)")
        print(f"   RMSE: ₹{rmse:,.2f}")
        
        if training_time < 1.0:
            print("   ✅ Performance target achieved: Training completed under 1 second!")
        else:
            print("   ⚠️  Performance target exceeded: Training took over 1 second")
        
        # Ask if user wants to predict another house
        another = input("\n🔍 Predict another house? (y/n): ").lower().strip()
        if another not in ['y', 'yes', 'Y', 'YES']:
            break
    
    print("\n" + "=" * 60)
    print("🎉 Thank you for using the House Price Predictor!")
    print("=" * 60)
    print("Optimization features used:")
    print("- Vectorized operations only (no loops)")
    print("- Minimal preprocessing (drop missing values only)")
    print("- Efficient sklearn algorithms")
    print("- Limited plotting points for visualization")
    print("- Indian Rupees (INR) conversion")

if __name__ == "__main__":
    main() 