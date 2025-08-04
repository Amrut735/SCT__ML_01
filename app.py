#!/usr/bin/env python3
"""
SmartPrice - House Price Prediction Web Application
A modern web interface for predicting house prices in Indian Rupees (INR)
"""

from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
import json
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = 'smartprice_secret_key_2024'

# Global variables for model
model = None
scaler = None
r2_score_value = None
rmse_value = None

def train_model():
    """Train the house price prediction model."""
    global model, scaler, r2_score_value, rmse_value
    
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
    r2_score_value = model.score(X_test_scaled, y_test)
    rmse_value = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, scaler, r2_score_value, rmse_value

def predict_price(square_footage, bedrooms, bathrooms):
    """Predict house price in INR."""
    if model is None or scaler is None:
        return None
    
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
        return f"₹{lakhs:.2f} lakhs"
    else:
        return f"₹{price:,.2f}"

def create_visualization():
    """Create actual vs predicted visualization."""
    # Load data for visualization
    df = pd.read_csv('house_data.csv')
    df['price'] = df['price'] * 83  # Convert to INR
    df = df.dropna()
    
    X = df[['square_footage', 'bedrooms', 'bathrooms']].values
    y = df['price'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize features
    scaler_viz = MinMaxScaler()
    X_train_scaled = scaler_viz.fit_transform(X_train)
    X_test_scaled = scaler_viz.transform(X_test)
    
    # Train model for visualization
    model_viz = LinearRegression()
    model_viz.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model_viz.predict(X_test_scaled)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='#4A90E2', s=50)
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Price (₹)', fontsize=12)
    plt.ylabel('Predicted Price (₹)', fontsize=12)
    plt.title('House Price Prediction: Actual vs Predicted (INR)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R² score to plot
    r2 = r2_score(y_test, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/')
def index():
    """Main page with prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request."""
    try:
        # Get form data
        square_footage = float(request.form['square_footage'])
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        
        # Validate input
        if square_footage <= 0 or bedrooms <= 0 or bathrooms <= 0:
            return jsonify({
                'success': False,
                'error': 'All values must be positive numbers!'
            })
        
        # Make prediction
        predicted_price = predict_price(square_footage, bedrooms, bathrooms)
        
        if predicted_price is None:
            return jsonify({
                'success': False,
                'error': 'Model not trained. Please try again.'
            })
        
        # Format price
        formatted_price = format_price_inr(predicted_price)
        
        # Create prediction history entry
        prediction_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'square_footage': square_footage,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'predicted_price': predicted_price,
            'formatted_price': formatted_price
        }
        
        # Store in session
        if 'prediction_history' not in session:
            session['prediction_history'] = []
        
        session['prediction_history'].append(prediction_entry)
        
        # Keep only last 10 predictions
        if len(session['prediction_history']) > 10:
            session['prediction_history'] = session['prediction_history'][-10:]
        
        return jsonify({
            'success': True,
            'predicted_price': predicted_price,
            'formatted_price': formatted_price,
            'rmse': rmse_value,
            'r2_score': r2_score_value,
            'prediction_history': session['prediction_history']
        })
        
    except ValueError:
        return jsonify({
            'success': False,
            'error': 'Please enter valid numbers for all fields!'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        })

@app.route('/visualization')
def visualization():
    """Generate and return visualization."""
    try:
        plot_url = create_visualization()
        return jsonify({
            'success': True,
            'plot_url': plot_url
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error generating visualization: {str(e)}'
        })

@app.route('/get_history')
def get_history():
    """Get prediction history."""
    history = session.get('prediction_history', [])
    return jsonify({
        'success': True,
        'history': history
    })

@app.route('/clear_history')
def clear_history():
    """Clear prediction history."""
    session['prediction_history'] = []
    return jsonify({
        'success': True,
        'message': 'History cleared successfully!'
    })

if __name__ == '__main__':
    # Train model on startup
    print("🏠 Training SmartPrice model...")
    train_model()
    print(f"✅ Model trained successfully!")
    print(f"📊 R² Score: {r2_score_value:.4f}")
    print(f"📈 RMSE: ₹{rmse_value:,.2f}")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000) 