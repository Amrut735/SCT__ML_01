#!/usr/bin/env python3
"""
HOUSELYTICS - House Price Prediction Web Application (Production Version)
A sophisticated web interface with silent colors and advanced visualizations
Optimized for production deployment on Render
"""

import os
import io
import base64
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for production
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure matplotlib for production
plt.style.use('default')
sns.set_palette("husl")

app = Flask(__name__)
app.secret_key = 'houselytics_secret_key_2024'

# Global variables for model
model = None
scaler = None
r2_value = 0.0
rmse_value = 0.0
mae_value = 0.0
training_data = None

def train_model():
    """Train the house price prediction model."""
    global model, scaler, r2_value, rmse_value, mae_value, training_data
    
    try:
        # Load and prepare data
        data = pd.read_csv('house_data.csv')
        training_data = data.copy()
        
        # Prepare features and target
        X = data[['square_feet', 'bedrooms', 'bathrooms']]
        y = data['price']
        
        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Calculate metrics
        y_pred = model.predict(X_scaled)
        r2_value = r2_score(y, y_pred)
        rmse_value = np.sqrt(mean_squared_error(y, y_pred))
        mae_value = mean_absolute_error(y, y_pred)
        
        print(f"✅ Model trained successfully!")
        print(f"📊 R² Score: {r2_value:.4f}")
        print(f"📈 RMSE: ₹{rmse_value:,.2f}")
        print(f"📉 MAE: ₹{mae_value:,.2f}")
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        # Create dummy data if file not found
        np.random.seed(42)
        n_samples = 1000
        square_feet = np.random.uniform(800, 5000, n_samples)
        bedrooms = np.random.randint(1, 6, n_samples)
        bathrooms = np.random.uniform(1, 4, n_samples)
        price = square_feet * 5000 + bedrooms * 1000000 + bathrooms * 500000 + np.random.normal(0, 200000, n_samples)
        
        data = pd.DataFrame({
            'square_feet': square_feet,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'price': price
        })
        training_data = data.copy()
        
        X = data[['square_feet', 'bedrooms', 'bathrooms']]
        y = data['price']
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        y_pred = model.predict(X_scaled)
        r2_value = r2_score(y, y_pred)
        rmse_value = np.sqrt(mean_squared_error(y, y_pred))
        mae_value = mean_absolute_error(y, y_pred)

def predict(square_feet, bedrooms, bathrooms):
    """Make a house price prediction with confidence intervals."""
    if model is None or scaler is None:
        return None, None, None
    
    # Prepare input
    input_data = np.array([[square_feet, bedrooms, bathrooms]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Calculate confidence interval (95%)
    confidence_interval = rmse_value * 1.96
    lower_bound = max(0, prediction - confidence_interval)
    upper_bound = prediction + confidence_interval
    
    return prediction, lower_bound, upper_bound

def create_actual_vs_predicted_plot():
    """Create actual vs predicted plot."""
    if training_data is None or model is None:
        return None
    
    X = training_data[['square_feet', 'bedrooms', 'bathrooms']]
    y_actual = training_data['price']
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_actual, y_pred, alpha=0.6, color='#2c3e50')
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
    plt.xlabel('Actual Price (₹)', fontsize=12)
    plt.ylabel('Predicted Price (₹)', fontsize=12)
    plt.title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add R² score text
    plt.text(0.05, 0.95, f'R² Score: {r2_value:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

def create_feature_importance_plot():
    """Create feature importance plot."""
    if model is None:
        return None
    
    features = ['Square Feet', 'Bedrooms', 'Bathrooms']
    importance = np.abs(model.coef_)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(features, importance, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Coefficient Magnitude', fontsize=12)
    plt.title('Feature Importance Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, importance):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(importance)*0.01,
                f'{value:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

def create_price_distribution_plot():
    """Create price distribution plot."""
    if training_data is None:
        return None
    
    plt.figure(figsize=(12, 6))
    
    # Price distribution
    plt.subplot(1, 2, 1)
    plt.hist(training_data['price'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    plt.xlabel('Price (₹)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('House Price Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Square footage vs Price
    plt.subplot(1, 2, 2)
    plt.scatter(training_data['square_feet'], training_data['price'], alpha=0.6, color='#e74c3c')
    plt.xlabel('Square Feet', fontsize=12)
    plt.ylabel('Price (₹)', fontsize=12)
    plt.title('Square Feet vs Price', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    """Main page."""
    return render_template('index_enhanced.html')

@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')

@app.route('/analytics')
def analytics():
    """Analytics page."""
    return render_template('analytics.html')

@app.route('/refresh')
def refresh():
    """Refresh the application."""
    session.clear()
    return jsonify({'status': 'success', 'message': 'Application refreshed successfully!'})

@app.route('/predict', methods=['POST'])
def predict_price():
    """Predict house price."""
    try:
        data = request.get_json()
        square_feet = float(data['square_feet'])
        bedrooms = int(data['bedrooms'])
        bathrooms = float(data['bathrooms'])
        
        prediction, lower_bound, upper_bound = predict(square_feet, bedrooms, bathrooms)
        
        if prediction is not None:
            # Store in session
            if 'predictions' not in session:
                session['predictions'] = []
            
            prediction_data = {
                'id': len(session['predictions']) + 1,
                'square_feet': square_feet,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'predicted_price': prediction,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            session['predictions'].append(prediction_data)
            session.modified = True
            
            return jsonify({
                'status': 'success',
                'prediction': prediction,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'rmse': rmse_value
            })
        else:
            return jsonify({'status': 'error', 'message': 'Model not trained'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/visualization/<plot_type>')
def get_visualization(plot_type):
    """Get visualization plots."""
    try:
        if plot_type == 'actual_vs_predicted':
            plot_data = create_actual_vs_predicted_plot()
        elif plot_type == 'feature_importance':
            plot_data = create_feature_importance_plot()
        elif plot_type == 'price_distribution':
            plot_data = create_price_distribution_plot()
        else:
            return jsonify({'status': 'error', 'message': 'Invalid plot type'})
        
        if plot_data:
            return jsonify({'status': 'success', 'plot': plot_data})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to generate plot'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_history')
def get_history():
    """Get prediction history."""
    predictions = session.get('predictions', [])
    return jsonify({'predictions': predictions})

@app.route('/model_stats')
def model_stats():
    """Get model statistics."""
    return jsonify({
        'r2_score': r2_value,
        'rmse': rmse_value,
        'mae': mae_value,
        'training_samples': len(training_data) if training_data is not None else 0
    })

@app.route('/data_insights')
def data_insights():
    """Get data insights."""
    if training_data is None:
        return jsonify({'status': 'error', 'message': 'No training data available'})
    
    insights = {
        'total_properties': len(training_data),
        'avg_price': training_data['price'].mean(),
        'median_price': training_data['price'].median(),
        'min_price': training_data['price'].min(),
        'max_price': training_data['price'].max(),
        'avg_square_feet': training_data['square_feet'].mean(),
        'avg_bedrooms': training_data['bedrooms'].mean(),
        'avg_bathrooms': training_data['bathrooms'].mean(),
        'price_std': training_data['price'].std(),
        'square_feet_std': training_data['square_feet'].std()
    }
    
    return jsonify(insights)

if __name__ == '__main__':
    # Train model on startup
    print("🏠 Training HOUSELYTICS model...")
    train_model()
    print(f"✅ Model trained successfully!")
    
    # Get port from environment variable (for Render) or use default
    port = int(os.environ.get('PORT', 5001))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False) 