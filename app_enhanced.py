#!/usr/bin/env python3
"""
HOUSELYTICS - House Price Prediction Web Application
A sophisticated web interface with silent colors and advanced visualizations
"""

from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import io
import base64
import json
from datetime import datetime
import os

# Set seaborn style for better plots
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

app = Flask(__name__)
app.secret_key = 'houselytics_secret_key_2024'

# Global variables for model
model = None
scaler = None
r2_score_value = None
rmse_value = None
mae_value = None
training_data = None

def train_model():
    """Train the house price prediction model."""
    global model, scaler, r2_score_value, rmse_value, mae_value, training_data
    
    # Load and preprocess data
    df = pd.read_csv('house_data.csv')
    
    # Convert USD to INR (1 USD ≈ 83 INR)
    df['price'] = df['price'] * 83
    
    # Drop missing values
    df = df.dropna()
    
    # Store training data for visualizations
    training_data = df.copy()
    
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
    mae_value = mean_absolute_error(y_test, y_pred)
    
    return model, scaler, r2_score_value, rmse_value, mae_value

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

def create_actual_vs_predicted_plot():
    """Create actual vs predicted visualization."""
    if training_data is None:
        return None
    
    # Prepare data
    X = training_data[['square_footage', 'bedrooms', 'bathrooms']].values
    y = training_data['price'].values
    
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Actual vs Predicted scatter plot
    ax1.scatter(y_test, y_pred, alpha=0.6, color='#6B7280', s=60, edgecolors='#374151', linewidth=0.5)
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], '--', color='#DC2626', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Price (₹)', fontsize=12, color='#374151')
    ax1.set_ylabel('Predicted Price (₹)', fontsize=12, color='#374151')
    ax1.set_title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold', color='#111827')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add R² score to plot
    r2 = r2_score(y_test, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F9FAFB", alpha=0.9, edgecolor='#D1D5DB'))
    
    # Residuals plot
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, color='#6B7280', s=60, edgecolors='#374151', linewidth=0.5)
    ax2.axhline(y=0, color='#DC2626', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Price (₹)', fontsize=12, color='#374151')
    ax2.set_ylabel('Residuals (₹)', fontsize=12, color='#374151')
    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold', color='#111827')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def create_feature_importance_plot():
    """Create feature importance visualization."""
    if model is None:
        return None
    
    # Get feature importance (coefficients)
    feature_names = ['Square Footage', 'Bedrooms', 'Bathrooms']
    coefficients = model.coef_
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar plot
    bars = ax.barh(feature_names, coefficients, color=['#6B7280', '#9CA3AF', '#D1D5DB'])
    
    # Add value labels on bars
    for i, (bar, coef) in enumerate(zip(bars, coefficients)):
        ax.text(bar.get_width() + (max(coefficients) * 0.01), bar.get_y() + bar.get_height()/2, 
                f'{coef:.2e}', va='center', fontsize=10, color='#374151')
    
    ax.set_xlabel('Coefficient Value', fontsize=12, color='#374151')
    ax.set_title('Feature Importance (Linear Regression Coefficients)', fontsize=14, fontweight='bold', color='#111827')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def create_price_distribution_plot():
    """Create price distribution visualization."""
    if training_data is None:
        return None
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Price histogram
    ax1.hist(training_data['price'], bins=30, color='#6B7280', alpha=0.7, edgecolor='#374151')
    ax1.set_xlabel('Price (₹)', fontsize=12, color='#374151')
    ax1.set_ylabel('Frequency', fontsize=12, color='#374151')
    ax1.set_title('House Price Distribution', fontsize=14, fontweight='bold', color='#111827')
    ax1.grid(True, alpha=0.3)
    
    # Square footage vs Price scatter
    ax2.scatter(training_data['square_footage'], training_data['price'], 
               alpha=0.6, color='#6B7280', s=50, edgecolors='#374151', linewidth=0.5)
    ax2.set_xlabel('Square Footage', fontsize=12, color='#374151')
    ax2.set_ylabel('Price (₹)', fontsize=12, color='#374151')
    ax2.set_title('Square Footage vs Price', fontsize=14, fontweight='bold', color='#111827')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def create_advanced_analytics_plot():
    """Create advanced analytics visualization."""
    if training_data is None:
        return None
    
    # Create comprehensive analytics plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Price vs Bedrooms box plot
    training_data.boxplot(column='price', by='bedrooms', ax=ax1, color='#6B7280')
    ax1.set_title('Price Distribution by Bedrooms', fontsize=12, fontweight='bold', color='#111827')
    ax1.set_xlabel('Number of Bedrooms', fontsize=10, color='#374151')
    ax1.set_ylabel('Price (₹)', fontsize=10, color='#374151')
    ax1.grid(True, alpha=0.3)
    
    # 2. Price vs Bathrooms box plot
    training_data.boxplot(column='price', by='bathrooms', ax=ax2, color='#9CA3AF')
    ax2.set_title('Price Distribution by Bathrooms', fontsize=12, fontweight='bold', color='#111827')
    ax2.set_xlabel('Number of Bathrooms', fontsize=10, color='#374151')
    ax2.set_ylabel('Price (₹)', fontsize=10, color='#374151')
    ax2.grid(True, alpha=0.3)
    
    # 3. Square footage distribution
    ax3.hist(training_data['square_footage'], bins=25, color='#D1D5DB', alpha=0.7, edgecolor='#374151')
    ax3.set_title('Square Footage Distribution', fontsize=12, fontweight='bold', color='#111827')
    ax3.set_xlabel('Square Footage', fontsize=10, color='#374151')
    ax3.set_ylabel('Frequency', fontsize=10, color='#374151')
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation heatmap
    corr_matrix = training_data[['square_footage', 'bedrooms', 'bathrooms', 'price']].corr()
    im = ax4.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto')
    ax4.set_xticks(range(len(corr_matrix.columns)))
    ax4.set_yticks(range(len(corr_matrix.columns)))
    ax4.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax4.set_yticklabels(corr_matrix.columns)
    ax4.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold', color='#111827')
    
    # Add correlation values to heatmap
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    
    # Save plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/')
def index():
    """Main page with prediction form."""
    return render_template('index_enhanced.html')

@app.route('/about')
def about():
    """About page with detailed information."""
    return render_template('about.html')

@app.route('/analytics')
def analytics():
    """Analytics page with comprehensive data analysis."""
    return render_template('analytics.html')

@app.route('/refresh')
def refresh():
    """Refresh the application and clear form."""
    # Clear prediction history
    session['prediction_history'] = []
    return jsonify({
        'success': True,
        'message': 'Application refreshed successfully!'
    })

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
        
        # Calculate confidence interval (simplified)
        confidence_range = rmse_value * 1.96  # 95% confidence interval
        lower_bound = max(0, predicted_price - confidence_range)
        upper_bound = predicted_price + confidence_range
        
        # Create prediction history entry
        prediction_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'square_footage': square_footage,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'predicted_price': predicted_price,
            'formatted_price': formatted_price,
            'confidence_lower': format_price_inr(lower_bound),
            'confidence_upper': format_price_inr(upper_bound)
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
            'confidence_lower': format_price_inr(lower_bound),
            'confidence_upper': format_price_inr(upper_bound),
            'rmse': rmse_value,
            'mae': mae_value,
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

@app.route('/visualization/<plot_type>')
def visualization(plot_type):
    """Generate and return visualization."""
    try:
        if plot_type == 'actual_vs_predicted':
            plot_url = create_actual_vs_predicted_plot()
        elif plot_type == 'feature_importance':
            plot_url = create_feature_importance_plot()
        elif plot_type == 'price_distribution':
            plot_url = create_price_distribution_plot()

        else:
            return jsonify({
                'success': False,
                'error': 'Invalid plot type'
            })
        
        if plot_url is None:
            return jsonify({
                'success': False,
                'error': 'Model not trained. Please try again.'
            })
        
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

@app.route('/model_stats')
def model_stats():
    """Get model statistics."""
    return jsonify({
        'success': True,
        'r2_score': r2_score_value,
        'rmse': rmse_value,
        'mae': mae_value,
        'training_samples': len(training_data) if training_data is not None else 0
    })

@app.route('/data_insights')
def data_insights():
    """Get comprehensive data insights."""
    if training_data is None:
        return jsonify({
            'success': False,
            'error': 'No training data available'
        })
    
    try:
        # Calculate various statistics
        price_stats = {
            'mean': training_data['price'].mean(),
            'median': training_data['price'].median(),
            'std': training_data['price'].std(),
            'min': training_data['price'].min(),
            'max': training_data['price'].max()
        }
        
        sqft_stats = {
            'mean': training_data['square_footage'].mean(),
            'median': training_data['square_footage'].median(),
            'std': training_data['square_footage'].std(),
            'min': training_data['square_footage'].min(),
            'max': training_data['square_footage'].max()
        }
        
        # Bedroom and bathroom distributions
        bedroom_dist = training_data['bedrooms'].value_counts().to_dict()
        bathroom_dist = training_data['bathrooms'].value_counts().to_dict()
        
        # Price per square foot
        price_per_sqft = training_data['price'] / training_data['square_footage']
        price_per_sqft_stats = {
            'mean': price_per_sqft.mean(),
            'median': price_per_sqft.median(),
            'std': price_per_sqft.std()
        }
        
        return jsonify({
            'success': True,
            'price_stats': price_stats,
            'sqft_stats': sqft_stats,
            'bedroom_distribution': bedroom_dist,
            'bathroom_distribution': bathroom_dist,
            'price_per_sqft': price_per_sqft_stats,
            'total_properties': len(training_data)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error calculating insights: {str(e)}'
        })

if __name__ == '__main__':
    # Train model on startup
    print("🏠 Training HOUSELYTICS model...")
    train_model()
    print(f"✅ Model trained successfully!")
    print(f"📊 R² Score: {r2_score_value:.4f}")
    print(f"📈 RMSE: ₹{rmse_value:,.2f}")
    print(f"📉 MAE: ₹{mae_value:,.2f}")
    
    # Get port from environment variable (for Render) or use default
    import os
    port = int(os.environ.get('PORT', 5001))
    
    # Run the app
    app.run(debug=False, host='0.0.0.0', port=port) 