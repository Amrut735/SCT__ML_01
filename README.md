# 🏠 HOUSELYTICS - House Price Prediction Web Application

A sophisticated machine learning web application for predicting house prices with advanced analytics and visualizations.

## 🌐 Live Application

**🚀 Deployed and Live**: [https://sct-ml-01.onrender.com](https://sct-ml-01.onrender.com)

*Experience HOUSELYTICS in action with real-time predictions and analytics!*

## 🌟 Features

- **Instant Price Prediction**: Get house price predictions instantly based on square footage, bedrooms, and bathrooms
- **Confidence Intervals**: 95% confidence intervals for predictions with RMSE error ranges
- **Advanced Visualizations**: Multiple interactive charts and plots
- **Real-time Model Statistics**: Live display of R² Score, RMSE, MAE, and training samples
- **Comprehensive Analytics**: Detailed data insights and statistical analysis
- **Prediction History**: Track all your previous predictions
- **Responsive Design**: Modern, elegant UI that works on all devices
- **Production Ready**: Optimized for cloud deployment with error handling

## 📊 Visualizations Available

- **Actual vs Predicted Plot**: Scatter plot with perfect prediction line and residuals
- **Feature Importance**: Horizontal bar chart showing feature coefficients
- **Price Distribution**: Histogram of prices and square footage vs price scatter plot

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn (Linear Regression)
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Custom CSS with gradient backgrounds
- **Deployment**: Render (Cloud Platform)
- **Production Server**: Gunicorn

## 🚀 Quick Start

### Option 1: Use Live Application
Visit **[https://sct-ml-01.onrender.com](https://sct-ml-01.onrender.com)** to use the application immediately!

### Option 2: Local Development

#### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

#### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Amrut735/SCT_ML_01.git
   cd SCT_ML_01
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app_enhanced.py
   ```

4. **Open your browser**
   Navigate to `http://127.0.0.1:5001`

## 📖 How to Use

1. **Enter Property Details**:
   - Square Footage
   - Number of Bedrooms
   - Number of Bathrooms

2. **Get Instant Prediction**:
   - Click "Predict Price" to get the estimated house price
   - View confidence intervals and error ranges

3. **Explore Visualizations**:
   - Click on different visualization buttons to see various charts
   - Analyze feature importance and data distributions

4. **Access Analytics**:
   - Navigate to the Analytics section for detailed insights
   - View comprehensive statistical analysis

5. **Learn More**:
   - Visit the About section for project information and model performance

## 📈 Model Performance

- **R² Score**: 0.8054 (80.54% accuracy)
- **RMSE**: ₹4,229,702.15
- **MAE**: ₹3,394,627.00
- **Training Samples**: 889 properties (cleaned from 1000+)
- **Model Type**: Linear Regression with MinMaxScaler normalization

## 🏗️ Project Structure

```
houselytics/
├── app_enhanced.py              # Main Flask application (production-ready)
├── app.py                       # Basic Flask application
├── app_production.py            # Production configuration
├── demo.py                      # Comprehensive demonstration script
├── house_price_predictor.py     # Core ML model implementation
├── requirements.txt             # Python dependencies
├── render.yaml                  # Render deployment configuration
├── gunicorn.conf.py            # Production server configuration
├── house_data.csv              # Training dataset
├── templates/                  # HTML templates
│   ├── index_enhanced.html     # Main page
│   ├── about.html              # About page
│   └── analytics.html          # Analytics page
├── README.md                   # This file
├── DEPLOYMENT_GUIDE.md         # Deployment instructions
├── PROJECT_SUMMARY.md          # Project overview
└── .gitignore                  # Git ignore rules
```

## 🔧 API Endpoints

- `GET /` - Main application page
- `POST /predict` - Get price prediction
- `GET /visualization/<type>` - Get visualization plots
- `GET /model_stats` - Get model performance metrics
- `GET /data_insights` - Get comprehensive data statistics
- `GET /about` - About page
- `GET /analytics` - Analytics page
- `GET /refresh` - Clear prediction history
- `GET /get_history` - Retrieve prediction history

## 🎨 Design Philosophy

HOUSELYTICS features a sophisticated "silent" color scheme with:
- Professional gray tones
- Subtle gradient backgrounds
- Clean, modern typography
- Responsive design for all devices
- Intuitive user interface

## 🔒 Security & Performance

- Input validation and sanitization
- Session-based prediction history
- Optimized model loading
- Efficient data processing
- Secure API endpoints
- Production-ready error handling
- Automatic model training on startup

## 🚀 Recent Updates

### ✅ Bug Fixes
- **Fixed startup error**: Resolved `train_model` function definition order issue
- **Improved error handling**: Enhanced exception handling for production deployment
- **Optimized deployment**: Streamlined Render deployment configuration

### 🆕 New Features
- **Live deployment**: Application now available at [https://sct-ml-01.onrender.com](https://sct-ml-01.onrender.com)
- **Enhanced reliability**: Better error recovery and model training
- **Production optimization**: Gunicorn server configuration for cloud deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Email**: info@houselytics.in
- **Project Link**: https://github.com/Amrut735/SCT_ML_01
- **Live Application**: [https://sct-ml-01.onrender.com](https://sct-ml-01.onrender.com)

## 🙏 Acknowledgments

- Built with Flask and Scikit-learn
- Data visualization powered by Matplotlib and Seaborn
- Modern UI design with CSS3 and JavaScript
- House price dataset for training and testing
- Deployed on Render cloud platform

🤝 Contributing
Feel free to fork this project and submit pull requests for improvements!

---

**HOUSELYTICS** - Making house price prediction elegant and insightful! 🏠✨

*Try it now at [https://sct-ml-01.onrender.com](https://sct-ml-01.onrender.com)* 
