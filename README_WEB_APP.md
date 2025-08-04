# 🏠 SmartPrice - House Price Prediction Web Application

A modern, responsive web application for predicting house prices in Indian Rupees (INR) with a beautiful user interface and real-time predictions.

## 🌟 Features

### 🎨 **Modern UI Design**
- **Silent Colors**: Professional color scheme with gradients and subtle shadows
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **Interactive Elements**: Smooth animations and hover effects
- **Clean Layout**: Two-column design with input form and prediction history

### 🏠 **Core Functionality**
- **Real-time Predictions**: Instant house price predictions in INR
- **Input Validation**: Ensures all inputs are positive numbers
- **Indian Context**: Prices displayed in lakhs and crores
- **Prediction History**: Stores last 10 predictions with timestamps

### 📊 **Advanced Features**
- **Model Performance Metrics**: R² Score and RMSE displayed
- **Visualization**: Actual vs Predicted prices chart (matplotlib)
- **Session Management**: Persistent prediction history across sessions
- **Error Handling**: User-friendly error messages

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

### 3. Access the Web App
Open your browser and go to: `http://localhost:5000`

## 📱 User Interface

### **Header Section**
- SmartPrice logo with house icon
- Navigation menu (Home, About)
- Professional dark blue header

### **Main Content Area**
- **Left Panel**: Property Details Input Form
  - Square footage input with ruler icon
  - Bedrooms input with bed icon
  - Bathrooms input with bath icon
  - "Predict Price" button with lightbulb icon

- **Right Panel**: Prediction History
  - Real-time updates of predictions
  - Timestamp for each prediction
  - Clear history functionality

### **Visualization Section**
- Generate actual vs predicted prices chart
- R² score displayed on the plot
- Professional matplotlib visualization

### **Footer**
- Contact information
- Phone, email, and address details

## 🎯 How to Use

1. **Enter Property Details**:
   - Fill in square footage (e.g., 2000)
   - Enter number of bedrooms (e.g., 3)
   - Specify number of bathrooms (e.g., 2)

2. **Get Instant Prediction**:
   - Click "Predict Price" button
   - View predicted price in INR with lakhs/crores
   - See model performance metrics (R² Score, RMSE)

3. **View History**:
   - Check prediction history on the right panel
   - See all previous predictions with timestamps

4. **Generate Visualization**:
   - Click "Generate Visualization" button
   - View actual vs predicted prices chart

## 🛠️ Technical Architecture

### **Backend (Flask)**
- **Model Training**: Linear regression with MinMaxScaler
- **API Endpoints**: 
  - `/predict` - Handle prediction requests
  - `/visualization` - Generate matplotlib charts
  - `/get_history` - Retrieve prediction history
  - `/clear_history` - Clear prediction history

### **Frontend (HTML/CSS/JavaScript)**
- **Responsive Design**: CSS Grid and Flexbox
- **Interactive JavaScript**: AJAX requests and DOM manipulation
- **Modern Styling**: CSS3 with gradients and animations

### **Data Processing**
- **INR Conversion**: USD to INR (1:83 ratio)
- **Feature Scaling**: MinMaxScaler for optimal performance
- **Model Evaluation**: R² Score and RMSE calculation

## 🎨 Design Features

### **Color Scheme**
- **Primary**: #3498db (Blue)
- **Secondary**: #2c3e50 (Dark Blue)
- **Success**: #27ae60 (Green)
- **Background**: Gradient from #667eea to #764ba2
- **Cards**: White with subtle shadows

### **Typography**
- **Font Family**: Segoe UI, Tahoma, Geneva, Verdana, sans-serif
- **Headings**: Bold with proper hierarchy
- **Body Text**: Clean and readable

### **Interactive Elements**
- **Buttons**: Gradient backgrounds with hover effects
- **Input Fields**: Icons and focus states
- **Loading Spinner**: Animated CSS spinner
- **Hover Effects**: Smooth transitions

## 📊 Model Performance

The web application uses the same optimized model as the command-line version:

- **R² Score**: ~80.5% accuracy
- **Training Time**: <0.01 seconds
- **Prediction Time**: <0.001 seconds per house
- **Currency**: All prices in Indian Rupees (INR)

## 🔧 API Endpoints

### **POST /predict**
Predict house price based on input parameters.

**Request Body:**
```json
{
  "square_footage": 2000,
  "bedrooms": 3,
  "bathrooms": 2
}
```

**Response:**
```json
{
  "success": true,
  "predicted_price": 45906626.32,
  "formatted_price": "₹45,906,626.32 (₹4.59 crores)",
  "rmse": 4229702.15,
  "r2_score": 0.8054,
  "prediction_history": [...]
}
```

### **GET /visualization**
Generate and return actual vs predicted prices chart.

**Response:**
```json
{
  "success": true,
  "plot_url": "base64_encoded_image_data"
}
```

### **GET /get_history**
Retrieve prediction history from session.

### **POST /clear_history**
Clear prediction history.

## 🚀 Deployment

### **Local Development**
```bash
python app.py
```

### **Production Deployment**
For production deployment, consider using:
- **Gunicorn**: `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
- **Docker**: Containerize the application
- **Cloud Platforms**: Deploy to Heroku, AWS, or Google Cloud

## 📱 Mobile Responsiveness

The application is fully responsive and works on:
- **Desktop**: Full two-column layout
- **Tablet**: Adaptive layout with proper spacing
- **Mobile**: Single-column layout with touch-friendly buttons

## 🔒 Security Features

- **Input Validation**: Server-side validation of all inputs
- **Error Handling**: Graceful error handling and user feedback
- **Session Management**: Secure session handling for prediction history

## 🎉 Key Advantages

1. **User-Friendly**: Intuitive interface with clear instructions
2. **Real-Time**: Instant predictions with loading indicators
3. **Professional**: Modern design suitable for business use
4. **Responsive**: Works on all devices and screen sizes
5. **Indian Context**: Prices in INR with lakhs/crores display
6. **Visual**: Interactive charts and performance metrics
7. **Persistent**: Maintains prediction history across sessions

## 🌟 Ready for Production

The SmartPrice web application is production-ready with:
- ✅ Modern, responsive UI
- ✅ Real-time predictions
- ✅ Professional design
- ✅ Error handling
- ✅ Performance optimization
- ✅ Indian market focus
- ✅ Visualization capabilities

Visit `http://localhost:5000` to experience the SmartPrice web application! 🏠🇮🇳 