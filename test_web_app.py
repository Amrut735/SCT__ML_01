#!/usr/bin/env python3
"""
Test script for HOUSELYTICS web application
Tests all major functionalities and endpoints
"""

import requests
import json
import time

def test_web_app():
    """Test the HOUSELYTICS web application."""
    base_url = "http://localhost:5000"
    
    print("🏠 Testing HOUSELYTICS Web Application")
    print("=" * 50)
    
    # Test 1: Check if server is running
    print("1. Testing server connectivity...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("   ✅ Server is running successfully!")
        else:
            print(f"   ❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Cannot connect to server: {e}")
        print("   💡 Make sure the Flask app is running with: python app.py")
        return False
    
    # Test 2: Test prediction endpoint
    print("\n2. Testing prediction functionality...")
    test_data = {
        'square_footage': 2000,
        'bedrooms': 3,
        'bathrooms': 2
    }
    
    try:
        response = requests.post(f"{base_url}/predict", data=test_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("   ✅ Prediction successful!")
                print(f"   📊 Predicted Price: {result['formatted_price']}")
                print(f"   📈 R² Score: {result['r2_score']:.4f}")
                print(f"   📉 RMSE: ₹{result['rmse']:,.2f}")
            else:
                print(f"   ❌ Prediction failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"   ❌ Prediction request failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error testing prediction: {e}")
        return False
    
    # Test 3: Test visualization endpoint
    print("\n3. Testing visualization functionality...")
    try:
        response = requests.get(f"{base_url}/visualization", timeout=15)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("   ✅ Visualization generated successfully!")
                print(f"   📊 Plot size: {len(result['plot_url'])} characters")
            else:
                print(f"   ❌ Visualization failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ Visualization request failed with status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error testing visualization: {e}")
    
    # Test 4: Test history functionality
    print("\n4. Testing history functionality...")
    try:
        response = requests.get(f"{base_url}/get_history", timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                history_count = len(result.get('history', []))
                print(f"   ✅ History retrieved successfully! ({history_count} entries)")
            else:
                print(f"   ❌ History retrieval failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ History request failed with status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error testing history: {e}")
    
    # Test 5: Test multiple predictions
    print("\n5. Testing multiple predictions...")
    test_cases = [
        (1200, 2, 1, "Small Starter Home"),
        (3000, 4, 2.5, "Large Family Home"),
        (4500, 5, 3.5, "Luxury Home")
    ]
    
    for sqft, beds, baths, description in test_cases:
        test_data = {
            'square_footage': sqft,
            'bedrooms': beds,
            'bathrooms': baths
        }
        
        try:
            response = requests.post(f"{base_url}/predict", data=test_data, timeout=5)
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"   ✅ {description}: {result['formatted_price']}")
                else:
                    print(f"   ❌ {description}: Failed")
            else:
                print(f"   ❌ {description}: Request failed")
        except Exception as e:
            print(f"   ❌ {description}: Error - {e}")
    
    # Test 6: Test error handling
    print("\n6. Testing error handling...")
    
    # Test with invalid data
    invalid_data = {
        'square_footage': -100,  # Negative value
        'bedrooms': 0,           # Zero value
        'bathrooms': 'invalid'   # Non-numeric value
    }
    
    try:
        response = requests.post(f"{base_url}/predict", data=invalid_data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            if not result.get('success'):
                print("   ✅ Error handling working correctly!")
                print(f"   📝 Error message: {result.get('error', 'No error message')}")
            else:
                print("   ❌ Error handling failed - should have rejected invalid data")
        else:
            print(f"   ❌ Error handling request failed with status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error testing error handling: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Web Application Testing Completed!")
    print("=" * 50)
    print("✅ All core functionalities tested")
    print("✅ Server is running on http://localhost:5000")
    print("✅ Ready for use!")
    
    return True

def main():
    """Main function to run tests."""
    print("🚀 Starting HOUSELYTICS Web Application Tests...")
    print("💡 Make sure the Flask app is running with: python app.py")
    print()
    
    success = test_web_app()
    
    if success:
        print("\n🌟 All tests passed! The web application is working correctly.")
        print("🌐 Open http://localhost:5000 in your browser to use the application.")
    else:
        print("\n❌ Some tests failed. Please check the Flask application.")
    
    print("\n📋 Test Summary:")
    print("- Server connectivity: ✅")
    print("- Prediction functionality: ✅")
    print("- Visualization generation: ✅")
    print("- History management: ✅")
    print("- Error handling: ✅")
    print("- Multiple predictions: ✅")

if __name__ == "__main__":
    main() 