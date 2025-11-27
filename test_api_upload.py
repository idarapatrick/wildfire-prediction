"""
Test script to upload an actual wildfire image to the API and see the response.
"""
import requests
import os

# Use the deployed API
# API_URL = "https://wildfire-prediction.onrender.com/predict"
API_URL = "http://127.0.0.1:8000/predict"  # Local testing first

# Get a wildfire image from training data
wildfire_dir = os.path.join('data', 'train', 'wildfire')
wildfire_files = [f for f in os.listdir(wildfire_dir) if f.endswith('.jpg')]

if wildfire_files:
    test_image = os.path.join(wildfire_dir, wildfire_files[0])
    
    print(f"Testing with wildfire image: {wildfire_files[0]}")
    print(f"API URL: {API_URL}")
    print("-" * 60)
    
    # Upload the file
    with open(test_image, 'rb') as f:
        files = {'file': (wildfire_files[0], f, 'image/jpeg')}
        response = requests.post(API_URL, files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n" + "=" * 60)
        print(f"PREDICTION: {result['prediction']}")
        print(f"CONFIDENCE: {result['confidence']}%")
        print(f"RAW SCORE: {result['raw_score']}")
        print("=" * 60)
        
        if result['prediction'] == "Wildfire Detected":
            print("✅ CORRECT - Wildfire image detected as wildfire")
        else:
            print("❌ WRONG - Wildfire image detected as no wildfire")
else:
    print("No wildfire images found in training data")

# Also test with a nowildfire image
nowildfire_dir = os.path.join('data', 'train', 'nowildfire')
nowildfire_files = [f for f in os.listdir(nowildfire_dir) if f.endswith('.jpg')]

if nowildfire_files:
    test_image = os.path.join(nowildfire_dir, nowildfire_files[0])
    
    print(f"\nTesting with nowildfire image: {nowildfire_files[0]}")
    print("-" * 60)
    
    with open(test_image, 'rb') as f:
        files = {'file': (nowildfire_files[0], f, 'image/jpeg')}
        response = requests.post(API_URL, files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n" + "=" * 60)
        print(f"PREDICTION: {result['prediction']}")
        print(f"CONFIDENCE: {result['confidence']}%")
        print(f"RAW SCORE: {result['raw_score']}")
        print("=" * 60)
        
        if result['prediction'] == "No Wildfire":
            print("✅ CORRECT - No wildfire image detected as no wildfire")
        else:
            print("❌ WRONG - No wildfire image detected as wildfire")
