"""
Test script to determine what preprocessing the trained model expects.
"""
import tensorflow as tf
import numpy as np
import os

MODEL_PATH = os.path.join('models', 'wildfire_model.h5')

# Load the model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Create a simple test image (all white pixels)
test_img_255 = np.ones((1, 128, 128, 3)) * 255.0  # [0-255] range
test_img_01 = np.ones((1, 128, 128, 3)) * 1.0      # [0-1] range

# Get predictions
pred_255 = model.predict(test_img_255, verbose=0)[0][0]
pred_01 = model.predict(test_img_01, verbose=0)[0][0]


print("MODEL PREPROCESSING TEST RESULTS")
print(f"\nInput: White image (all pixels = max value)")
print(f"  - With [0-255] range: Prediction = {pred_255:.6f}")
print(f"  - With [0-1] range:   Prediction = {pred_01:.6f}")

print("\n" + "="*60)
print("INTERPRETATION:")


# Test with a sample wildfire and nowildfire image
train_wildfire_dir = os.path.join('data', 'train', 'wildfire')
train_nowildfire_dir = os.path.join('data', 'train', 'nowildfire')

if os.path.exists(train_wildfire_dir) and os.path.exists(train_nowildfire_dir):
    wildfire_files = [f for f in os.listdir(train_wildfire_dir) if f.endswith('.jpg')]
    nowildfire_files = [f for f in os.listdir(train_nowildfire_dir) if f.endswith('.jpg')]
    
    if wildfire_files and nowildfire_files:
        # Test with actual images
        wildfire_path = os.path.join(train_wildfire_dir, wildfire_files[0])
        nowildfire_path = os.path.join(train_nowildfire_dir, nowildfire_files[0])
        
        print("\nTesting with ACTUAL training images:")
      
        
        for test_path, label in [(wildfire_path, "WILDFIRE"), (nowildfire_path, "NO WILDFIRE")]:
            img = tf.keras.utils.load_img(test_path, target_size=(128, 128))
            img_array = tf.keras.utils.img_to_array(img)
            
            # Test both preprocessing methods
            img_255 = np.expand_dims(img_array, 0)  # [0-255]
            img_01 = np.expand_dims(img_array / 255.0, 0)  # [0-1]
            
            pred_255 = model.predict(img_255, verbose=0)[0][0]
            pred_01 = model.predict(img_01, verbose=0)[0][0]
            
            print(f"\n{label} image: {os.path.basename(test_path)}")
            print(f"  Using [0-255] preprocessing: score = {pred_255:.4f}")
            print(f"    -> Predicted as: {'WILDFIRE' if pred_255 > 0.5 else 'NO WILDFIRE'}")
            print(f"  Using [0-1] preprocessing:   score = {pred_01:.4f}")
            print(f"    -> Predicted as: {'WILDFIRE' if pred_01 > 0.5 else 'NO WILDFIRE'}")

print("CONCLUSION:")

print("The correct preprocessing method is the one that predicts")
print("wildfire images with scores > 0.5 and nowildfire images < 0.5")