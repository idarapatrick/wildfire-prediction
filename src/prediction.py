import tensorflow as tf
import numpy as np
import os


IMG_SIZE = (128, 128)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'wildfire_model.h5')

# Global variable to hold the model 
_model = None

def load_model_instance():
    """
    Loads the Keras model from disk if it is not already loaded.
    
    Returns:
        tf.keras.Model: The loaded Keras model instance.
    
    Raises:
        FileNotFoundError: If the .h5 model file does not exist.
    """
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")
    return _model

def predict_image(image_file):
    """
    Processes an image file and generates a wildfire prediction.

    Args:
        image_file (str or file-like): The path to the image or file object to predict on.

    Returns:
        dict: A dictionary containing:
            - prediction (str): 'Wildfire Detected' or 'No Wildfire'.
            - confidence (float): The percentage confidence of the prediction.
            - raw_score (float): The raw sigmoid output (0 to 1).
    """
    model = load_model_instance()

    # Load image - this returns PIL Image in RGB mode
    img = tf.keras.utils.load_img(image_file, target_size=IMG_SIZE)
    
    # Convert PIL Image to numpy array [0-255]
    img_array = tf.keras.utils.img_to_array(img)
    
    print(f"Debug - Loaded image shape: {img_array.shape}, dtype: {img_array.dtype}")
    print(f"Debug - Pixel range: min={img_array.min():.2f}, max={img_array.max():.2f}, mean={img_array.mean():.2f}")
    
    # Normalize to [0, 1] to match training (notebook divided by 255)
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    print(f"Debug - After preprocessing shape: {img_array.shape}, range: [{img_array.min():.6f}, {img_array.max():.6f}]")

    # Predict
    prediction_score = model.predict(img_array, verbose=0)[0][0]
    
    # Interpret Result (Sigmoid output: 0 to 1)
    # Class mapping: 0=nowildfire, 1=wildfire (alphabetical order)
    if prediction_score > 0.5:
        result = "Wildfire Detected"
        confidence = float(prediction_score)
    else:
        result = "No Wildfire"
        confidence = 1.0 - float(prediction_score)
    
    print(f"Debug - Raw score: {prediction_score:.6f}, Prediction: {result}, Confidence: {confidence:.4f}")
        
    return {
        "prediction": result,
        "confidence": round(confidence * 100, 2),
        "raw_score": float(prediction_score)
    }