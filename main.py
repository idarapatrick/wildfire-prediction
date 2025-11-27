from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from typing import List
import shutil
import os
import time
from datetime import datetime

from src.prediction import predict_image
from src.train_pipeline import run_training

app = FastAPI(title="Wildfire Prediction API")

START_TIME = datetime.now()

# Training status tracker
training_status = {
    "is_training": False,
    "last_training_time": None,
    "last_training_result": None,
    "last_training_message": None,
    "training_logs": []  # Store training progress logs
}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_WILDFIRE_DIR = os.path.join(BASE_DIR, 'data', 'train', 'wildfire')
TRAIN_NOWILDFIRE_DIR = os.path.join(BASE_DIR, 'data', 'train', 'nowildfire')

@app.get("/")
def home():
    """
    Health check endpoint.
    Returns:
        dict: A welcome message confirming the API is live.
    """
    return {"message": "The Wildfire Detection API is live. Use /docs for the interface."}

@app.get("/status")
def get_status():
    """
    Returns the model up-time and system status.
    Required for the 'Model up-time' UI task.
    
    Returns:
        dict: uptime string and status.
    """
    current_time = datetime.now()
    uptime_duration = current_time - START_TIME
    return {
        "status": "online",
        "uptime": str(uptime_duration).split('.')[0], # Format: HH:MM:SS
        "started_at": START_TIME.isoformat()
    }

@app.get("/debug")
def debug_info():
    """
    Debug endpoint to verify preprocessing version.
    """
    import inspect
    from src.prediction import predict_image
    
    # Get the source code of predict_image function
    source = inspect.getsource(predict_image)
    has_normalization = "/ 255.0" in source
    
    # Check if model file exists
    model_path = os.path.join(BASE_DIR, 'models', 'wildfire_model.h5')
    model_exists = os.path.exists(model_path)
    model_size = os.path.getsize(model_path) if model_exists else 0
    
    return {
        "preprocessing_version": "v2_with_normalization" if has_normalization else "v1_without_normalization",
        "has_divide_by_255": has_normalization,
        "model_file_exists": model_exists,
        "model_file_path": model_path,
        "model_size_mb": round(model_size / (1024*1024), 2) if model_exists else 0,
        "base_dir": BASE_DIR,
        "code_snippet": source[500:800] if len(source) > 500 else source
    }

@app.get("/test_prediction")
def test_prediction():
    """
    Test prediction with a sample training image to verify preprocessing.
    """
    import numpy as np
    from src.prediction import load_model_instance
    
    # Create a test image: white pixels (max value)
    test_img_array = np.ones((1, 128, 128, 3)) * 255.0
    test_img_normalized = test_img_array / 255.0
    
    model = load_model_instance()
    
    # Test with both preprocessing methods
    pred_raw = model.predict(test_img_array, verbose=0)[0][0]
    pred_normalized = model.predict(test_img_normalized, verbose=0)[0][0]
    
    # Also test with an actual wildfire image if available
    wildfire_test = None
    wildfire_path = os.path.join(TRAIN_WILDFIRE_DIR)
    if os.path.exists(wildfire_path):
        wildfire_files = [f for f in os.listdir(wildfire_path) if f.endswith('.jpg')]
        if wildfire_files:
            import tensorflow as tf
            test_file = os.path.join(wildfire_path, wildfire_files[0])
            img = tf.keras.utils.load_img(test_file, target_size=(128, 128))
            img_array = tf.keras.utils.img_to_array(img)
            img_normalized = np.expand_dims(img_array / 255.0, 0)
            wildfire_test = {
                "file": wildfire_files[0],
                "prediction_score": float(model.predict(img_normalized, verbose=0)[0][0]),
                "predicted_as": "WILDFIRE" if model.predict(img_normalized, verbose=0)[0][0] > 0.5 else "NO WILDFIRE"
            }
    
    return {
        "white_image_test": {
            "raw_255_score": float(pred_raw),
            "normalized_01_score": float(pred_normalized)
        },
        "actual_wildfire_image": wildfire_test,
        "interpretation": "Normalized (0-1) preprocessing should give higher scores for wildfire images"
    }

@app.get("/data_stats")
def get_data_stats():
    """
    Returns statistics about the training data for visualizations.
    Required for 'Visualizations' UI task.
    
    Returns:
        dict: Counts of images in each category.
    """
    # Count files in directories
    wildfire_count = len(os.listdir(TRAIN_WILDFIRE_DIR)) if os.path.exists(TRAIN_WILDFIRE_DIR) else 0
    nowildfire_count = len(os.listdir(TRAIN_NOWILDFIRE_DIR)) if os.path.exists(TRAIN_NOWILDFIRE_DIR) else 0
    
    return {
        "wildfire": wildfire_count,
        "nowildfire": nowildfire_count,
        "total": wildfire_count + nowildfire_count
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict wildfire from an uploaded image.
    Args:
        file (UploadFile): The image file uploaded by the user.
    Returns:
        dict: prediction result containing class and confidence.
    """
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        result = predict_image(temp_filename)
        return result
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/upload_data")
async def upload_training_data(
    label: str, 
    files: List[UploadFile] = File(...)
):
    """
    Endpoint to upload bulk data for future training.
    Args:
        label (str): The class label ('wildfire' or 'nowildfire').
        files (List[UploadFile]): A list of image files to upload.
    Returns:
        dict: A confirmation message.
    """
    if label not in ['wildfire', 'nowildfire']:
        return {"error": "Label must be 'wildfire' or 'nowildfire'"}

    target_dir = TRAIN_WILDFIRE_DIR if label == 'wildfire' else TRAIN_NOWILDFIRE_DIR
    os.makedirs(target_dir, exist_ok=True)
    
    count = 0
    for file in files:
        file_path = os.path.join(target_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        count += 1
        
    return {"message": f"Successfully uploaded {count} images to class '{label}'"}

@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Endpoint to trigger the model retraining pipeline.
    Runs in background and updates training status.
    """
    global training_status
    
    if training_status["is_training"]:
        return {"error": "Training already in progress. Please wait for it to complete."}
    
    training_status["is_training"] = True
    training_status["last_training_result"] = None
    training_status["last_training_message"] = "Training in progress..."
    training_status["training_logs"] = []  # Clear previous logs
    
    def training_task():
        global training_status
        
        try:
            # Pass the training_status dict to run_training so it can update logs in real-time
            result = run_training(status_dict=training_status)
            
            training_status["is_training"] = False
            training_status["last_training_time"] = datetime.now().isoformat()
            training_status["last_training_result"] = "success"
            training_status["last_training_message"] = result
            training_status["training_logs"].append(f"\n✅ {result}")
            
        except Exception as e:
            training_status["is_training"] = False
            training_status["last_training_time"] = datetime.now().isoformat()
            training_status["last_training_result"] = "error"
            training_status["last_training_message"] = f"Training failed: {str(e)}"
            training_status["training_logs"].append(f"\n❌ Error: {str(e)}")
    
    background_tasks.add_task(training_task)
    return {"message": "Retraining started in background. Use /training_status to check progress."}

@app.get("/training_status")
def get_training_status():
    """
    Get the current status of model training.
    
    Returns:
        dict: Training status information including whether training is in progress,
              last training time, result, and detailed message.
    """
    return training_status