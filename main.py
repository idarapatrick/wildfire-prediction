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
    Runs in background.
    """
    background_tasks.add_task(run_training)
    return {"message": "Retraining started in background. The model will update automatically upon completion."}