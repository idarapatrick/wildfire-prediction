import tensorflow as tf
import os
from src.preprocessing import load_data
from src.model import build_model, compile_for_finetuning

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'data', 'test') # Using test as validation for simplicity
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'wildfire_model.h5')

def get_base_model_from_loaded(model):
    """
    Extract the base MobileNetV2 model from a loaded model.
    
    Args:
        model (tf.keras.Model): The loaded model.
    
    Returns:
        tf.keras.Model: The MobileNetV2 base model layer.
    """
    # The base model is typically the second layer after preprocessing
    for layer in model.layers:
        if 'mobilenetv2' in layer.name.lower():
            return layer
    return None

def run_training():
    """
    Orchestrates the full training pipeline: Data Loading -> Transfer Learning -> Fine Tuning -> Saving.
    
    This function is designed to be triggered by the API or run manually.
    If an existing model is found, it continues training from that model (incremental learning).
    Otherwise, it creates a new model from scratch.

    Returns:
        str: A status message indicating success or failure.
    """
    print("--- Starting Training Pipeline ---")
    # Load Data
    train_ds = load_data(TRAIN_DIR)
    val_ds = load_data(VAL_DIR)

    if not train_ds or not val_ds:
        return "Error: Data not found."

    # Check if existing model exists
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading existing model from {MODEL_SAVE_PATH}")
        print("This will continue training on top of your previously trained model.")
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        base_model = get_base_model_from_loaded(model)
    else:
        print("No existing model found. Building new model from scratch.")
        model, base_model = build_model()
    
    # Phase 1 Training (Transfer Learning)
    history_1 = model.fit(
        train_ds,
        epochs=4,
        validation_data=val_ds
    )

    # Phase 2 Training (Fine Tuning)
    model = compile_for_finetuning(model, base_model)
    
    total_epochs = 4 + 7
    
    history_2 = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=history_1.epoch[-1],
        validation_data=val_ds
    )

    # 5. Save Model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved successfully at: {MODEL_SAVE_PATH}")
    
    return "Training Completed Successfully"

if __name__ == "__main__":
    run_training()