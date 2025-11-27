import tensorflow as tf
import os
import shutil
from datetime import datetime
from src.preprocessing import load_data
from src.model import build_model, compile_for_finetuning

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'data', 'test') # Using test as validation for simplicity
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'wildfire_model.h5')
BACKUP_DIR = os.path.join(BASE_DIR, 'models', 'backups')

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
    The new model is only saved if it performs better than the existing one.

    Returns:
        str: A status message indicating success or failure.
    """
    print("--- Starting Training Pipeline ---")
    
    # Load Data
    train_ds = load_data(TRAIN_DIR)
    val_ds = load_data(VAL_DIR)

    if not train_ds or not val_ds:
        return "Error: Data not found."

    # Evaluate existing model performance if it exists
    existing_model_accuracy = None
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading existing model from {MODEL_SAVE_PATH}")
        existing_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        
        # Evaluate current model
        print("Evaluating existing model performance...")
        existing_loss, existing_model_accuracy = existing_model.evaluate(val_ds, verbose=0)
        print(f"Existing model validation accuracy: {existing_model_accuracy:.4f}")
        
        # Create backup
        os.makedirs(BACKUP_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f'wildfire_model_backup_{timestamp}.h5')
        shutil.copy(MODEL_SAVE_PATH, backup_path)
        print(f"Backup created at: {backup_path}")
        
        print("Continuing training on top of the existing model...")
        model = existing_model
        base_model = get_base_model_from_loaded(model)
    else:
        print("No existing model found. Building new model from scratch.")
        model, base_model = build_model()
    
    # Phase 1 Training (Transfer Learning)
    print("\n--- Phase 1: Transfer Learning ---")
    history_1 = model.fit(
        train_ds,
        epochs=4,
        validation_data=val_ds
    )

    # Phase 2 Training (Fine Tuning)
    print("\n--- Phase 2: Fine Tuning ---")
    model = compile_for_finetuning(model, base_model)
    
    total_epochs = 4 + 7
    
    history_2 = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=history_1.epoch[-1],
        validation_data=val_ds
    )

    # Evaluate new model
    print("\n--- Evaluating Retrained Model ---")
    new_loss, new_model_accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Retrained model validation accuracy: {new_model_accuracy:.4f}")
    
    # Compare and decide whether to keep new model
    if existing_model_accuracy is not None:
        if new_model_accuracy >= existing_model_accuracy:
            print(f"\n✅ Performance improved! ({existing_model_accuracy:.4f} → {new_model_accuracy:.4f})")
            print(f"Saving new model to: {MODEL_SAVE_PATH}")
            model.save(MODEL_SAVE_PATH)
            return f"Training Completed Successfully. Accuracy improved from {existing_model_accuracy:.2%} to {new_model_accuracy:.2%}"
        else:
            print(f"\n⚠️ Performance decreased ({existing_model_accuracy:.4f} → {new_model_accuracy:.4f})")
            print("Keeping existing model. New model discarded.")
            return f"Training completed but existing model retained. New model accuracy ({new_model_accuracy:.2%}) did not exceed existing ({existing_model_accuracy:.2%})"
    else:
        # First time training
        model.save(MODEL_SAVE_PATH)
        print(f"Model saved successfully at: {MODEL_SAVE_PATH}")
        return f"Training Completed Successfully. Model accuracy: {new_model_accuracy:.2%}"

if __name__ == "__main__":
    run_training()