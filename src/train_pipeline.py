import tensorflow as tf
import os
import shutil
from datetime import datetime
from src.preprocessing import load_data
from src.model import build_model, compile_for_finetuning

# Global callback for progress tracking
progress_callback = None

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

class ProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to track training progress in real-time."""
    def __init__(self, status_dict, phase_name):
        super().__init__()
        self.status_dict = status_dict
        self.phase_name = phase_name
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_msg = f"{self.phase_name} - Epoch {epoch + 1}: loss={logs.get('loss', 0):.4f}, accuracy={logs.get('accuracy', 0):.4f}, val_loss={logs.get('val_loss', 0):.4f}, val_accuracy={logs.get('val_accuracy', 0):.4f}"
        if 'training_logs' in self.status_dict:
            self.status_dict['training_logs'].append(log_msg)
        print(log_msg)

def run_training(status_dict=None):
    """
    Orchestrates the full training pipeline: Data Loading -> Transfer Learning -> Fine Tuning -> Saving.
    
    This function is designed to be triggered by the API or run manually.
    If an existing model is found, it continues training from that model (incremental learning).
    The new model is only saved if it performs better than the existing one.

    Args:
        status_dict (dict): Optional dictionary to track training progress in real-time.

    Returns:
        str: A status message indicating success or failure.
    """
    if status_dict:
        status_dict['training_logs'].append("Starting Training Pipeline")
    print("Starting Training Pipeline")
    
    # Load Data
    if status_dict:
        status_dict['training_logs'].append("Loading training data...")
    train_ds = load_data(TRAIN_DIR)
    val_ds = load_data(VAL_DIR)

    if not train_ds or not val_ds:
        return "Error: Data not found."

    # Evaluate existing model performance if it exists
    existing_model_accuracy = None
    if os.path.exists(MODEL_SAVE_PATH):
        msg = f"Loading existing model from {MODEL_SAVE_PATH}"
        if status_dict:
            status_dict['training_logs'].append(msg)
        print(msg)
        existing_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        
        # Evaluate current model
        msg = "Evaluating existing model performance..."
        if status_dict:
            status_dict['training_logs'].append(msg)
        print(msg)
        existing_loss, existing_model_accuracy = existing_model.evaluate(val_ds, verbose=0)
        msg = f"Existing model validation accuracy: {existing_model_accuracy:.4f}"
        if status_dict:
            status_dict['training_logs'].append(msg)
        print(msg)
        
        # Create backup
        os.makedirs(BACKUP_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f'wildfire_model_backup_{timestamp}.h5')
        shutil.copy(MODEL_SAVE_PATH, backup_path)
        msg = f"Backup created at: {backup_path}"
        if status_dict:
            status_dict['training_logs'].append(msg)
        print(msg)
        
        msg = "Continuing training on top of the existing model..."
        if status_dict:
            status_dict['training_logs'].append(msg)
        print(msg)
        model = existing_model
        base_model = get_base_model_from_loaded(model)
    else:
        msg = "No existing model found. Building new model from scratch."
        if status_dict:
            status_dict['training_logs'].append(msg)
        print(msg)
        model, base_model = build_model()
    
    # Phase 1 Training (Transfer Learning)
    msg = "\nPhase 1: Transfer Learning"
    if status_dict:
        status_dict['training_logs'].append(msg)
    print(msg)
    
    callbacks = []
    if status_dict:
        callbacks.append(ProgressCallback(status_dict, "Phase 1"))
    
    history_1 = model.fit(
        train_ds,
        epochs=4,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=0  # Suppress default output since we're using callback
    )

    # Phase 2 Training (Fine Tuning)
    msg = "\nPhase 2: Fine Tuning"
    if status_dict:
        status_dict['training_logs'].append(msg)
    print(msg)
    model = compile_for_finetuning(model, base_model)
    
    total_epochs = 4 + 7
    
    callbacks = []
    if status_dict:
        callbacks.append(ProgressCallback(status_dict, "Phase 2"))
    
    history_2 = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=history_1.epoch[-1],
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=0  # Suppress default output since we're using callback
    )

    # Evaluate new model
    msg = "\nEvaluating Retrained Model"
    if status_dict:
        status_dict['training_logs'].append(msg)
    print(msg)
    new_loss, new_model_accuracy = model.evaluate(val_ds, verbose=0)
    msg = f"Retrained model validation accuracy: {new_model_accuracy:.4f}"
    if status_dict:
        status_dict['training_logs'].append(msg)
    print(msg)
    
    # Compare and decide whether to keep new model
    if existing_model_accuracy is not None:
        if new_model_accuracy >= existing_model_accuracy:
            msg = f"Performance improved! ({existing_model_accuracy:.4f} → {new_model_accuracy:.4f})"
            if status_dict:
                status_dict['training_logs'].append(msg)
            print(msg)
            msg = f"Saving new model to: {MODEL_SAVE_PATH}"
            if status_dict:
                status_dict['training_logs'].append(msg)
            print(msg)
            model.save(MODEL_SAVE_PATH)
            return f"Training Completed Successfully. Accuracy improved from {existing_model_accuracy:.2%} to {new_model_accuracy:.2%}"
        else:
            msg = f"Performance decreased ({existing_model_accuracy:.4f} → {new_model_accuracy:.4f})"
            if status_dict:
                status_dict['training_logs'].append(msg)
            print(msg)
            msg = "Keeping existing model. New model discarded."
            if status_dict:
                status_dict['training_logs'].append(msg)
            print(msg)
            return f"Training completed but existing model retained. New model accuracy ({new_model_accuracy:.2%}) did not exceed existing ({existing_model_accuracy:.2%})"
    else:
        # First time training
        model.save(MODEL_SAVE_PATH)
        msg = f"Model saved successfully at: {MODEL_SAVE_PATH}"
        if status_dict:
            status_dict['training_logs'].append(msg)
        print(msg)
        return f"Training Completed Successfully. Model accuracy: {new_model_accuracy:.2%}"

if __name__ == "__main__":
    run_training()