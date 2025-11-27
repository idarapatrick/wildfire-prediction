import tensorflow as tf
import os

# Configuration
BATCH_SIZE = 32
IMG_SIZE = (128, 128) # MobileNetV2 input

def load_data(data_dir):
    """
    Loads images from a directory and creates a TensorFlow dataset.
    
    **PREPROCESSING STEPS APPLIED:**
    1. Automatic resizing to 128x128 pixels (MobileNetV2 input size)
    2. Pixel value normalization: dividing by 255 to scale to [0, 1] range
    3. Binary labeling: 0 = nowildfire, 1 = wildfire (alphabetical order)
    4. Data batching: Groups images into batches of 32 for efficient training
    5. Performance optimization: Caching and prefetching for faster loading

    Args:
        data_dir (str): The path to the directory containing image subfolders.
                        (e.g., 'data/train' containing 'wildfire' and 'nowildfire')

    Returns:
        tf.data.Dataset: A PrefetchedDataset object optimized for training.
                         Returns None if the directory does not exist.
    """

    # Check if directory exists to avoid crashing
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return None

    # Load data using the modern Keras utility
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary' # Important for binary_crossentropy
    )

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return dataset