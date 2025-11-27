import tensorflow as tf

def build_model(input_shape=(128, 128, 3)):
    """
    Constructs a MobileNetV2-based model for transfer learning.
    
    **CUSTOM PRE-TRAINED MODEL:**
    - Uses MobileNetV2 architecture pre-trained on ImageNet (1000 classes)
    - Base model is frozen initially to preserve learned features
    - Custom classification head for wildfire binary classification
    - This serves as the foundation for incremental retraining
    
    The base model is initially frozen to train only the top classification layers.

    Args:
        input_shape (tuple): The shape of input images (height, width, channels).
                             Defaults to (128, 128, 3).

    Returns:
        tuple: A tuple containing:
            - model (tf.keras.Model): The compiled Keras model.
            - base_model (tf.keras.Model): The MobileNetV2 base (for later fine-tuning).
    """

    preprocess_layer = tf.keras.applications.mobilenet_v2.preprocess_input
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model initially
    base_model.trainable = False

    # Create the architecture
    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess_layer(inputs) # Apply MobileNet specific preprocessing
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    # Compile for the initial transfer learning phase
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['binary_accuracy']
    )

    return model, base_model

def compile_for_finetuning(model, base_model, learning_rate=1e-5):
    """
    Unfreezes specific layers of the base model for fine-tuning.

    Args:
        model (tf.keras.Model): The existing trained model.
        base_model (tf.keras.Model): The MobileNetV2 base component.
        learning_rate (float): A lower learning rate for delicate updates. 
                               Defaults to 1e-5.

    Returns:
        tf.keras.Model: The re-compiled model ready for fine-tuning.
    """

    # Unfreeze the base model
    base_model.trainable = True

    # Fine-tune from layer 100 onwards
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Re-compile with low learning rate
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['binary_accuracy']
    )
    
    return model