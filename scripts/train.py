"""
train.py

Author: Ayush
Roll Number: 113

This script defines the transfer learning model architecture using EfficientNetB0,
sets up training callbacks, and provides a training pipeline with a verification mode.
"""

import os
import sys
import argparse
import shutil
import tempfile
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# Add project root to python path to resolve imports when running from different directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.data_pipeline import load_and_preprocess_dataset, prepare_dataset

def build_model(num_classes, input_shape=(224, 224, 3)):
    """
    Builds a transfer learning model with an EfficientNetB0 backbone and custom head.
    """
    # Load the pre-trained EfficientNetB0 base model
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze the pre-trained base layers
    base_model.trainable = False
    
    # Build the complete architecture
    inputs = layers.Input(shape=input_shape)
    
    # Pre-trained base (ensure training=False to keep BatchNorm stats frozen if applicable)
    x = base_model(inputs, training=False)
    
    # Custom head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="transfer_learning_efficientnetb0")
    return model

def train_model(data_dir, models_dir, logs_dir, batch_size=32, lr=1e-3, epochs=10):
    """
    Loads data, compiles model, sets up callbacks, and trains the model.
    """
    print(f"Loading data from: {data_dir}")
    train_raw_ds, val_raw_ds = load_and_preprocess_dataset(
        data_dir,
        img_size=(224, 224),
        batch_size=batch_size,
        val_split=0.2,
        seed=42
    )
    
    num_classes = len(train_raw_ds.class_names)
    print(f"Number of classes: {num_classes} ({train_raw_ds.class_names})")
    
    # Prepare datasets for training (cache, shuffle, augment, prefetch)
    train_ds = prepare_dataset(train_raw_ds, augment=True, batch_size=batch_size, shuffle_buffer=1000)
    val_ds = prepare_dataset(val_raw_ds, augment=False, batch_size=batch_size, shuffle_buffer=0)
    
    # Build model
    print("Building transfer learning model...")
    model = build_model(num_classes=num_classes)
    model.summary()
    
    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Callbacks
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(models_dir, "best_model.keras")
    
    my_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=logs_dir,
            histogram_freq=1
        )
    ]
    
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=my_callbacks
    )
    
    history_path = os.path.join(models_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f)
    print(f"Training completed. Best model saved to: {checkpoint_path}")
    print(f"History saved to: {history_path}")
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transfer Learning Image Classification Model")
    parser.add_argument("--data_dir", type=str, default="", help="Path to raw image directory")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Directory for TensorBoard logs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--verify", action="store_true", help="Run a quick verification test with synthetic data")
    
    args = parser.parse_args()
    
    # If no data_dir is provided, or if the verify flag is set, run verification mode
    if args.verify or not args.data_dir:
        print("=" * 60)
        print("Running train.py standalone mock verification...")
        print("=" * 60)
        
        # Create a temporary directory for mock images
        temp_dir = tempfile.mkdtemp()
        temp_models_dir = tempfile.mkdtemp()
        temp_logs_dir = tempfile.mkdtemp()
        
        classes = ["cat", "dog", "bird"]
        num_images_per_class = 10
        img_height, img_width = 224, 224
        
        try:
            print(f"Generating synthetic images in: {temp_dir}")
            for class_name in classes:
                class_path = os.path.join(temp_dir, class_name)
                os.makedirs(class_path, exist_ok=True)
                for i in range(num_images_per_class):
                    # Random RGB images
                    img_data = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
                    img_tensor = tf.convert_to_tensor(img_data, dtype=tf.uint8)
                    img_encoded = tf.io.encode_jpeg(img_tensor)
                    
                    img_file_path = os.path.join(class_path, f"mock_{i}.jpg")
                    tf.io.write_file(img_file_path, img_encoded)
                    
            print(f"Generated {len(classes) * num_images_per_class} mock images.")
            print("-" * 50)
            
            # Run model training for 2 epochs
            print("Running mock training for 2 epochs...")
            train_model(
                data_dir=temp_dir,
                models_dir=temp_models_dir,
                logs_dir=temp_logs_dir,
                batch_size=4,
                lr=1e-3,
                epochs=2
            )
            
            print("-" * 50)
            print("Model architecture verification COMPLETED successfully.")
            
        finally:
            print("Cleaning up temporary directories...")
            shutil.rmtree(temp_dir)
            shutil.rmtree(temp_models_dir)
            shutil.rmtree(temp_logs_dir)
            print("Cleanup complete.")
            print("=" * 60)
    else:
        # Normal training flow
        train_model(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            logs_dir=args.logs_dir,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs
        )
