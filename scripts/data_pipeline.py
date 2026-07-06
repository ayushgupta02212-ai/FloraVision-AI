"""
data_pipeline.py

Author: Ayush
Roll Number: 113

This module handles loading, splitting, augmenting, and performance-tuning
image datasets for training deep learning models using tf.data.Dataset.
"""

import os
import shutil
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def get_data_augmentation_pipeline():
    """
    Creates a Keras Sequential model for data augmentation.
    Includes: Horizontal/Vertical flip, Rotation, Zoom, and Brightness adjustments.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2)
    ], name="data_augmentation")

def load_and_preprocess_dataset(data_dir, img_size=(224, 224), batch_size=32, val_split=0.2, seed=42):
    """
    Loads raw images from data_dir and creates training and validation tf.data.Datasets.
    Returns:
        train_ds: The training dataset (unbatched or batched by image_dataset_from_directory)
        val_ds: The validation dataset (unbatched or batched by image_dataset_from_directory)
    """
    # Note: image_dataset_from_directory returns a batched dataset.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )
    
    return train_ds, val_ds

def prepare_dataset(ds, augment=False, batch_size=32, shuffle_buffer=1000):
    """
    Optimizes dataset pipeline using caching, shuffling, optional augmentation, and prefetching.
    """
    # 1. Cache dataset in memory to avoid repeated disk reads
    ds = ds.cache()
    
    # 2. Shuffle dataset (only if training, since validation split doesn't need training shuffling)
    # Note: image_dataset_from_directory already shuffles on loading, but after caching, 
    # we want to reshuffle each epoch.
    if shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
        
    # 3. Apply augmentation dynamically after caching
    if augment:
        augmentation_pipeline = get_data_augmentation_pipeline()
        ds = ds.map(
            lambda x, y: (augmentation_pipeline(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
    # 4. Prefetch to overlap CPU data preprocessing with GPU training
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

if __name__ == "__main__":
    print("=" * 60)
    print("Running data_pipeline.py standalone verification...")
    print("=" * 60)
    
    # Create a temporary directory for mock images
    temp_dir = tempfile.mkdtemp()
    classes = ["cat", "dog", "bird"]
    num_images_per_class = 15
    img_height, img_width = 224, 224
    
    try:
        print(f"Creating synthetic/dummy dataset in temporary directory: {temp_dir}")
        for class_name in classes:
            class_path = os.path.join(temp_dir, class_name)
            os.makedirs(class_path, exist_ok=True)
            
            # Generate random color patterns and save as JPEGs using tf.io
            for i in range(num_images_per_class):
                # Create a synthetic image (random RGB values between 0 and 255)
                img_data = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
                
                # Encode and save using tensorflow
                img_tensor = tf.convert_to_tensor(img_data, dtype=tf.uint8)
                img_encoded = tf.io.encode_jpeg(img_tensor)
                
                img_file_path = os.path.join(class_path, f"mock_{i}.jpg")
                tf.io.write_file(img_file_path, img_encoded)
                
        print("Synthetic dataset successfully generated.")
        print(f"Total images created: {len(classes) * num_images_per_class}")
        print("-" * 50)
        
        # Load datasets
        batch_size = 4
        print("Loading datasets using load_and_preprocess_dataset...")
        train_raw_ds, val_raw_ds = load_and_preprocess_dataset(
            temp_dir,
            img_size=(img_height, img_width),
            batch_size=batch_size,
            val_split=0.2,
            seed=42
        )
        
        class_names = train_raw_ds.class_names
        print(f"Detected class names: {class_names}")
        
        # Prepare datasets
        print("Preparing datasets (caching, augmentation, prefetching)...")
        train_ds = prepare_dataset(train_raw_ds, augment=True, batch_size=batch_size, shuffle_buffer=100)
        val_ds = prepare_dataset(val_raw_ds, augment=False, batch_size=batch_size, shuffle_buffer=0)
        
        # Verify training batches
        print("\nVerifying training dataset batches (with augmentation):")
        for i, (images, labels) in enumerate(train_ds.take(2)):
            print(f"Batch {i + 1}:")
            print(f"  Images shape: {images.shape} (Expected: ({batch_size}, {img_height}, {img_width}, 3))")
            print(f"  Labels shape: {labels.shape} (Expected: ({batch_size}, {len(classes)}))")
            
            # Print basic stats of the batch to verify data range and types
            min_val = tf.reduce_min(images).numpy()
            max_val = tf.reduce_max(images).numpy()
            mean_val = tf.reduce_mean(images).numpy()
            print(f"  Pixel stats -> Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f}")
            
            # Check labels representation (should be one-hot encoded since label_mode='categorical')
            print(f"  One-hot labels:\n{labels.numpy()}")
            
        # Verify validation batches
        print("\nVerifying validation dataset batches (without augmentation):")
        for i, (images, labels) in enumerate(val_ds.take(1)):
            print(f"Validation Batch:")
            print(f"  Images shape: {images.shape} (Expected: (<= {batch_size}, {img_height}, {img_width}, 3))")
            print(f"  Labels shape: {labels.shape}")
            
        print("\nData pipeline verification COMPLETED successfully.")
        
    finally:
        # Clean up temporary directory
        print("-" * 50)
        print("Cleaning up temporary directory...")
        shutil.rmtree(temp_dir)
        print("Cleanup complete.")
        print("=" * 60)
