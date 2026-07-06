"""
grad_cam.py

Author: Ayush Kr. Gupta
Roll Number: 113

This module implements the Grad-CAM (Gradient-weighted Class Activation Mapping)
visualization algorithm for Keras 3 and EfficientNetB0 models. It generates heatmaps
to explain model prediction decisions by highlighting critical regions in input images.
"""

import os
import sys
import shutil
import argparse
import tempfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add project root to python path to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.data_pipeline import load_and_preprocess_dataset, prepare_dataset

def generate_gradcam(model_path, img_path, output_path, last_conv_layer_name="top_activation"):
    """
    Computes the Grad-CAM heatmap and overlays it on the input image.
    """
    # 1. Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # 2. Extract nested pre-trained model and head layers
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or layer.name == "efficientnetb0":
            base_model = layer
            break
            
    if base_model is None:
        raise ValueError("EfficientNetB0 base model layer not found in loaded model.")
        
    # Extract head layers (any layer after base model)
    head_layers = []
    base_model_found = False
    for layer in model.layers:
        if layer == base_model:
            base_model_found = True
            continue
        if base_model_found:
            head_layers.append(layer)
            
    # 3. Create a dual-output model for the base network
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    base_grad_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[last_conv_layer.output, base_model.output]
    )
    
    # 4. Load and preprocess the target image for prediction
    img_size = (224, 224)
    img_for_pred = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img_for_pred)
    img_array = np.expand_dims(img_array, axis=0) # Shape: (1, 224, 224, 3)
    
    # 5. Compute class gradients w.r.t. the last convolutional layer
    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = base_grad_model(img_array)
        
        # Pass base model output through custom head layers
        x = base_outputs
        for layer in head_layers:
            # Execute in inference mode (dropout inactive)
            x = layer(x, training=False)
        preds = x
        
        # Focus on the winning/predicted class
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        
    # Compute gradients of class score w.r.t the activation of conv layer
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Pool gradients over spatial dimensions (Global Average Pooling on grads)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Compute the weighted combination of the feature map channels
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Apply ReLU activation to emphasize features with positive influence, normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()
    
    # 6. Overlay the heatmap onto the original input image
    # Load original image in its native dimensions
    orig_img = tf.keras.utils.load_img(img_path)
    orig_img = tf.keras.utils.img_to_array(orig_img)
    
    # Resize the heatmap to match original image dimensions
    heatmap_resized = tf.image.resize(
        heatmap[..., tf.newaxis],
        (orig_img.shape[0], orig_img.shape[1])
    ).numpy().squeeze()
    
    # Rescale heatmap values to 0-255 integers
    heatmap_resized = np.uint8(255 * heatmap_resized)
    
    # Retrieve Jet colormap colors
    colormap = plt.colormaps["jet"]
    colormap_colors = colormap(np.arange(256))[:, :3]
    colorized_heatmap = colormap_colors[heatmap_resized]
    colorized_heatmap = np.uint8(255 * colorized_heatmap)
    
    # Superimpose heatmap and original image
    superimposed_img = colorized_heatmap * 0.4 + orig_img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    # 7. Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tf.keras.utils.save_img(output_path, superimposed_img)
    print(f"Successfully saved Grad-CAM overlay image to: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for image classification")
    parser.add_argument("--model_path", type=str, default="models/best_model.keras", help="Path to Keras model file")
    parser.add_argument("--img_path", type=str, required=False, help="Path to input image file")
    parser.add_argument("--output_path", type=str, default="data/processed/grad_cam_overlay.png", help="Path to output image")
    parser.add_argument("--verify", action="store_true", help="Run a quick verification test with synthetic data")
    
    args = parser.parse_args()
    
    # Verify or run if no image path provided
    if args.verify or not args.img_path:
        print("=" * 60)
        print("Running grad_cam.py standalone verification...")
        print("=" * 60)
        
        # Import train_model dynamically to avoid circular dependencies
        from scripts.train import train_model
        
        # Setup temporary directories
        temp_dir = tempfile.mkdtemp()
        temp_models_dir = tempfile.mkdtemp()
        temp_logs_dir = tempfile.mkdtemp()
        temp_output_path = os.path.join(tempfile.mkdtemp(), "grad_cam_verify.png")
        
        classes = ["cat", "dog", "bird"]
        num_images_per_class = 10
        img_height, img_width = 224, 224
        
        try:
            print("Generating temporary mock image and dataset...")
            test_img_path = None
            for class_name in classes:
                class_path = os.path.join(temp_dir, class_name)
                os.makedirs(class_path, exist_ok=True)
                for i in range(num_images_per_class):
                    img_data = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
                    img_tensor = tf.convert_to_tensor(img_data, dtype=tf.uint8)
                    img_encoded = tf.io.encode_jpeg(img_tensor)
                    
                    img_file_path = os.path.join(class_path, f"mock_{i}.jpg")
                    tf.io.write_file(img_file_path, img_encoded)
                    
                    if test_img_path is None:
                        test_img_path = img_file_path
            
            # Train a quick model to create a valid Keras model file
            print("Training mock model...")
            train_model(
                data_dir=temp_dir,
                models_dir=temp_models_dir,
                logs_dir=temp_logs_dir,
                batch_size=4,
                lr=1e-3,
                epochs=1
            )
            
            model_file = os.path.join(temp_models_dir, "best_model.keras")
            
            # Run Grad-CAM
            print("\nRunning generate_gradcam on the mock image...")
            generate_gradcam(
                model_path=model_file,
                img_path=test_img_path,
                output_path=temp_output_path
            )
            
            # Check outputs
            output_exists = os.path.exists(temp_output_path)
            print("-" * 50)
            print(f"Deliverable Grad-CAM heatmap exists: {output_exists}")
            
            if output_exists:
                print("Grad-CAM generation verified successfully.")
            else:
                raise RuntimeError("Grad-CAM overlay file was not successfully created.")
                
        finally:
            print("Cleaning up temporary directories...")
            shutil.rmtree(temp_dir)
            shutil.rmtree(temp_models_dir)
            shutil.rmtree(temp_logs_dir)
            if os.path.exists(os.path.dirname(temp_output_path)):
                shutil.rmtree(os.path.dirname(temp_output_path))
            print("Cleanup complete.")
            print("=" * 60)
    else:
        generate_gradcam(
            model_path=args.model_path,
            img_path=args.img_path,
            output_path=args.output_path
        )
