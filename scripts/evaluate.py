"""
evaluate.py

Author: Ayush Kr. Gupta
Roll Number: 113

This module handles loading a trained classification model, evaluating its performance
on a validation dataset, and saving metrics plots (loss/accuracy curves and confusion matrix).
"""

import os
import sys
import json
import argparse
import shutil
import tempfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Add project root to python path to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.data_pipeline import load_and_preprocess_dataset, prepare_dataset

def plot_history(history_dict, output_dir):
    """
    Plots training and validation loss/accuracy curves.
    """
    epochs = range(1, len(history_dict['loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(epochs, history_dict['loss'], 'bo-', label='Training Loss')
    if 'val_loss' in history_dict:
        ax1.plot(epochs, history_dict['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    if 'accuracy' in history_dict:
        ax2.plot(epochs, history_dict['accuracy'], 'bo-', label='Training Accuracy')
        if 'val_accuracy' in history_dict:
            ax2.plot(epochs, history_dict['val_accuracy'], 'ro-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved training curves to: {plot_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """
    Generates and saves a confusion matrix plot.
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names, 
        yticklabels=class_names,
        title='Confusion Matrix',
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
            
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

def evaluate_model(model_path, history_path, data_dir, output_dir):
    """
    Loads model and history, runs validation evaluation, and saves metric visualizations.
    """
    # 1. Load the model
    print(f"Loading trained model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # 2. Plot history curves
    print(f"Loading training history from: {history_path}")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history_dict = json.load(f)
        plot_history(history_dict, output_dir)
    else:
        print(f"WARNING: History file not found at {history_path}. Skipping training curves.")
        
    # 3. Load validation dataset
    print(f"Loading validation dataset from: {data_dir}")
    _, val_raw_ds = load_and_preprocess_dataset(
        data_dir,
        img_size=(224, 224),
        batch_size=16,
        val_split=0.2,
        seed=42
    )
    class_names = val_raw_ds.class_names
    
    # Prepare val dataset (no shuffle, no augment)
    val_ds = prepare_dataset(val_raw_ds, augment=False, batch_size=16, shuffle_buffer=0)
    
    # 4. Run evaluation
    print("Evaluating model performance on validation set...")
    loss, accuracy = model.evaluate(val_ds)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # 5. Predict to compute confusion matrix
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    plot_confusion_matrix(y_true, y_pred, class_names, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Classification Model & Generate Visualizations")
    parser.add_argument("--model_path", type=str, default="models/best_model.keras", help="Path to best saved model")
    parser.add_argument("--history_path", type=str, default="models/history.json", help="Path to training history JSON")
    parser.add_argument("--data_dir", type=str, default="", help="Path to evaluation dataset")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save generated charts")
    parser.add_argument("--verify", action="store_true", help="Run a quick verification test with synthetic data")
    
    args = parser.parse_args()
    
    # If no data directory is provided or verify flag is active, execute verification flow
    if args.verify or not args.data_dir:
        print("=" * 60)
        print("Running evaluate.py standalone verification...")
        print("=" * 60)
        
        # Import train_model dynamically to avoid circular dependencies
        from scripts.train import train_model
        
        # Setup temporary testing structure
        temp_dir = tempfile.mkdtemp()
        temp_models_dir = tempfile.mkdtemp()
        temp_logs_dir = tempfile.mkdtemp()
        temp_output_dir = tempfile.mkdtemp()
        
        classes = ["cat", "dog", "bird"]
        num_images_per_class = 10
        img_height, img_width = 224, 224
        
        try:
            print("Generating temporary synthetic images...")
            for class_name in classes:
                class_path = os.path.join(temp_dir, class_name)
                os.makedirs(class_path, exist_ok=True)
                for i in range(num_images_per_class):
                    img_data = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
                    img_tensor = tf.convert_to_tensor(img_data, dtype=tf.uint8)
                    img_encoded = tf.io.encode_jpeg(img_tensor)
                    img_file_path = os.path.join(class_path, f"mock_{i}.jpg")
                    tf.io.write_file(img_file_path, img_encoded)
            
            # Train a quick model to create a valid keras file and history
            print("Training mock model...")
            train_model(
                data_dir=temp_dir,
                models_dir=temp_models_dir,
                logs_dir=temp_logs_dir,
                batch_size=4,
                lr=1e-3,
                epochs=2
            )
            
            # Run evaluation
            model_file = os.path.join(temp_models_dir, "best_model.keras")
            history_file = os.path.join(temp_models_dir, "history.json")
            
            print("\nRunning evaluate_model on the mock model...")
            evaluate_model(
                model_path=model_file,
                history_path=history_file,
                data_dir=temp_dir,
                output_dir=temp_output_dir
            )
            
            # Verify outputs
            curves_exists = os.path.exists(os.path.join(temp_output_dir, 'training_curves.png'))
            matrix_exists = os.path.exists(os.path.join(temp_output_dir, 'confusion_matrix.png'))
            
            print("-" * 50)
            print(f"Deliverable 'training_curves.png' exists: {curves_exists}")
            print(f"Deliverable 'confusion_matrix.png' exists: {matrix_exists}")
            
            if curves_exists and matrix_exists:
                print("Evaluation metrics generation verified successfully.")
            else:
                raise RuntimeError("Evaluation files were not successfully created.")
            
        finally:
            print("Cleaning up temporary directories...")
            shutil.rmtree(temp_dir)
            shutil.rmtree(temp_models_dir)
            shutil.rmtree(temp_logs_dir)
            shutil.rmtree(temp_output_dir)
            print("Cleanup complete.")
            print("=" * 60)
            
    else:
        evaluate_model(
            model_path=args.model_path,
            history_path=args.history_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
