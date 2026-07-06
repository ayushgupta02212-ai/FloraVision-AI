"""
convert_tflite.py

Author: Ayush Kr. Gupta
Roll Number: 113

This script handles loading a trained Keras model and converting it to
the optimized TensorFlow Lite (.tflite) format suitable for edge deployment.
"""

import os
import sys
import argparse
import shutil
import tempfile
import numpy as np
import tensorflow as tf

# Add project root to python path to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def convert_to_tflite(model_path, tflite_output_path):
    """
    Loads Keras model, converts it to TensorFlow Lite, and writes the output.
    """
    print(f"Loading Keras model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
        
    model = tf.keras.models.load_model(model_path)
    
    print("Initializing TensorFlow Lite converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Perform standard conversion
    print("Converting model (this may take a few moments)...")
    tflite_model = converter.convert()
    
    # Save the output file
    os.makedirs(os.path.dirname(tflite_output_path), exist_ok=True)
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"Successfully saved TensorFlow Lite model to: {tflite_output_path}")
    
    # 8. Verify the converted model can load in the interpreter
    print("Verifying TFLite model by loading into Interpreter...")
    interpreter = tf.lite.Interpreter(model_path=tflite_output_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors information
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"TFLite Input Details: {input_details}")
    print(f"TFLite Output Details: {output_details}")
    print("TFLite Model verified successfully.")
    
    return tflite_output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Keras model to TensorFlow Lite format")
    parser.add_argument("--model_path", type=str, default="models/best_model.keras", help="Path to input Keras model")
    parser.add_argument("--tflite_output_path", type=str, default="models/model.tflite", help="Path to output TFLite file")
    parser.add_argument("--verify", action="store_true", help="Run a quick verification test with synthetic model")
    
    args = parser.parse_args()
    
    if args.verify or not os.path.exists(args.model_path):
        # Verification or fallback verification if no model exists yet
        print("=" * 60)
        print("Running convert_tflite.py standalone verification...")
        print("=" * 60)
        
        # Setup temporary directories
        temp_models_dir = tempfile.mkdtemp()
        temp_tflite_path = os.path.join(temp_models_dir, "model_verify.tflite")
        
        try:
            # Construct a very simple Keras model for testing (avoids full training)
            print("Creating a simple Keras model for quick conversion test...")
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
            outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            temp_model_path = os.path.join(temp_models_dir, "temp_model.keras")
            model.save(temp_model_path)
            print(f"Temporary model saved to: {temp_model_path}")
            print("-" * 50)
            
            # Convert
            convert_to_tflite(
                model_path=temp_model_path,
                tflite_output_path=temp_tflite_path
            )
            
            # Check file exists
            tflite_exists = os.path.exists(temp_tflite_path)
            print("-" * 50)
            print(f"Deliverable TFLite file exists: {tflite_exists}")
            
            if tflite_exists:
                print("TFLite model conversion verified successfully.")
            else:
                raise RuntimeError("TFLite model was not successfully created.")
                
        finally:
            print("Cleaning up temporary directories...")
            shutil.rmtree(temp_models_dir)
            print("Cleanup complete.")
            print("=" * 60)
    else:
        convert_to_tflite(
            model_path=args.model_path,
            tflite_output_path=args.tflite_output_path
        )
