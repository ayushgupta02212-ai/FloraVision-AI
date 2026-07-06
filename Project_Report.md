# Project Report: FloraVision AI

## Overview
FloraVision AI is an intelligent web application designed for automated flower species identification and classification. Developed as a pair-programming project for my internship at ByteWings, the app allows users to upload images of flowers, instantly identifies their species using a deep learning classifier, and uses advanced explainable AI to highlight the parts of the image that influenced the classification decision.

## Methodology
The development of FloraVision AI followed a structured deep learning pipeline:
1. **Data Pipeline & Training**: The classifier was trained on a dataset comprising thousands of images across multiple distinct flower classes (e.g., daisy, dandelion, rose, sunflower, tulip). Data augmentation (rotation, zooming, horizontal flips) was applied to prevent overfitting.
2. **Transfer Learning with EfficientNetB0**: Rather than training a convolutional neural network from scratch, I leveraged transfer learning using the pretrained **EfficientNetB0** architecture. EfficientNetB0 provides excellent feature extraction capabilities with a relatively small parameter footprint, making it ideal for deployment.
3. **Interpretability with Grad-CAM**: To foster trust and provide explainability, the app implements Gradient-weighted Class Activation Mapping (Grad-CAM). Grad-CAM visualizes the gradients flowing into the final convolutional layer of the network to produce a coarse localization map, highlighting the most discriminative regions of the image (like petals or stamen) that the model focused on when making its prediction.

## Challenges
During development, several key technical challenges were encountered and resolved:
1. **Dataset Scale & File Handling**: Processing and loading thousands of high-resolution images proved to be memory-intensive. I optimized the input pipeline using TensorFlow's `tf.data` API to stream images from disk efficiently.
2. **Training Complexity**: Managing hyperparameter tuning, learning rate scheduling, and validation splits to achieve high classification accuracy while avoiding overfitting required extensive iteration.
3. **Cloud Deployment Memory Constraints**: Deploying the model to Streamlit Cloud introduced severe RAM constraints. The standard TensorFlow package consumes a significant amount of memory, causing the container to crash during startup or model loading (`Error installing requirements` / container termination). I resolved this by:
   - Replacing the full `tensorflow` library with `tensorflow-cpu>=2.15.0` in the `requirements.txt` file to drastically reduce the memory footprint.
   - Adding `libhdf5-dev` to `packages.txt` to help Streamlit handle HDF5/Keras model dependencies.

## Conclusion
FloraVision AI successfully demonstrates how deep transfer learning can be paired with modern web frameworks like Streamlit to deliver a responsive, interpretable AI utility. The classifier achieves excellent accuracy on the target flower classes. Looking forward, further edge optimization using **TensorFlow Lite (TFLite)** is highly beneficial, as converting the model to TFLite will enable deployment on resource-constrained devices, lowering latency and removing server-side memory limitations entirely.
