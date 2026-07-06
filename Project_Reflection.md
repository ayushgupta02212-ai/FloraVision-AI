# Project Reflection: My Experience with FloraVision AI

## What Went Well
Developing FloraVision AI was a rewarding experience that allowed me to bridge the gap between machine learning theory and practical application.
- **Model Performance & Accuracy**: Leveraging EfficientNetB0 enabled the model to achieve high classification accuracy relatively quickly, showcasing the power of transfer learning.
- **UI Design**: The Streamlit interface is intuitive, responsive, and visually appealing, allowing users to easily upload images and view model predictions alongside their Grad-CAM visualizations.

## Hurdles
Like any real-world engineering project, the path to completion had its obstacles:
- **Cloud Deployment Frustrations**: The biggest bottleneck was deploying the application to Streamlit Cloud. Encountering memory exhaustion limits and seeing the container crash repeatedly because of TensorFlow's heavy footprint was frustrating.
- **The "Aha!" Moment**: The turning point came when I realized that we didn't need the GPU-enabled TensorFlow libraries for a CPU-only cloud environment. Switching to a local deployment-inspired optimization strategy—using `tensorflow-cpu` and fine-tuning dependency files like `requirements.txt` and `packages.txt`—resolved the memory leaks and allowed the deployment to run smoothly.

## Lessons Learned
This project provided invaluable hands-on lessons:
- **Version Control (Git)**: Using Git effectively to track code updates, manage commits, merge changes, and sync with the remote repository taught me the importance of a clean version history.
- **Dependency Management**: I learned that `requirements.txt` must be curated carefully, distinguishing between development environments and lightweight production environments.
- **Production Constraints**: Development on local workstations with ample RAM and GPU acceleration often hides bottlenecks that become obvious only under strict cloud resource constraints.

## Future Goals
I plan to carry these lessons into my next projects at ByteWings:
1. **Edge-first Design**: In my next project, I will design model architectures with mobile and edge constraints in mind from day one, exploring quantization and TFLite conversion much earlier in the cycle.
2. **Robust CI/CD & Deployment Testing**: I will incorporate lightweight testing and dependency checks early in the development process to catch memory and library compatibility issues before they reach production.
