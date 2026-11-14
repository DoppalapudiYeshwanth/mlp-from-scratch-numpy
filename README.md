# Multi-Layer Perceptron (MLP) from Scratch in NumPy

A modular, pure-NumPy neural network framework built from scratch without PyTorch, TensorFlow, or JAX—featuring custom forward/backward passes, analytical gradient verification, MNIST training, and an interactive Streamlit web app.

## Architecture & Results
- **Network:** 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)
- **Optimizer & Loss:** Adam (lr=0.001) + Fused Softmax Cross-Entropy
- **Gradient Check:** Verified via central finite differences (max relative error < 1e-6)
- **Validation Accuracy:** **97.68%** on MNIST (10 epochs)

## Project Structure
- `activations.py` - `ReLU`, `Sigmoid`, and numerically stable `Softmax`
- `layers.py` - `Dense` layer with He/Xavier initialization
- `losses.py` - `MSELoss`, `CrossEntropyLoss`, and fused `SoftmaxCrossEntropy`
- `optimizers.py` - Stateful `SGD` (with momentum) and `Adam`
- `model.py` - `Sequential` container
- `grad_check.py` - Numerical vs. analytical gradient verification
- `train.py` - Mini-batch training pipeline saving `weights.npz`
- `app.py` - Streamlit digit drawing & image upload inference UI

## Quickstart
1. Install dependencies: `pip install -r requirements.txt`
2. Verify backpropagation math: `python grad_check.py`
3. Train model and generate weights: `python train.py`
4. Launch interactive classifier: `streamlit run app.py`
