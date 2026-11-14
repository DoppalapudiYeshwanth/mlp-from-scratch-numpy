import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="NumPy MLP Digit Classifier", layout="centered")
st.title("MLP from Scratch (NumPy)")
st.caption("Architecture: 784 → 128 → 64 → 10 | Validation Accuracy: 97.68%")

@st.cache_resource
def load_weights():
    data = np.load("weights.npz")
    return data["W0"], data["b0"], data["W1"], data["b1"], data["W2"], data["b2"]

W0, b0, W1, b1, W2, b2 = load_weights()

def predict(x):
    h1 = np.maximum(0, x @ W0 + b0)
    h2 = np.maximum(0, h1 @ W1 + b1)
    logits = h2 @ W2 + b2
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return (exp / np.sum(exp, axis=1, keepdims=True))[0]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Digit (0–9)")
    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        return_image_data=True,
        key="canvas",
    )
    uploaded = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])

with col2:
    st.subheader("Prediction")
    x = None

    if uploaded is not None:
        img = Image.open(uploaded).convert("L")
        # Invert if background is bright (black digit on white paper)
        if np.mean(img) > 127:
            img = ImageOps.invert(img)
        img = img.resize((28, 28), Image.BILINEAR)
        x = (np.array(img, dtype=np.float32) / 255.0).reshape(1, 784)
        st.image(img, caption="28×28 Model Input", width=84)
    elif canvas.image_data is not None and np.any(canvas.image_data[:, :, 0] > 0):
        img = Image.fromarray(canvas.image_data[:, :, 0].astype(np.uint8)).resize((28, 28), Image.BILINEAR)
        x = (np.array(img, dtype=np.float32) / 255.0).reshape(1, 784)

    if x is not None:
        probs = predict(x)
        pred = int(np.argmax(probs))
        st.metric(label="Predicted Digit", value=str(pred), delta=f"{probs[pred] * 100:.2f}% confidence")
        st.bar_chart({str(i): float(probs[i]) for i in range(10)}, height=200)
    else:
        st.info("Draw a digit or upload an image to run inference.")