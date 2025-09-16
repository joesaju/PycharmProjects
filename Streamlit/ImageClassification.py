import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import os
from PIL import Image

# -----------------------------
# CIFAR-10 Class Labels
# -----------------------------
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# -----------------------------
# Load & Preprocess Data
# -----------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# -----------------------------
# Build Model Function
# -----------------------------
def build_model():
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(3072,)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# -----------------------------
# Load or Train Model
# -----------------------------
MODEL_PATH = "cifar10_ann.h5"

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = build_model()
    history = model.fit(x_train_flat, y_train_cat,
                        epochs=20,
                        batch_size=128,
                        validation_split=0.2,
                        verbose=1)
    model.save(MODEL_PATH)

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("🖼️ CIFAR-10 Image Classification using ANN")
st.write("This app uses an Artificial Neural Network to classify images from the CIFAR-10 dataset.")

# Evaluate model
test_loss, test_acc = model.evaluate(x_test_flat, y_test_cat, verbose=0)
st.write(f"✅ Model Test Accuracy: **{test_acc * 100:.2f}%**")

# -----------------------------
# User Input: Select Test Image
# -----------------------------
st.subheader("🔍 Test Image Prediction")
idx = st.slider("Select an image index (0–9999)", 0, 9999, 0)

img = x_test[idx]
true_label = int(y_test[idx])
pred_label = np.argmax(model.predict(x_test_flat[idx].reshape(1, -1)))

st.image(img, caption=f"True: {class_names[true_label]} | Predicted: {class_names[pred_label]}", use_column_width=True)

# -----------------------------
# User Upload Option
# -----------------------------
st.subheader("📤 Upload Your Own Image (Optional)")
uploaded_file = st.file_uploader("Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess uploaded image
    img = Image.open(uploaded_file).resize((32, 32))
    img_array = np.array(img).astype("float32") / 255.0
    if img_array.shape == (32, 32, 3):  # Ensure RGB
        flat_img = img_array.reshape(1, -1)
        pred_label = np.argmax(model.predict(flat_img))
        st.image(img, caption=f"Predicted: {class_names[pred_label]}", use_column_width=True)
    else:
        st.error("Uploaded image must be RGB with 3 channels (not grayscale).")
