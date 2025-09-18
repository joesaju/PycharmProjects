import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)    # (10000, 28, 28, 1)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))  # Train just 1 epoch for demo

model.save("digit_cnn.h5")
print("✅ Model trained and saved as digit_cnn.h5")

import cv2
img = cv2.imread(r"D:\python\New folder\openCV\Lena.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img.shape
print(img)
cv2.imshow("Lena image")
cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np
# Create a 5x5 grayscale image (values 0..255, dtype=uint8)
g = np.array([
    [  0,  50, 100, 150, 200],
    [  5,  55, 105, 155, 205],
    [10,  60, 110, 160, 210],
    [15,  65, 115, 165, 215],
    [20,  70, 120, 170, 220]
], dtype=np.uint8)

print("shape:", g.shape)        # -> (5,5)
print("dtype:", g.dtype)        # -> uint8
print("pixel (row=0,col=0):",g[0,0])




