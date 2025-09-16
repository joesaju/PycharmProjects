import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# -----------------------------
# 1. Load & Preprocess Data
# -----------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values (0–255 → 0–1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten images (32x32x3 = 3072 features)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# -----------------------------
# 2. Build ANN Model
# -----------------------------
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(3072,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# 3. Train Model
# -----------------------------
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=128,
                    validation_split=0.2,
                    verbose=1)

# -----------------------------
# 4. Evaluate on Test Data
# -----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# -----------------------------
# 5. Plot Training History
# -----------------------------
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

# -----------------------------
# 6. User Interaction
# -----------------------------
def predict_image(index):
    img = x_test[index].reshape(32, 32, 3)  # reshape back for display
    true_label = np.argmax(y_test[index])
    pred_label = np.argmax(model.predict(x_test[index].reshape(1, -1)))

    plt.imshow(img)
    plt.axis('off')
    plt.title(f"True: {class_names[true_label]} | Predicted: {class_names[pred_label]}")
    plt.show()

# Example: Ask user for an index
try:
    idx = int(input("Enter a test image index (0–9999): "))
    predict_image(idx)
except:
    print("Invalid input. Please enter a number between 0 and 9999.")
