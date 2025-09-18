import numpy as np
import matplotlib.pyplot as plt
# Create a 5x5 RGB image by stacking channels
r = np.zeros((5,5), dtype=np.uint8)   # red channel all 255
g = np.zeros((5,5), dtype=np.uint8) + 255  # green 0
b = np.zeros((5,5), dtype=np.uint8)        # blue 0

rgb = np.stack([r, g, b], axis=-1)  # shape (5,5,3)
print("shape:", rgb.shape)  # -> (5,5,3)
print("pixel at (0,0):", rgb[0,0])  # -> [255, 0, 0]  (R,G,B)
plt.figure(figsize=(2,2))
plt.title("5x5 Red Image (RGB)")
plt.imshow(rgb)
plt.axis('off')
plt.show()