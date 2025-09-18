import numpy as np
import cv2
import matplotlib.pyplot as plt
# Create a 5x5 grayscale image (values 0..255, dtype=uint8)
g = np.array([
    [  0,  50, 100, 150, 200],
    [  5,  55, 105, 155, 205],
    [10,  60, 110, 160, 210],
    [15,  65, 115, 165, 215],
    [20,  70, 120, 170, 220]
], dtype=np.uint8)

cv2.imshow("image", g)

print("shape:", g.shape)        # -> (5,5)
print("dtype:", g.dtype)        # -> uint8
print("pixel (row=0,col=0):", g[0,0])
plt.figure(figsize=(3,3))
plt.title("5x5 Grayscale")
plt.imshow(g)
plt.imshow(g, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.axis('on')
plt.show()