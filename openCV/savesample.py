import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread(r"D:\python\New folder\openCV\Lena.jpg")  # BGR format
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

h, w,_= img.shape
img[h//2,w//2]
print(img)
print("Top-left pixel (gray):", gray[0,0])

# Change a 50x50 patch in top-left to white (255)
img_mod = img.copy()
img_mod[0:50, 0:50] = (255, 255, 255)  # BGR white
plt.imshow(cv2.cvtColor(img_mod, cv2.COLOR_BGR2RGB))
plt.title("Modified: white patch")
plt.axis('off')
plt.show()

cv2.imwrite("modified_sample.jpg", img_mod)
print("Saved modified_sample.jpg")