import cv2
import matplotlib.pyplot as plt

img = cv2.imread('data/friend_ecg.jpg')
plt.figure(figsize=(12,6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Converted ECG Image')
plt.show()