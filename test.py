
import cv2
import matplotlib.pyplot as plt
lena_BGR = cv2.imread("dataset/mini/test/label/35.png")
lena_RGB = cv2.cvtColor(lena_BGR, cv2.COLOR_BGR2RGB)
plt.imshow(lena_RGB)
plt.show()
gray = cv2.cvtColor(lena_BGR, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

num_objects, labels  = cv2.connectedComponents(binary, connectivity=4);
print(num_objects)
plt.imshow(binary)
plt.show()