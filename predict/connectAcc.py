import cv2
import matplotlib.pyplot as plt

def getConnectedAcc(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    num_objects, labels = cv2.connectedComponents(binary, connectivity=4);
    return num_objects

if __name__ == "__main__":
    lena_BGR = cv2.imread("../dataset/mini/val/label/41.png")
    print(getConnectedAcc(lena_BGR))

