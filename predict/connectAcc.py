import cv2
import matplotlib.pyplot as plt
import numpy as np

def getErrorNum(srcImg, compareImg):
    gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    num_labels, labels = cv2.connectedComponents(binary, connectivity=4)
    garyPredict = cv2.cvtColor(compareImg, cv2.COLOR_BGR2GRAY)
    retPredict, binaryPredict = cv2.threshold(garyPredict, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    shortLines = 0
    for i in range(1, num_labels):
        mask = labels != i
        output = binaryPredict.copy()
        output = output.reshape(1024, 1024, 1)
        output[:, :, 0][mask] = 0

        num_labels2, labels2 = cv2.connectedComponents(output, connectivity=4)
        if num_labels2 > 2:
            shortLines = shortLines + num_labels2 - 2

            #cv2.imshow('oginal1', output)
            #cv2.waitKey()
            #cv2.destroyAllWindows()


    return shortLines

def getShortError(label, predict):
    return getErrorNum(predict, label)

def getOpenError(label, predict):
    return getErrorNum(label, predict)

if __name__ == "__main__":
    label = cv2.imread("../dataset/mini/val/label/45.png")
    predict = cv2.imread("../dataset/mini/val/label/45s.png")

    print(getShortError(label, predict))
    print(getOpenError(label, predict))

