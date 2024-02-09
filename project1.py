import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('image test 1.png')

if image is None:
    print("Error: Image not loaded.")
    exit()

def histogram(x):
    roi = image[x[1]:x[1]+x[3], x[0]:x[0]+x[2]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()


image1FirstFull=[315,267,125,189]
image1FirstUpper=[315,267,125,189//2]

image1SecondFull=[456,247,124,209]
image1SecondUpper=[456,247,124,209//2]


image2FirstFull=[235,128,70,277]
image2FirstUpper=[235,128,70,277//2]

image2SecondFull=[330,130,55,259]
image2SecondUpper=[330,130,55,259//2]

image2ThirdFull=[381,269,149,186]
image2ThirdUpper=[381,269,149,186//2]


#histogram(image1FirstFull)
#histogram(image1FirstUpper)
