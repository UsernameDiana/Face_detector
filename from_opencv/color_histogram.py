import cv2
import numpy as np
from matplotlib import pyplot as plt
images = ['../facedetect/images_resize/face_of_1.jpg','../facedetect/images_resize/face_of_2.jpg','../facedetect/images_resize/face_of_3.jpg']
for image in images:
    img = cv2.imread(image, -1)
    cv2.imshow('1',img)

    color = ('b','g','r')
    for channel,col in enumerate(color):
        histr = cv2.calcHist([img],[channel],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.title('Histogram for color scale picture')
    plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break             # ESC key to exit
cv2.destroyAllWindows()