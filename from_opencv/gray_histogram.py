import cv2
import numpy as np
from matplotlib import pyplot as plt
images = ['../facedetect/images_resize/face_of_1.jpg','../facedetect/images_resize/face_of_2.jpg','../facedetect/images_resize/face_of_3.jpg']
for image in images:
    gray_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('1',gray_img)
    hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
    plt.hist(gray_img.ravel(),256,[0,256])
    plt.title('Histogram for gray scale picture')
    plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF     
    if k == 27: break             # ESC key to exit 
cv2.destroyAllWindows()