import numpy as np
import cv2
from scipy.spatial import distance

img_rgb = cv2.imread('captured_resized_face.jpg')
template = cv2.imread('../images_resize/face_of_1.jpg')
t_w, t_h, p = template.shape

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
img_gray = cv2.resize(img_gray, (t_w, t_h), interpolation=cv2.INTER_CUBIC)
print(img_gray.shape, len(img_gray))
img_gray = img_gray.flatten()

gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
print(gray_template.shape, len(gray_template))
gray_template = gray_template.flatten()

dst = distance.euclidean(img_gray, gray_template)
dst = dst / (t_w * t_h)
print(dst)