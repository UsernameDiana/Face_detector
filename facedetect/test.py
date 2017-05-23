import numpy as np
import cv2
from scipy.spatial import distance

img_rgb = cv2.imread('captured_resized_face.jpg')
print(img_rgb.shape, len(img_rgb))
img_rgb = img_rgb.flatten()
template = cv2.imread('images_resize/face_of_1.jpg')
print(template.shape, len(template))
template = template.flatten()

dst = distance.euclidean(img_rgb, template)
print(dst)