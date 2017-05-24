import cv2
import numpy as np

img_rgb = cv2.imread('captured_resized_face.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
print(img_gray.shape)

template = cv2.imread('../images_resize/face_of_1.jpg')
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
print(template_gray.shape)

w, h = template_gray.shape[::-1]
count = 0

# comparison in a list, matching a detected face with the templates
res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
threshold = 0.4
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]): # points which have values greater than threshold.
    count += 1
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)


print("Total Points Detected: {}".format(count))
cv2.imshow('Detected',img_rgb)