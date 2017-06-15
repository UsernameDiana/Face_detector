import numpy as np
import cv2
from scipy.spatial import distance

template_images = ['../images_resize/face_of_1.jpg', '../images_resize/face_of_2.jpg', '../images_resize/face_of_3.jpg']
img_rgb = cv2.imread('captured_resized_face.jpg')
dst_values = {}
for ti in template_images:
    template = cv2.imread(ti)
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
    if ti not in dst_values.keys():
        dst_values[ti] = dst
closest = min(dst_values.items(), key=lambda x: x[1])[0]
print(dst_values)
result = cv2.imread(closest)
while True:
    cv2.imshow('Closest', result)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()