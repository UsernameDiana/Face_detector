'''
Calculates euclidian distance between the original template and the captured image from video

USAGE:
    2_euclidian_distance.py
'''
import os
import cv2
from scipy.spatial import distance

template_images = ['../images_resize/face_of_1.jpg', '../images_resize/face_of_2.jpg', '../images_resize/face_of_3.jpg']
captured_image = cv2.imread('captured_resized_face.jpg')

dst_values = {}
for temp in template_images:
    template = cv2.imread(temp)
    t_w, t_h, p = template.shape

    capt_image_gray = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)
    capt_image_gray = cv2.resize(capt_image_gray, (t_w, t_h), interpolation=cv2.INTER_CUBIC)
    capt_image_gray = capt_image_gray.flatten()

    template_image_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_image_gray = template_image_gray.flatten()

    dist = distance.euclidean(capt_image_gray, template_image_gray)
    dist = dist / (t_w * t_h)
    if temp not in dst_values.keys():
        dst_values[temp] = dist

closest = min(dst_values.items(), key=lambda x: x[1])[0]
print("RESULTS")
print("distance values are: ")
print(50*"_")
for img in template_images:
    value = dst_values[img]
    image_name = os.path.basename(img)
    print("Euclidian distance: {} : {}".format(image_name, value) )
result = cv2.imread(closest)

while True:
    cv2.imshow('Detected face is closest to', result)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()