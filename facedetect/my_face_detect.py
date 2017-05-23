import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')

matyas = cv2.imread('images_resize/face_of_2.jpg')
gray_matyas = cv2.cvtColor(matyas, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture(1)
match = []
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        resized = cv2.resize(roi_gray, (83, 83), interpolation=cv2.INTER_CUBIC)
        print(type(roi_gray), resized.shape)
        cv2.imwrite('captured_resized_face.jpg', resized)
        cap_face = cv2.imread('captured_reized_face.jpg')
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()