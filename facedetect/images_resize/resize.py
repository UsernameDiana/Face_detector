import cv2

faceCascade = cv2.CascadeClassifier('../../haarcascades/haarcascade_frontalface_default.xml')
images = ['1.jpg','2.jpg','3.jpg']

for img in images:
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        new_img = image[y:y+h, x:x+w]
        cv2.imwrite('face_of_'+img, new_img)

