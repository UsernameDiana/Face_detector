import cv2

faceCascade = cv2.CascadeClassifier('../../haarcascades/haarcascade_frontalface_default.xml')
images = ['../sample_images/1.jpg','../sample_images/2.jpg','../sample_images/3.jpg']

for img in images:
    print(img)
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print(x, y, x+w, y+h)
        new_img = image[y:y+h, x:x+w]
        cv2.imwrite(img, new_img)

