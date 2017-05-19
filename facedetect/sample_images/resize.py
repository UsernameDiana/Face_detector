import cv2

faceCascade = cv2.CascadeClassifier('../../haarcascades/haarcascade_frontalface_default.xml')

image = cv2.imread('../sample_images/3.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = faceCascade.detectMultiScale(gray, 1.3, 5)
print(len(faces))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found" ,image)
cv2.waitKey(0)
cv2.destroyAllWindows()