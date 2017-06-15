import cv
import os

def detect_from_video_save_resized_face(path_output_folder="facedetect",
                                        path_cascade="../haarcascades/haarcascade_frontalface_default.xml"):

    """    
        Saves resized image after face detected on webcam.
        :param 
        path_output : str
            path to out put image 
        path_cascade : str
            path to cascade depending on what you are interested to detect
        """

    diana = cv2.imread('images_resize/face_of_3.jpg')
    gray_diana = cv2.cvtColor(diana, cv2.COLOR_BGR2GRAY)

    w, h = gray_diana.shape
    print("width and height of image: " + w, h)

    face_cascade = cv2.CascadeClassifier(path_cascade)

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),1)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            resized = cv2.resize(roi_color, (w, h), interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(os.path.join(path_output_folder, 'captured_resized_face.jpg'), resized)
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


