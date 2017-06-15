'''
detects faces using haarcascades from video file and saves the result

USAGE:
    1_detect_face_in_video.py [<video_source>]
'''

import cv2
import sys
import getopt

# local modules
from video import create_capture

face_cascade = cv2.CascadeClassifier('../../haarcascades/haarcascade_frontalface_default.xml')

diana = cv2.imread('../images_resize/face_of_3.jpg')
gray_diana = cv2.cvtColor(diana, cv2.COLOR_BGR2GRAY)
# take with and hight of image, its a tuple!!!!
w, h = gray_diana.shape
print(w, h) # 83, 83

if __name__ == '__main__': # wont run if imported as module in another module
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '')
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)

    cap = create_capture(video_src, fallback='synth:bg=../images_resize/3.jpg:noise=0.05')

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

        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()