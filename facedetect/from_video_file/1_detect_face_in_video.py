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

if __name__ == '__main__':
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '')
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)

    cap = create_capture(video_src, fallback='synth:bg=../images_resize/3.jpg:noise=0.05')

    while True:
        ret, frame = cap.read() # ret = return value
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),1) # fills in the rectangle
            roi_color = frame[y:y + h, x:x + w]

            resized = cv2.resize(roi_color, (83, 83), interpolation=cv2.INTER_CUBIC)
            print(type(roi_color), resized.shape)
            cv2.imwrite('captured_resized_face.jpg', resized)

        cv2.imshow('img',frame)
        k = cv2.waitKey(35)
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()