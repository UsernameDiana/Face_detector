# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

# local modules
from video import create_capture
from common import clock, draw_str


# detecting if there is a face
def detect (img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


first_frame = None

def process_frame(frame, amp):
    global first_frame

    # convert the frame to grayscale, and blur it
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if first_frame is None:
        first_frame = gray

    # compute the absolute difference between the current frame and
    # first frame
    frame_delta = cv2.absdiff(first_frame, gray) * amp
    thresh = cv2.threshold(frame_delta, 40, 255, cv2.THRESH_BINARY)[1]
 
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)

    return thresh, frame_delta


if __name__ == '__main__':
    global text
    frames = []

    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")
    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)

    cam = create_capture(video_src, fallback='synth:bg=../sample_images/2.jpg:noise=0.05')

    while True:
        ret, frame, amp, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        thresh, frame_delta = process_frame(frame, amp)
        #find_contours(frame, thresh, area)

        text = 'Nothing to see...'
        # draw the text on the frame
        cv2.putText(frame, "Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

         # show the frame and record if the user presses a key
        cv2.imshow("Video Stream", frame)
        cv2.imshow("Threshold", thresh)
        cv2.imshow("Frame Delta", frame_delta)

        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect', vis)

        frames.append((frame, thresh, frame_delta))
        if cv2.waitKey(5) == 27:
            break


    cv2.destroyAllWindows()