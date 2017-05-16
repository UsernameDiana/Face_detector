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


if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

