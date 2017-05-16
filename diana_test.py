# Python 2/3 compatibility
from __future__ import print_function
import sys, cv2
import numpy as np
from numpy import random

PY3 = sys.version_info[0] == 3
if PY3:
    xrange = range

# local modules
from video import create_capture
from common import clock, draw_str


# detecting with cascade and drawing rectangules
def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

# making gaussian blur
def make_gaussians(cluster_n, img_size):
    points = []
    ref_distrs = []
    for i in xrange(cluster_n):
        mean = (0.1 + 0.8*random.rand(2)) * img_size
        a = (random.rand(2, 2)-0.5)*img_size*0.1
        cov = np.dot(a.T, a) + img_size*0.05*np.eye(2)
        n = 100 + random.randint(900)
        pts = random.multivariate_normal(mean, cov, n)
        points.append( pts )
        ref_distrs.append( (mean, cov) )
    points = np.float32( np.vstack(points) )
    return points, ref_distrs

def draw_gaussain(img, mean, cov, color):
    x, y = np.int32(mean)
    w, u, vt = cv2.SVDecomp(cov)
    ang = np.arctan2(u[1, 0], u[0, 0])*(180/np.pi)
    s1, s2 = np.sqrt(w)*3.0
    cv2.ellipse(img, (x, y), (s1, s2), ang, 0, 360, color, 1, cv2.LINE_AA)
    
    

# main run method
if __name__ == '__main__':
    import sys, getopt
    print(__doc__)
    cluster_n = 5
    img_size = 512