# Face recognition.

Face detection is a computer technology being used in a variety of applications that identifies human faces in digital images and videos.  
We will be using Python and OpenCV, which is the most popular library for computer vision, it uses haar cascades to search for faces.


### Goal
Our goal is to use a single pictures of each of our faces, use face-detector on the image, then resize, the image to only our detected face, so we are left with single image, resized to show only our face. Then calculate grayscale histogram and when using webcam or video file with faces, the program should calculate the indifference from the detected face in video and the resized images that the program loops through. then which calculation is the closest to 0, that should be the match.


![alt tag](https://images.duckduckgo.com/iu/?u=https%3A%2F%2Fsophosnews.files.wordpress.com%2F2015%2F02%2Fface-detection_550.jpg%3Fw%3D640&f=1)


### Steps we need to do:
1. Getting sample images.

2. Once we have sample data, we're going to need the Python programming language and OpenCV. Set up the work environment with importing the necessary modules. Work environment has to be in the same folder as the sample data/videos.
```Python
import numpy as np
import cv2

etc...
```
3. Run facedetect from opencv on the images, resize each image to the same size as detected face, apply grayscale histogram of each image.  

![alt_tag](https://github.com/UsernameDiana/Face_detector/blob/master/histogramsfordocumentaion.jpg)

4. Build the sum of the differences, when comparing picture and recognised face from video.


### Packages we will use:
* OpenCV - computer vision and machine learning software library.  
* Matplotlib - package for plotting. 
* Numpy - package for mathematical calculations. 
* Thresholding - thresholding any pixel as black if it is any darker or white if it is lighter than average ?

### Usage of programs:
Run in this order:
1. resize.py - creates template picture
2. my_face_detect.py - detects face and saves the last face detected before closing program
3. euclidian_distance.py - takes the templates images and the last face detected and from there computes the distanced and returns the closest
