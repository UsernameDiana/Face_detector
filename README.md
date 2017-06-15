# Face recognition.

### Goal
The basics of face detection using Haar Feature-based Cascade Classifiers, then calculate difference of the detected face in video and the template image to see who was detected.
Using haar cascades for detection is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.


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
