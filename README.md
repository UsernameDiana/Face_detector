# Face recognition.

Face detection is a computer technology being used in a variety of applications that identifies human faces in digital images and videos.  
We will be using Python and OpenCV, which is the most popular library for computer vision, it uses haar cascades to search for faces.


### Goal
Our goal is to use a single pictures of each of our faces, use face-detector on the image, then resize, the image to only our detected face, so we are left with single image, resized to show only our face. Then calculate grey scale histogram and when using web camera or video file with faces, the program should calculate the indifference from the detected face in video and the resized images that the program loops through. then which calculation is the closest to 0, that should be the match.


![alt tag](https://images.duckduckgo.com/iu/?u=https%3A%2F%2Fsophosnews.files.wordpress.com%2F2015%2F02%2Fface-detection_550.jpg%3Fw%3D640&f=1)


### Steps we need to do:
1. Getting sample video/images.

2. Once we have sample data, we're going to need the Python programming language and OpenCV. Set up the work enviorement with importing the nececary modules. Work enviorement has to be in the same folder as the sample data/videos.
```Python
import numpy as np
import cv2

etc...
```
3. Run facedetect from opencv on the images, resize each image to the same size as detected face, apply gray scale histogram of each image.  

![alt tag](https://scontent.xx.fbcdn.net/v/t34.0-0/p280x280/18644345_10154930437348241_2107617379_n.jpg?_nc_ad=z-m&oh=cee4ab19fcf9316a909e3570ca447fd1&oe=592679D9)

4. Build the sum of the indiferences, when comparing picture and recognised face from video.


### Packages we will use:
* OpenCV - computer vision and machine learning software library.  
* Matplotlib - package for plotting. 
* Numpy - package for mathematical calculations. 
* Thresholding - thresholding any pixel as black if it is any darker or white if it is lighter then average ?

### Usage of programm:
