import cv2
def detect_templ_and_resize_save(image_path, path_output, path_cascade):
    """    
    Saves resized image after face detecting.
    :param 
    image_path : str
        path to image
    path_output : str
        path to out put image 
    path_cascade : str
        path to cascade depending on what you are interested to detect
    """
    faceCascade = cv2.CascadeClassifier('../../haarcascades/haarcascade_frontalface_default.xml')
    images = ['1.jpg','2.jpg','3.jpg']

    for img in images:
        image = cv2.imread(img) # reading each image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale

        faces = faceCascade.detectMultiScale(gray, 1.3, 5) # detects face using cascades

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2) # drawing rectangle on top left and bottom right corner
            new_img = image[y:y+h, x:x+w] # cropping using slice :
            print(new_img.shape)
            cv2.imwrite('face_of_'+img, new_img)
