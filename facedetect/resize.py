import cv2
import os

def detect_templ_and_resize_save(image_path=["images_resize/1.jpg", "images_resize/2.jpg", "images_resize/3.jpg"],
                                 path_output_folder="/images_resize",
                                 path_cascade="../haarcascades/haarcascade_frontalface_default.xml"):
    """    
    Saves resized image after face detecting.
    
    :param 
    image_path : list
        list of paths to image
    path_output : str
        path to out put image 
    path_cascade : str
        path to cascade depending on what you are interested to detect
    """

    faceCascade = cv2.CascadeClassifier(path_cascade)
    images = (image_path)

    for img in images:
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            new_img = image[y:y+h, x:x+w]
            print("tuple of array dimension: ")
            print(new_img.shape)
            cv2.imwrite(os.path.join(path_output_folder,'face_of_'+img), new_img)