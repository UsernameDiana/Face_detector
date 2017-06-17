import os
import cv2
from scipy.spatial import distance

def calculate_difference_in_images_recognized(path_to_temp_resized=["images_resize/face_of_1.jpg", "images_resize/face_of_2.jpg", "images_resize/face_of_3.jpg"],
                                              path_to_captured_from_video="captured_resized_face.jpg"):
    """    
        Calculates euclidian distance.
        
        :param 
        path_to_temp_resized : list
            list of paths to resized template images
        path_to_captured_from_video : str
            path to resized image captured from video 
        """

    template_images = (path_to_temp_resized)
    img_rgb = cv2.imread(path_to_captured_from_video)

    dst_values = {}
    for temp in template_images:
        template = cv2.imread(temp)
        t_w, t_h, p = template.shape

        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (t_w, t_h), interpolation=cv2.INTER_CUBIC)
        print(img_gray.shape, len(img_gray))
        img_gray = img_gray.flatten()

        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        print(gray_template.shape, len(gray_template))
        gray_template = gray_template.flatten()

        dst = distance.euclidean(img_gray, gray_template)
        dst = dst / (t_w * t_h)

        if temp not in dst_values.keys():
            dst_values[temp] = dst

    closest = min(dst_values.items(), key=lambda x: x[1])[0]

    print("RESULTS")
    print("distance values are: ")
    print(50 * "_")
    for img in template_images:
        value = dst_values[img]
        image_name = os.path.basename(img)
        print("Euclidian distance: {} : {}".format(image_name, value))
    result = cv2.imread(closest)

    while True:
        cv2.imshow('Detected face is closest to', result)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()