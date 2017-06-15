'''
Runs resize, facedetection from webcam and calculates difference to find out whos face was detected

USAGE:
    run_program.py
'''

import resize
#import face_detect_webcam
import euclidian_distance

def face_detection(image_path, path_output_folder, path_cascade, path_to_captured_from_video, path_to_temp_resized):

    """
    :param image_path: 
    :param path_output_folder: 
    :param path_cascade: 
    :param path_to_captured_from_video: 
    :param path_to_temp_resized: 
    :return: 
    """

    resize.detect_templ_and_resize_save(image_path, path_output_folder, path_cascade)
    #my_face_detect.detect_from_video_save_resized_face(path_output_folder, path_cascade)
    euclidian_distance(path_to_temp_resized, path_to_captured_from_video)

image_path=["images_resize/1.jpg", "images_resize/2.jpg", "images_resize/3.jpg"]
path_output_folder = "images_resize"
path_cascade="../haarcascades/haarcascade_frontalface_default.xml"
path_to_captured_from_video="captured_resized_face.jpg"
path_to_temp_resized=["images_resize/face_of_1.jpg", "images_resize/face_of_2.jpg", "images_resize/face_of_3.jpg"]


face_detection(image_path, path_output_folder, path_cascade, path_to_captured_from_video, path_to_temp_resized)