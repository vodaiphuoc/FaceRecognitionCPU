from Mtcnn.mtcnn import MTCNN
from Centerface.centerface import CenterFace
import numpy as np
import cv2
from typing import Tuple, List
import tensorflow as tf

def _mtcnn_forward(detector_instance: MTCNN,
                    input_image: np.ndarray, 
                    face_target_size: Tuple[int,int]
                    ) -> Tuple[tf.Tensor, List[Tuple[int]]]:
    """
    A feedforward function for MTCNN model
    Output: Tuple[tensor, list] Expect tensor has dimension (batch, H, W, C)
    """
    detected_results = detector_instance.detect_faces(input_image)
    processed_faces = []
    boxes = []
    for single_result in detected_results:
        ( x, y, w, h ), confident_score = single_result["box"], single_result["confidence"]
        print("confident score: ", confident_score)
        if confident_score < 0.5:
            continue
        else:
            # Extract the face region from the image
            result_face = input_image[y:y+h, x:x+w]
            result_face = cv2.resize(result_face, face_target_size)
            # append to result
            processed_faces.append(result_face)
            boxes.append(( x, y, w, h ))
    
    return tf.convert_to_tensor(processed_faces, dtype=tf.float32), boxes

def _center_face(detector_instance: CenterFace, 
                input_image: np.ndarray,
                face_target_size: Tuple[int, int]
                ) -> Tuple[tf.Tensor, List[Tuple[int]]]:
    H, W,_ = input_image.shape
    detected_results, _ = detector_instance(input_image, H, W)
    processed_faces = []
    boxes = []
    for single_result in detected_results:
        x1, y1, x2, y2 , confident_score = single_result
        if confident_score < 0.5:
            continue
        else:
            # Extract the face region from the image
            result_face = input_image[int(y1):int(y2), int(x1):int(x2)]
            result_face = cv2.resize(result_face, face_target_size)
            processed_faces.append(result_face)
            boxes.append((int(x1),int(y1), int(x2)-int(x1), int(y2)-int(y1)))

    return tf.convert_to_tensor(processed_faces, dtype=tf.float32), boxes
