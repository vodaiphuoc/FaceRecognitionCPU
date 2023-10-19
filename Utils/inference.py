from Mtcnn.mtcnn import MTCNN
from Centerface.centerface import CenterFace
import os
import numpy as np
import cv2
from Utils.detection_utils import _mtcnn_forward, _center_face
from Deepface import (
    VGGFace,
    Facenet,
    ArcFace
)
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from typing import Tuple, Union, List

class Regconition(object):
    configs = {
            "VGG_Face": {"resize": (224, 224), 
                        "model": VGGFace.baseModel,
                        "weight_path": "pretrained_weights\\VGG_Face\\vgg_face_weights.h5" 
                        },
            "Facenet": {"resize": (160, 160),
                        "dim": 128 ,
                        "model": Facenet.InceptionResNetV2,
                        "weight_path": "pretrained_weights\\Facenet\\facenet_weights.h5"
                        },
            "Facenet512": {"resize": (160, 160),
                        "dim": 512,
                        "model": Facenet.InceptionResNetV2,
                        "weight_path": "pretrained_weights\\Facenet512\\facenet512_weights.h5"
                        },
            "ArcFace": {"resize": (112, 112),
                        "model": ArcFace.Get_ArcFace,
                        "weight_path": "pretrained_weights\\ArcFace\\arcface_weights.h5"
                        }
    }
    detector_configs = {"MTCNN" : _mtcnn_forward,
                        "CenterFace": _center_face
                        }

    def __init__(self,  detector_model_name: str ,
                        reg_model_name: str, 
                        camera_input_shape: tuple = (1280,720)) -> None:

        self.face_target_size = Regconition.configs.get(reg_model_name).get("resize")
        self.camera_input_shape = camera_input_shape

        # config detector
        if detector_model_name == "MTCNN":
            self.detector_instance = MTCNN()
            self.detector_forward = Regconition.detector_configs.get(detector_model_name)
        elif detector_model_name == "CenterFace":
            self.detector_instance = CenterFace(landmarks = True)
            self.detector_forward = Regconition.detector_configs.get(detector_model_name)
        
        # build and load weight of regconition model
        self.face_reg_instance = Regconition.build_model(reg_model_name)

    @classmethod
    def build_model(cls,model_name: str)-> Model:
        """Construct and load pretrained weights to face recognition model"""
        model_cfg = Regconition.configs.get(model_name)
        model_make_func = model_cfg.get("model")
        model = None
        # get architecture
        if model_cfg.get("dim") is not None:
            model = model_make_func(model_cfg.get("dim"))
        else:
            model = model_make_func()
        # load weigths
        model.load_weights(model_cfg.get("weight_path"))
        print("Model is ready for serving")
        return model

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,160,160,3), dtype=tf.float32)])
    def recog_graph_inference(self, data: tf.Tensor)-> List[tf.Tensor]:
        return self.face_reg_instance(data, training = False)

    def feed_forward_pipeline(self,input_image: np.ndarray)-> Union[Tuple[np.ndarray, list], int]:
        """"Main pipeline includes face detection and face recognition"""
        # stage 0: resize input image
        input_image = cv2.resize(input_image, self.camera_input_shape)
        # stage 1: get faces
        detected_faces, boxes = self.detector_forward( self.detector_instance, 
                                                        input_image,
                                                        self.face_target_size)
        print("find :{} faces".format(detected_faces.shape[0]))
        if len(detected_faces) != 0: 
            # stage 2: get embedding with input model
            embeddings = self.recog_graph_inference(detected_faces)
            return embeddings.numpy(), boxes
        else: 
            return 0