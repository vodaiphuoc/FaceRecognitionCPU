import gdown
import os

if __name__ == "__main__":
    model_dict = {"Facenet": "facenet_weights",
                "VGG_Face": "vgg_face_weights",
                "Facenet512": "facenet512_weights",
                "ArcFace": "arcface_weights"
                }
    
    for folder_name, model_weights_name in model_dict.items():
        target_dir = os.getcwd()+"\\pretrained_weights\\"+folder_name+"\\"+ model_weights_name+".h5"
        # check if the file already exist
        if os.path.isfile(target_dir):
            print("The {} model has already donwloaded".format(folder_name))
        else:
            os.mkdir(os.getcwd()+"\\pretrained_weights\\"+folder_name)
            print("The {} is downloaded in {}".format(folder_name, target_dir))
            url="https://github.com/serengil/deepface_models/releases/download/v1.0/"+model_weights_name+".h5"
            gdown.download(url, target_dir)
