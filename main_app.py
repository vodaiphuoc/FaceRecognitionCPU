import cv2
import math
import os
from Utils.inference import Regconition
import sqlite3
import numpy as np
from datetime import datetime
import argparse
from typing import Tuple, List
from Utils.metrics import cosin_sim

def _init_program_(args, camera_shape: tuple)-> Tuple[Regconition, np.ndarray, np.ndarray, List[str]]:
    # Open the default camera (change index if you have multiple cameras)
    detector_name = args.det_model_name
    model_name = args.reg_model_name
    table_name = "face_encodings_"+ model_name
    # Iniit face regconition model
    model = Regconition(detector_name,model_name, camera_shape)

    # Connect to the SQLite database
    conn = sqlite3.connect("database\\face_embeddings.db")
    cursor = conn.cursor()
    
    # Retrieve all face encodings and names from the database and preprocessing
    cursor.execute("SELECT encoding, name FROM "+table_name)
    rows = cursor.fetchall()
    conn.close()
    stored_embeddings = []
    embedding_lengths = []
    stored_user_name = []
    # Iterate over the retrieved rows
    for row in rows:
        # Retrieve the stored embedding and name
        stored_embedding = np.frombuffer(row[0], dtype=np.float32)
        stored_embeddings.append(stored_embedding)

        embedding_length = math.sqrt(sum([axis*axis for axis in stored_embedding]))
        embedding_lengths.append(1/embedding_length)
        
        stored_user_name.append(row[1])

    stored_embeddings = np.array(stored_embeddings)
    # transpose to column vector
    embedding_lengths = np.transpose(np.array(embedding_lengths))

    return model, stored_embeddings, embedding_lengths , stored_user_name

def main(args):
    # init camera
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    camera.set(cv2.CAP_PROP_FRAME_WIDTH , 800) 
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT , 600)
    model, stored_embeddings, embedding_lengths , stored_user_name = _init_program_(args, camera_shape = (800,600))
    print("Stored embeddings shape: ", stored_embeddings.shape)
    # main loop
    while True:
        # Read the camera frame
        ret, frame = camera.read()

        # get embedding, handling case of no detected face
        return_result = model.feed_forward_pipeline(frame)
        if return_result == 0:
            continue
        else:
            embeddings, face_boxes = return_result[0], return_result[1]

        # Iterate over the detected faces
        for embedding, (x, y, w, h) in zip(embeddings, face_boxes):
            # Recognize the face by comparing the embedding with the stored encodings
            recognized_name = cosin_sim(embedding, stored_embeddings, embedding_lengths , stored_user_name)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the recognized name above the face rectangle
            cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("Face Recognition", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Release the camera and close the database connection
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--det', dest='det_model_name', metavar='N', type=str,
                        help='name of face detection model')
    parser.add_argument('--reg', dest='reg_model_name', type=str,
                        help='name of face recognition model')

    args = parser.parse_args()
    main(args)