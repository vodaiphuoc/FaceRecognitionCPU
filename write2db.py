import os
import cv2
import numpy as np
import sqlite3
from Utils.inference import Regconition

detector_name = "MTCNN"
model_name = "ArcFace"
# Iniit face regconition model
model = Regconition(detector_name,model_name)

# Connect to the SQLite database or create a new one
conn = sqlite3.connect("database\\face_embeddings.db")
cursor = conn.cursor()
# Create a table to store the face encodings and image names
tabel_name = "face_encodings_"+ model_name
cursor.execute("CREATE TABLE IF NOT EXISTS "+tabel_name +" (id INTEGER PRIMARY KEY, encoding BLOB, name TEXT, library TEXT)")

# Specify the root folder path containing the image files
root_folder = "photos"

def extract_name_from_filename(filename: str)-> str:
    # Split the filename by "_" to extract the name
    parts = filename.split("_")
    if len(parts) >= 1:
        return parts[0]
    else:
        return "Unknown"

def process_images(folder_path):
    for root, dirs, files in os.walk(folder_path):
        # Iterate over the files in the current folder
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                # Extract the name from the filename
                name = extract_name_from_filename(file)

                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                embeddings, _ = model.feed_forward_pipeline(image)
                
                for single_embedding in embeddings:
                    print("Length of single embedding: ", embeddings.shape, ", embedding dtype: ", embeddings.dtype)
                    # Convert the face embedding to bytes for storing in the database
                    embedding_bytes = np.array(single_embedding).tobytes()
                    cursor.execute("INSERT INTO "+tabel_name+" (encoding, name) VALUES (?, ?)", (embedding_bytes, name))
                    print("Inserted")
                    print("check insert dim of embedding: ", np.frombuffer(embedding_bytes, dtype = np.float32).shape)

        # Commit the changes after processing all images in the current folder
        conn.commit()
# Process images in the root folder and its subfolders
process_images(root_folder)

# Close the database connection
conn.close()