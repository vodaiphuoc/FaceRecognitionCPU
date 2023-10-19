from sklearn.metrics.pairwise import cosine_similarity
import cv2
import os, math
from Utils.inference import Regconition

path = os.getcwd() 
print("Init")
obj = Regconition("MTCNN","Facenet")
img = cv2.imread(path + "/photos/hoang/hoang_0.jpeg")
vector1, _ = obj.feed_forward_pipeline(img)
print("type of embedding: ",vector1)


img3 = cv2.imread(path + "/photos/hoang/hoang_1.jpeg")
vector3, _ = obj.feed_forward_pipeline(img3)

def cosine_similarity_Ver1(vector1, vector2):
    sum = 0.0
    length_1 = 0.0
    length_2 = 0.0
    for each_axis_1, each_axis_2 in zip(vector1, vector2):
        sum += each_axis_1*each_axis_2
        length_1 += each_axis_1*each_axis_1
        length_2 += each_axis_2*each_axis_2

    length_1 = math.sqrt(length_1)
    length_2 = math.sqrt(length_2)
    return sum/(length_1*length_2)

print(cosine_similarity_Ver1(vector1[0], vector3[0]))
# print(cosine_similarity_Ver1(vector1[0], vector2[0]))
# print(cosine_similarity_Ver1(vector2[0], vector3[0]))

print(cosine_similarity(vector1, vector3))
# print(cosine_similarity(vector1, vector2))
# print(cosine_similarity(vector2, vector3))
# print(cosine_similarity(vector2, vector3))