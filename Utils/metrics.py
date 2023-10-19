from typing import List
import numpy as np
import math

def cosin_sim( target_embedding: np.ndarray, 
                stored_embeddings: np.ndarray, 
                embedding_lengths: np.ndarray, 
                stored_user_name: List[str],
                minimum_distance: float = 0.6)-> str:
    """Calculate cosine similarity based on dot product of vector and matrix"""

    # find length of target_embedding
    target_embedding_length = math.sqrt(sum([axis*axis for axis in target_embedding]))
    multiply_2_length = np.multiply(embedding_lengths, 1/target_embedding_length)

    dot_product = np.dot(stored_embeddings, target_embedding)  
    cosine_similarity = np.multiply(dot_product, multiply_2_length).tolist()
    
    # Determine user
    best_metric = max(cosine_similarity)
    if best_metric > minimum_distance:
        best_index = cosine_similarity.index(best_metric)
        return stored_user_name[best_index]
    else:
        return "Unknow user"