import numpy as np
from detection_association_service.Embedders.Graph_embedder import GraphEmbedder
from detection_association_service.Embedders.time_embedder import TimeEmbedder


class Encoder:
    def __init__(self):
        self.graph_embedder = GraphEmbedder(max_dimensions=8)
        self.time_embedder = TimeEmbedder(embedding_dim=8)
        
    def encode(self,camera_id,time_stamp,embeddings):
        position_embedding = self.graph_embedder.create_embedding_from_camera(camera_id)
        time_embedding = self.time_embedder.get_time_embedding(time_stamp)

        detection_embeddings = [ np.concatenate((embedding, time_embedding, position_embedding)) for embedding in embeddings]
        return detection_embeddings
    
    def update_centroid(current_centroid, new_embedding, distance):
        """
        Update the centroid by incorporating the new embedding, based on the provided distance.
        Splits both current centroid and new embedding into two parts.

        :param current_centroid: The current centroid vector.
        :param new_embedding: The new embedding vector to update the centroid with.
        :param distance: The distance between the current centroid and the new embedding.
        
        :return: Updated centroid.
        """
        # Split the current centroid and new embedding into two parts
        current_centroid_apearance_embedding = current_centroid[:-16]   # All values except the last 16

        new_embedding_apearance_embedding = new_embedding[:-16]           # All values except the last 16
        new_embedding_spacial_embedding = new_embedding[-16:]            # Last 16 values

        # Calculate a weight (lambda) based on the distance
        # The smaller the distance, the higher the weight (more influence)
        lambda_weight = max(0, 1 - distance)  # Assuming the distance is normalized between 0 and 1

        # Update the centroid using the weighted average for both parts
        updated_apearance_embedding = lambda_weight * new_embedding_apearance_embedding + (1 - lambda_weight) * current_centroid_apearance_embedding

        # Reconstruct the updated centroid by combining the updated parts
        updated_centroid = np.concatenate([updated_apearance_embedding, new_embedding_spacial_embedding])

        return updated_centroid