import numpy as np

class TimeEmbedder:
    def __init__(self, embedding_dim=8, scaling_factor=1.0):
        """
        Initialize the TimeEmbedder.
        
        Args:
            embedding_dim (int): Dimensionality of the time embedding (default is 8).
            scaling_factor (float): A factor to scale the time embedding to have a meaningful impact on L2 distance.
        """
        self.embedding_dim = embedding_dim
        self.scaling_factor = scaling_factor

    def get_time_embedding(self, timestamp):
        """
        Generate a time embedding based on the timestamp (in seconds).
        
        Args:
            timestamp (float): The timestamp to generate the time embedding for.
        
        Returns:
            np.ndarray: The time embedding vector (embedding_dim length).
        """
        # Normalize time to a value between 0 and 1 within a 24-hour period
        total_seconds_in_day = 86400  # 24 hours * 60 minutes * 60 seconds
        normalized_time = (timestamp % total_seconds_in_day) / total_seconds_in_day
        
        # Apply the scaling factor to amplify the time differences in the embedding space
        scaled_time = normalized_time * self.scaling_factor

        # Generate the time embedding using sine and cosine functions
        embedding = np.zeros(self.embedding_dim)
        
        # Divide the embedding into sine and cosine functions to represent periodicity
        for i in range(self.embedding_dim // 2):
            embedding[2 * i] = np.sin(2 * np.pi * scaled_time * (2 ** i))
            embedding[2 * i + 1] = np.cos(2 * np.pi * scaled_time * (2 ** i))
        
        return embedding

    def calculate_l2_distance(self, embedding1, embedding2):
        """
        Calculate the L2 distance between two time embeddings.
        
        Args:
            embedding1 (np.ndarray): First time embedding.
            embedding2 (np.ndarray): Second time embedding.
        
        Returns:
            float: The L2 distance between the embeddings.
        """
        return np.linalg.norm(embedding1 - embedding2)


# Example usage
if __name__ == "__main__":
    # Create an instance of the TimeEmbedder class with scaling factor to amplify time differences
    time_embedder = TimeEmbedder(embedding_dim=8, scaling_factor=100.0)
    
    # Example timestamps (in seconds since epoch or any time scale)
    timestamp_1 = 146498.548774  # Example timestamp 1
    timestamp_2 = 146505.548774  # Example timestamp 2
    timestamp_3 = 146588.548774  # Example timestamp 3

    # Get the time embeddings for both timestamps
    embedding_1 = time_embedder.get_time_embedding(timestamp_1)
    print(f"Time embedding for timestamp {timestamp_1}: {embedding_1}")

    embedding_2 = time_embedder.get_time_embedding(timestamp_2)
    print(f"Time embedding for timestamp {timestamp_2}: {embedding_2}")

    embedding_3 = time_embedder.get_time_embedding(timestamp_3)
    print(f"Time embedding for timestamp {timestamp_3}: {embedding_3}")

    # Calculate L2 distance (this represents the similarity of time embeddings)
    distance_1_2 = time_embedder.calculate_l2_distance(embedding_1, embedding_2)
    print(f"L2 Distance between embeddings 1 and 2: {distance_1_2}")

    distance_1_3 = time_embedder.calculate_l2_distance(embedding_1, embedding_3)
    print(f"L2 Distance between embeddings 1 and 3: {distance_1_3}")
