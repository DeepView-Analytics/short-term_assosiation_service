import json
import os
import numpy as np
import pickle
from sklearn.manifold import MDS
import networkx as nx

class GraphEmbedder:
    def __init__(self, max_dimensions=8, model_path="detection_association_service/Embedders/graph_model.pkl"):
        """
        Initialize the GraphEmbedder with graph and camera-to-node mapping.

        Args:
            config_path (str): Path to the graph configuration JSON.
            camera_map_path (str): Path to the camera-to-node mapping JSON.
            max_dimensions (int): Maximum number of dimensions for embeddings.
            model_path (str): Path to save or load the embedding model.
        """
        self.config_path = r"D:\DeepView\MicroServices\detection_association\detection_association_service\Embedders\places_graph.json"  # Replace with your actual config file path

        self.camera_map_path = r"D:\DeepView\MicroServices\detection_association\detection_association_service\Embedders\camera_map.json"  # Replace with your actual camera map file path

        self.max_dimensions = max_dimensions
        self.model_path = model_path
        self.node_embeddings = {}
        self.camera_to_node_map = self._load_camera_map()

        # Load or create the embedding model
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            self._create_model()

    def _load_camera_map(self):
        """Load the camera-to-node mapping."""
        with open(self.camera_map_path, "r") as f:
            return json.load(f)

    def _load_model(self):
        """Load precomputed embeddings."""
        print("Loading model from:", self.model_path)
        with open(self.model_path, "rb") as f:
            self.node_embeddings = pickle.load(f)

    def _create_model(self):
        """Create embeddings from graph using shortest path lengths and MDS."""
        print("Creating embeddings from graph using shortest path lengths and MDS.")
        with open(self.config_path, "r") as f:
            graph_data = json.load(f)

        nodes = list(graph_data.keys())
        graph = self._build_graph(graph_data, nodes)

        # Compute the shortest path length between all pairs of nodes
        shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))

        # Create a distance matrix
        distance_matrix = np.zeros((len(nodes), len(nodes)))
        for i, node in enumerate(nodes):
            for j, neighbor in enumerate(nodes):
                distance_matrix[i, j] = shortest_paths[node].get(neighbor, float('inf'))

        # Use MDS to create embeddings
        mds = MDS(n_components=self.max_dimensions, dissimilarity='precomputed', random_state=42)
        embeddings = mds.fit_transform(distance_matrix)

        # Save embeddings into node_embeddings
        self.node_embeddings = {node: embedding for node, embedding in zip(nodes, embeddings)}

        self._save_model()

    def _build_graph(self, graph_data, nodes):
        """Build a NetworkX graph from the adjacency list."""
        G = nx.Graph()
        for node, neighbors in graph_data.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        return G

    def _save_model(self):
        """Save embeddings to disk."""
        print("Saving model to:", self.model_path)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.node_embeddings, f)

    def create_embedding(self, node):
        """
        Retrieve embedding for a specific node.

        Args:
            node (str): Node name.

        Returns:
            np.ndarray: Embedding for the node.
        """
        if node not in self.node_embeddings:
            raise ValueError(f"Node {node} not found in the graph.")
        return self.node_embeddings[node]

    def create_combined_embedding(self, camera_id_1, camera_id_2, method="add"):
        """
        Create a single embedding for two nodes mapped from camera IDs.

        Args:
            camera_id_1 (str): First camera ID.
            camera_id_2 (str): Second camera ID.
            method (str): Combination method ('add', 'average', 'concatenate').

        Returns:
            np.ndarray: Combined embedding for the two nodes.
        """
        emb1 = self.create_embedding_from_camera(camera_id_1)
        emb2 = self.create_embedding_from_camera(camera_id_2)

        if method == "add":
            return emb1 + emb2
        elif method == "average":
            return (emb1 + emb2) / 2
        elif method == "concatenate":
            return np.concatenate((emb1, emb2))
        else:
            raise ValueError("Invalid method. Choose from 'add', 'average', or 'concatenate'.")

    def create_embedding_from_camera(self, camera_id):
        """
        Retrieve embedding for a node mapped from a camera ID.

        Args:
            camera_id (str): Camera ID.

        Returns:
            np.ndarray: Embedding for the corresponding node.
        """
        if camera_id not in self.camera_to_node_map:
            raise ValueError(f"Camera ID {camera_id} not found in the mapping.")

        node = self.camera_to_node_map[camera_id]
        return self.create_embedding(node)

def main():

    # Initialize the GraphEmbedder
    graph_embedder = GraphEmbedder()

    # Test creating an embedding for a specific node
    node_id = "camera_1"  # Replace with a valid node ID from your graph
    print(f"Testing embedding for node '{node_id}':")
    nodeA_embedding = graph_embedder.create_embedding_from_camera(node_id)
    print("Embedding:", nodeA_embedding)

    node_id = "camera_3"  # Replace with a valid node ID from your graph
    print(f"Testing embedding for node '{node_id}':")
    nodeB_embedding = graph_embedder.create_embedding_from_camera(node_id)
    print("Embedding:", nodeB_embedding)

    node_id = "camera_8"  # Replace with a valid node ID from your graph
    print(f"Testing embedding for node '{node_id}':")
    nodeL_embedding = graph_embedder.create_embedding_from_camera(node_id)
    print("Embedding:", nodeL_embedding)


    node_id = "E"  # Replace with a valid node ID from your graph
    print(f"Testing embedding for node '{node_id}':")
    nodeE_embedding = graph_embedder.create_embedding(node_id)
    print("Embedding:", nodeE_embedding)

    # Test creating embeddings for specific nodes and calculating distances
    distance = np.linalg.norm(nodeA_embedding - nodeB_embedding)
    print(f"Distance between A and B: {distance}")

    # Test creating embeddings for specific nodes and calculating distances
    distance = np.linalg.norm(nodeA_embedding - nodeA_embedding)
    print(f"Distance between A and A: {distance}")

    # Test creating embeddings for specific nodes and calculating distances
    distance = np.linalg.norm(nodeA_embedding - nodeL_embedding)
    print(f"Distance between A and L: {distance}")
if __name__ == "__main__":
    main()
