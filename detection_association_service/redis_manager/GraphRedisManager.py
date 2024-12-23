from networkx import Graph
import redis
from typing import List, Tuple
from pydantic import BaseModel, Field
import uuid
from v3.detected_person_metadata import DetectionMetadata
class GraphRedisManager:
    def __init__(self, redis_host='localhost', redis_port=6379, graph_name='person_detection_graph'):
        """
        Initialize the GraphRedisManager.
        
        :param redis_host: Redis host.
        :param redis_port: Redis port.
        :param graph_name: Name of the graph in Redis.
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.graph_name = graph_name
        
        # Connect to Redis
        self.redis_client = redis.Redis(host=self.redis_host, port=self.redis_port, db=5)
        self.graph = Graph(self.graph_name, self.redis_client)
        
        # Ensure the graph exists or create it
        self._initialize_graph()

    def _initialize_graph(self):
        """Ensure the graph exists or create it."""
        # Check if the graph already exists
        if not self.graph.exists():
            print(f"Graph {self.graph_name} does not exist. Creating a new graph.")
            # Create the graph schema if it doesn't exist
            self.graph.query("""
                CREATE CONSTRAINT ON (p:Person) ASSERT p.person_key IS UNIQUE;
                CREATE CONSTRAINT ON (d:Detection) ASSERT d.detection_key IS UNIQUE;
            """)
        else:
            print(f"Graph {self.graph_name} already exists.")

    def update_graph(self, person_detection_pairs: List[Tuple[str, DetectionMetadata]]):
        """
        Update the graph by creating person and detection nodes and associating them.
        
        :param person_detection_pairs: List of tuples where each tuple contains a person_id and a detection_metadata object.
        """
        # Start a transaction to process the batch of nodes and relationships
        for person_id, detection_metadata in person_detection_pairs:
            # Create or merge the person node, ensuring uniqueness by `person_key`
            self.graph.query(f"""
                MERGE (p:Person {{person_key: '{person_id}'}})
            """)

            self.graph.query(f"""
                CREATE (d:Detection {{detection_key: '{detection_metadata.person_key}',
                                      frame_key: '{detection_metadata.frame_key}', 
                                      embedding_key: '{detection_metadata.embedding_key}', 
                                      keypoint_key: '{detection_metadata.keypoint_key}', 
                                      is_face_clear: {detection_metadata.is_face_clear}}})
            """)

            # Create a relationship between the person and the detection if it doesn't exist
            self.graph.query(f"""
                MATCH (p:Person {{person_key: '{person_id}'}}), 
                      (d:Detection {{detection_key: '{detection_metadata.person_key}'}})
                MERGE (p)-[:DETECTED]->(d)
            """)

        print(f"Graph updated: {len(person_detection_pairs)} person-detection associations.")
