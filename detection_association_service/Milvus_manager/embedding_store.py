import asyncio
import os
import sys
from PIL import Image
import uuid
from pymilvus import MilvusClient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class EmbeddingStore:
    def __init__(self, host='localhost', port='19530', collection_name='person_embeddings'):

        # Define schema and create collection if it doesn't exist
        self.collection_name = collection_name
        self.client =  MilvusClient(
            uri=f"http://{host}:{port}",
            token="root:Milvus"
        )
        
    def get_embedding_by_keys(self,keys):
        query_expr = "id in [{}]".format(', '.join(['"{}"'.format(key) for key in keys]))
        results = self.client.query(
            collection_name=self.collection_name,
            filter=query_expr,
        )
        results = list(results)
        embeddings = [result["embedding"] for result in results]
        return embeddings

