from itertools import chain
import json
import os
from aiokafka import AIOKafkaConsumer
import redis

from detection_association_service.Embedders import Encoder
from detection_association_service.Milvus_manager.embedding_store import EmbeddingStore
from detection_association_service.faiss_manager.faiss_manager import FaissManager
from detection_association_service.redis_manager import GraphRedisManager, RedisManager
from .producer import KafkaProducerService
from v3.frame_metadata import FrameMetadata
from v3.detected_person_metadata import DetectionMetadata
class KafkaConsumerService:
    def __init__(self, bootstrap_servers='192.168.111.131:9092', topic='person_detection_requests'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer = None  # Initialize the consumer as None
        self.encoder = Encoder()
        self.frames_metadata_manager_client = RedisManager(db=1)
        self.persons_metadata_manager_client = RedisManager(db=2)
        self.frames_data_manager_client = RedisManager(db=0)
        self.graph_redis_manager = GraphRedisManager()
        self.milvus_manager = EmbeddingStore()
        self.faiss_manager = FaissManager()
        self.producer = KafkaProducerService()
        self.therehold = 1.5
    async def start(self):
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))

        )
        await self.consumer.start()

        await self.producer.start()

        try:
            await self.consume_messages()
        finally:
            await self.consumer.stop()
            await self.producer.close()

    async def consume_messages(self):
        async for message in self.consumer:
            print("There is a message")
            frame_key = message.value
            camera_id = ":".join(message.split(":")[1:2])
            time_stamp = ":".join(message.split(":")[2:3])


            frame_metadata = self.frames_metadata_manager_client.get_one(str(frame_key))

            frame_metadata = {k.decode(): v.decode() if isinstance(v, bytes) else v for k, v in frame_metadata.items()}
            frame_metadata["detected_persons"] = json.loads(frame_metadata["detected_persons"])
            frame_metadata = FrameMetadata(**frame_metadata)
            detection_keys = [f"metadata:{frame_key}:{person_id}" for person_id in frame_metadata.detected_persons]
 
            detections_metadata_bytes = self.persons_metadata_manager_client.get_many(detection_keys)

            detection = []
            apearance_embedding_keys = []
            for det in detections_metadata_bytes : 
                detection_data = {k.decode(): v.decode() if isinstance(v, bytes) else v for k, v in det.items()}
                detection_data["bbox"] = json.loads(detection_data["bbox"])
                detection_metadata = DetectionMetadata(**detection_data)
                apearance_embedding_keys.append(detection_metadata.embedding_key)
                detection.append(detection_metadata)

            embeddings = self.milvus_manager.get_embedding_by_keys(apearance_embedding_keys)

            detection_embeddings = self.encoder(time_stamp=time_stamp,camera_id=camera_id,embeddings=embeddings)

            results = self.faiss_manager.query_faiss_many(detection_embeddings) 

            update_faiss_requst = []
            insert_faiss_request = []
            graph_association_update = []
            face_auth_request = []

            for i , (distance, person_id, whole_vector) in enumerate(results) :

                if person_id is not None and distance < self.therehold : 
                    #  assoicate it  to old cluster 
                    authFlag = self.faiss_manager.get_auth_flag(person_id)
                    updated_cluster_embedding = self.encoder.update_centroid(whole_vector,detection_embeddings[i],distance)
                    update_faiss_requst.append((person_id,updated_cluster_embedding))
                    graph_association_update.append((person_id,detection[i]))
                    if not authFlag and detection[i].is_face_clear :
                        face_auth_request.append((person_id,detection[i]))
                        authFlag= True
                    update_faiss_requst.append((person_id,updated_cluster_embedding,authFlag))
                else :
                    #  create a new cluster 
                    new_person_id = detection[i].person_key

                    if detection[i].is_face_clear:
                        face_auth_request.append((new_person_id,detection[i]))
                        authFlag = True
                    else : 
                        authFlag = False

                    insert_faiss_request.append((new_person_id,detection_embeddings[i],authFlag))
                    graph_association_update.append((new_person_id,detection[i]))

            self.faiss_manager.update_faiss(update_faiss_requst,insert_faiss_request)

            self.graph_redis_manager.update_graph(graph_association_update)

            self.producer.send_auth_request(face_auth_request)
