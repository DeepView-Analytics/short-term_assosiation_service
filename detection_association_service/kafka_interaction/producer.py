import json
import logging
import operator
import os
from aiokafka import AIOKafkaProducer
from typing import List, Tuple
from math import ceil
from v3.partitioneddetectionbatch import PartitionedDetectionBatch


class KafkaProducerService:
    def __init__(self, bootstrap_servers='127.0.0.1:9092', topic='person_detected_response'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None  # Initialize the producer as None
        self.max_bboxes_per_batch = int(os.getenv('RESPONSE_BATCH_SIZE', 10))


    async def start(self):
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: v.encode('utf-8')
            )
            await self.producer.start()
        except Exception as e:
            logging.error(f"Failed to start Kafka producer: {e}", exc_info=True)
            raise e


    async def send_auth_request(self, face_auth_request):
        
        request = []

        for person_id , detection in face_auth_request:
            face_bbox = detection.face_bbox.model_dump()
            face_bbox = json.dumps(detection.face_bbox)
            request.append((person_id,detection.frame_key,face_bbox))
        request = json.dumps(request)
        try:
            print(f"The topic target is: {self.topic}")
            future = await self.producer.send_and_wait(self.topic, request)
            print(f"Message sent successfully to topic {self.topic}")
        except Exception as e:
            logging.error(f"Failed to send message: {e}", exc_info=True)

    async def close(self):
        await self.producer.stop()