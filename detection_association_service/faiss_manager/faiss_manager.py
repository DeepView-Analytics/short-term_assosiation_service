import time
import redis
import faiss
import json
import numpy as np
from typing import List, Tuple
from redis.exceptions import LockError

class FaissManager:
    def __init__(self, redis_host="localhost", redis_port=6379, dimension=128, window=10, db=8):
        """
        Initialize the FaissManager with Redis as the backend store.

        :param redis_host: Redis server host.
        :param redis_port: Redis server port.
        :param dimension: Dimensionality of the embeddings.
        :param window: Time window to remove outdated vectors.
        """
        self.dimension = dimension
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True, db=db)
        self.window = window
        self.lock_timeout = 10
        self.index_key = "faiss_index"
        self.map_key = "person_map"
        self.lock_key = "faiss_lock"
        self.index = None
        self.person_map = {}
        self.load_from_redis()

    def load_from_redis(self):
        """Load FAISS index and person map from Redis."""
        try:
            # Load index
            index_data = self.redis_client.get(self.index_key)
            if index_data:
                self.index = faiss.deserialize_index(np.frombuffer(index_data, dtype=np.uint8))
                print("FAISS index loaded from Redis.")
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
                print("Initialized a new FAISS index.")

            # Load person map
            map_data = self.redis_client.get(self.map_key)
            if map_data:
                self.person_map = json.loads(map_data)
                print("Person map loaded from Redis.")
            else:
                self.person_map = {}
                print("Initialized a new person map.")
        except Exception as e:
            print(f"Error loading data from Redis: {e}")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.person_map = {}

    def save_to_redis(self):
        """Save FAISS index and person map to Redis."""
        try:
            # Save index
            index_data = faiss.serialize_index(self.index).tobytes()
            self.redis_client.set(self.index_key, index_data)

            # Save person map
            self.redis_client.set(self.map_key, json.dumps(self.person_map))
            print("FAISS index and person map saved to Redis.")
        except Exception as e:
            print(f"Error saving data to Redis: {e}")

    def acquire_lock(self):
        """Acquire a Redis lock, retrying until it becomes available."""
        while True:
            lock = self.redis_client.lock(self.lock_key, timeout=self.lock_timeout)
            if lock.acquire(blocking=True):
                return lock
            time.sleep(0.1)  # Small delay to prevent tight loops

    def release_lock(self, lock):
        """Release the acquired Redis lock."""
        try:
            lock.release()
        except Exception as e:
            print(f"Error releasing lock: {e}")

    def update_faiss(self, update_request: List[Tuple[str, np.ndarray, bool]], insert_request: List[Tuple[str, np.ndarray, bool]]):
        """Update and insert embeddings into the FAISS index."""
        try:
            lock = self.acquire_lock()  # Acquire the lock before proceeding
            # Rebuild vectors from current index
            vectors = []
            for i in range(len(self.person_map)):
                vectors.append(self.index.reconstruct(i))

            # Update or remove outdated vectors
            current_time = time.time()
            indexes_to_remove = []
            for i, (person_id, data) in enumerate(self.person_map.items()):
                if current_time - data['last_update'] > self.window:
                    indexes_to_remove.append(i)

            vectors = [v for i, v in enumerate(vectors) if i not in indexes_to_remove]
            self.person_map = {k: v for i, (k, v) in enumerate(self.person_map.items()) if i not in indexes_to_remove}

            # Process updates
            for person_id, new_vector, authFlag in update_request:
                if person_id in self.person_map:
                    idx = list(self.person_map.keys()).index(person_id)
                    vectors[idx] = new_vector
                    self.person_map[person_id]['last_update'] = current_time
                    self.person_map[person_id]['authFlag'] = authFlag
                else:
                    print(f"Person ID {person_id} not found. Skipping update.")

            # Process inserts
            for person_id, vector, authFlag in insert_request:
                if person_id not in self.person_map:
                    vectors.append(vector)
                    self.person_map[person_id] = {
                        "last_update": current_time,
                        "authFlag": authFlag
                    }

            # Rebuild index
            self.index.reset()
            self.index.add(np.array(vectors))
            self.save_to_redis()
        except LockError:
            print("Failed to acquire Redis lock. Try again later.")
        finally:
            self.release_lock(lock)  # Ensure the lock is released after operation

    def query_faiss_many(self, vectors: List[np.ndarray], k=1) -> List[Tuple[float, str]]:
        """Query the FAISS index for the top-k nearest neighbors."""
        lock = self.acquire_lock()  # This will wait until the lock becomes available
        try:
            vectors = np.array(vectors).astype("float32")
            distances, indices = self.index.search(vectors, k)
            results = []
            for i in range(len(vectors)):
                for j in range(k):
                    idx = int(indices[i, j])
                    if idx >= 0 and idx < len(self.person_map):
                        person_id = list(self.person_map.keys())[idx]
                        distance = distances[i, j]
                        results.append((distance, person_id))
            return results
        finally:
            self.release_lock(lock)  # Ensure the lock is released after query

    def get_auth_flag(self, person_id):

            # Check if the person_id exists in the person_map
            if person_id in self.person_map:
                return self.person_map[person_id].get('authFlag', None)
            else:
                print(f"Person ID {person_id} not found.")
                return None
        
# Example usage and test:
if __name__ == "__main__":
    manager = FaissManager()


    # Insert 3 person IDs with vectors
    insert_request = [
        ("person_5", np.random.rand(128).astype('float32'), False),
        ("person_6", np.random.rand(128).astype('float32'), False),
        ("person_7", np.random.rand(128).astype('float32'), False)
    ]
    manager.update_faiss([], insert_request)
  

    # Sleep for 30 seconds
    time.sleep(2)

    # Update one of the inserted person IDs
    update_request = [("person_6", np.random.rand(128).astype('float32') , False)]
    manager.update_faiss(update_request, [])


    # Sleep for another 30 seconds
    time.sleep(2)

    # Insert another person ID with vector
    insert_request = [("person_8", np.random.rand(128).astype('float32') ,False)]
    manager.update_faiss([], insert_request)




     # Query the FAISS index
    query_vectors = [np.random.rand(128).astype('float32'), np.random.rand(128).astype('float32')]
    results = manager.query_faiss_many(query_vectors, k=3)

    for result in results:
        print("------------------------------------")
        print(result)
