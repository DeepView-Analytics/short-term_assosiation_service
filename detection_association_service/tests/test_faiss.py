import sys
import unittest
import numpy as np
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faiss_manager.faiss_manager import FaissManager

class TestFaissManager(unittest.TestCase):

    def setUp(self):
        self.dimension = 128
        self.index_path = "test_faiss_index.index"
        self.mapping_path = "test_id_mapping.pkl"
        self.manager = FaissManager(dimension=self.dimension, index_path=self.index_path, mapping_path=self.mapping_path)

        # Cleanup any existing test files
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.mapping_path):
            os.remove(self.mapping_path)

    def tearDown(self):
        # Cleanup test files
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.mapping_path):
            os.remove(self.mapping_path)

    def test_save_vector(self):
        vector = [0.1] * self.dimension
        self.manager.save_vector(1, vector)
        results = self.manager.search_vector(vector, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)

    def test_delete_vector(self):
        vector = [0.1] * self.dimension
        self.manager.save_vector(1, vector)
        self.manager.delete_vector(1)
        results = self.manager.search_vector(vector, top_k=1)
        self.assertEqual(len(results), 0)

    def test_edit_vector(self):
        vector = [0.1] * self.dimension
        new_vector = [0.2] * self.dimension
        self.manager.save_vector(1, vector)
        self.manager.edit_vector(1, new_vector)
        results = self.manager.search_vector(new_vector, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)

    def test_search_vector_no_results(self):
        query_vector = [0.5] * self.dimension
        results = self.manager.search_vector(query_vector, top_k=1)
        self.assertEqual(len(results), 0)

    def test_save_duplicate_id(self):
        vector = [0.1] * self.dimension
        self.manager.save_vector(1, vector)
        with self.assertRaises(ValueError):
            self.manager.save_vector(1, vector)

    def test_edit_nonexistent_id(self):
        new_vector = [0.2] * self.dimension
        with self.assertRaises(ValueError):
            self.manager.edit_vector(1, new_vector)

    def test_delete_nonexistent_id(self):
        self.manager.delete_vector(999)  # Should not raise an exception

    def test_save_and_reload(self):
        vector = [0.1] * self.dimension
        self.manager.save_vector(1, vector)

        # Reload manager
        new_manager = FaissManager(dimension=self.dimension, index_path=self.index_path, mapping_path=self.mapping_path)
        results = new_manager.search_vector(vector, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)

if __name__ == "__main__":
    unittest.main()
