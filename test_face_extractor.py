import unittest
import numpy as np
# Assuming face_extractor.py is in the same directory or accessible in PYTHONPATH
from face_extractor import is_unique_face

class TestFaceExtractor(unittest.TestCase):

    def test_is_unique_face(self):
        known_encodings = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6])
        ]

        # Test case 1: New encoding is significantly different (should be unique)
        new_encoding_unique = np.array([0.7, 0.8, 0.9])
        self.assertTrue(is_unique_face(known_encodings, new_encoding_unique, tolerance=0.5))

        # Test case 2: New encoding is very similar to an existing one (should not be unique)
        new_encoding_not_unique = np.array([0.1, 0.21, 0.32]) # Close to known_encodings[0]
        # Calculate distance: np.linalg.norm(known_encodings[0] - new_encoding_not_unique) -> should be small
        self.assertFalse(is_unique_face(known_encodings, new_encoding_not_unique, tolerance=0.1))

        # Test case 3: New encoding is different but within a larger tolerance (should not be unique)
        new_encoding_within_tolerance = np.array([0.15, 0.25, 0.35])
        self.assertFalse(is_unique_face(known_encodings, new_encoding_within_tolerance, tolerance=0.3))

        # Test case 4: New encoding is different and outside a tighter tolerance (should be unique)
        self.assertTrue(is_unique_face(known_encodings, new_encoding_within_tolerance, tolerance=0.05))

        # Test case 5: Empty known_encodings list (any new encoding should be unique)
        empty_known_encodings = []
        self.assertTrue(is_unique_face(empty_known_encodings, new_encoding_unique, tolerance=0.5))

if __name__ == '__main__':
    unittest.main()
