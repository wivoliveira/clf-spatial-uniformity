import unittest
import numpy as np
from clf_spatial_uniformity import uniformity_index

class TestUniformityIndex(unittest.TestCase):

    def test_uniformity_index(self):
        array = np.array([
            [1, 1, 1, 2, 2],
            [1, 1, 1, 2, 2],
            [1, 1, 1, 2, 2],
            [2, 2, 2, 3, 3],
            [2, 2, 2, 3, 3],
        ])
        uniformity_threshold = 2
        result = uniformity_index(array, uniformity_threshold)
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], dict)
        self.assertIsInstance(result[1], np.ndarray)

if __name__ == '__main__':
    unittest.main()
