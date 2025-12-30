
import unittest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock
sys.modules['prefetch_generator'] = MagicMock()

from lib.utils.utils import xyxy2xywh

class TestUtils(unittest.TestCase):
    def test_xyxy2xywh_numpy(self):
        # [x1, y1, x2, y2] -> [x, y, w, h]
        input_box = np.array([[0, 0, 10, 10]], dtype=np.float32)
        expected_output = np.array([[5, 5, 10, 10]], dtype=np.float32)
        
        output = xyxy2xywh(input_box)
        np.testing.assert_array_equal(output, expected_output)

    def test_xyxy2xywh_torch(self):
        # [x1, y1, x2, y2] -> [x, y, w, h]
        input_box = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        expected_output = torch.tensor([[5, 5, 10, 10]], dtype=torch.float32)
        
        output = xyxy2xywh(input_box)
        self.assertTrue(torch.equal(output, expected_output))

    def test_xyxy2xywh_multiple(self):
        input_boxes = np.array([
            [0, 0, 10, 10],
            [10, 10, 20, 30]
        ], dtype=np.float32)
        # 1: x_c=5, y_c=5, w=10, h=10
        # 2: x_c=15, y_c=20, w=10, h=20
        expected_output = np.array([
            [5, 5, 10, 10],
            [15, 20, 10, 20]
        ], dtype=np.float32)
        
        output = xyxy2xywh(input_boxes)
        np.testing.assert_array_equal(output, expected_output)

if __name__ == '__main__':
    unittest.main()
