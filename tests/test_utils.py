
import unittest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mocking dependencies that might not be available in all environments
from unittest.mock import MagicMock
sys.modules['prefetch_generator'] = sys.modules['yaml'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.cluster'] = MagicMock()
sys.modules['scipy.cluster.vq'] = MagicMock()
sys.modules['scipy.signal'] = MagicMock()
sys.modules['tqdm'] = MagicMock()

from lib.utils.utils import xyxy2xywh, clean_str, is_parallel
from lib.utils.augmentations import _box_candidates


class TestUtils(unittest.TestCase):
    """Unit tests for utility functions in lib.utils.utils and lib.utils.augmentations"""
    def test_xyxy2xywh_numpy(self):
        # [x1, y1, x2, y2] -> [x_center, y_center, width, height]
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

    def test_xyxy2xywh_empty(self):
        # Handle cases where no boxes are detected
        input_boxes = np.zeros((0, 4), dtype=np.float32)
        expected_output = np.zeros((0, 4), dtype=np.float32)
        
        output = xyxy2xywh(input_boxes)
        np.testing.assert_array_equal(output, expected_output)

    def test_clean_str(self):
        input_str = "hello@world#test"
        expected_output = "hello_world_test"
        output = clean_str(input_str)
        self.assertEqual(output, expected_output)

        input_str2 = "data/images!"
        # Note: '!' is in the regex pattern [|@#!¡·$€%&()=?¿^*;:,¨´><+]
        expected_output2 = "data/images_"
        output2 = clean_str(input_str2)
        self.assertEqual(output2, expected_output2)

    def test__box_candidates(self):
        # box1: [0, 0, 10, 10], box2: [0, 0, 8, 8]
        # w1=10, h1=10, w2=8, h2=8
        # area1=100, area2=64, ratio=0.64
        # ar = 8/8 = 1
        box1 = np.array([[0, 0, 10, 10]]).T
        box2 = np.array([[0, 0, 8, 8]]).T
        
        # wh_thr=2, ar_thr=20, area_thr=0.1
        # w2 > 2 (8>2), h2 > 2 (8>2), area_ratio > 0.1 (0.64 > 0.1), ar < 20 (1 < 20) -> True
        output = _box_candidates(box1, box2)
        self.assertTrue(output[0])

        # Test failure due to aspect ratio
        box2_bad_ar = np.array([[0, 0, 25, 1]]).T
        # w2=25, h2=1, ar=25. 25 < 20 is False.
        output = _box_candidates(box1, box2_bad_ar)
        self.assertFalse(output[0])

    def test_is_parallel(self):
        model = torch.nn.Linear(10, 10)
        self.assertFalse(is_parallel(model))
        
        # Mock DataParallel
        dp_model = torch.nn.DataParallel(model)
        self.assertTrue(is_parallel(dp_model))

if __name__ == '__main__':
    unittest.main()
