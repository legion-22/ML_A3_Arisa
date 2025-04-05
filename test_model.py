import unittest
import numpy as np
from LogisticRegression import LogisticRegression

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        # Create model with weight like mock
        self.model = LogisticRegression(k=4, n=5, method='batch', use_penalty=False)
        self.model.W = np.random.rand(5, 4)

    def test_model_input(self):
        X_sample = np.random.rand(10, 5)  # (m=10, n=5)
        try:
            self.model.predict(X_sample)
        except Exception as e:
            self.fail(f"Model failed with valid input: {e}")

    def test_model_output_shape(self):
        X_sample = np.random.rand(7, 5)
        pred = self.model.predict(X_sample)
        self.assertEqual(pred.shape, (7,))