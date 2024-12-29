import unittest
import numpy as np
import tensorflow as tf
from models.ml_model import MLModel

class TestMLModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.input_shape = (60, 25)  # (lookback_period, n_features)
        self.output_shape = 5
        self.model = MLModel(input_shape=self.input_shape, output_shape=self.output_shape)
        
        # Create sample data
        self.batch_size = 32
        self.n_samples = 100
        self.X_train = np.random.random((self.n_samples, *self.input_shape))
        self.y_train = np.random.random((self.n_samples, self.output_shape))
        self.X_val = np.random.random((20, *self.input_shape))
        self.y_val = np.random.random((20, self.output_shape))

    def test_model_initialization(self):
        """Test if model is initialized correctly with given input shape."""
        self.assertIsNotNone(self.model)
        self.assertIsInstance(self.model.model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(self.model.model.input_shape[1:], self.input_shape)
        
        # Check output shape
        self.assertEqual(self.model.model.output_shape[-1], self.output_shape)

    def test_model_compilation(self):
        """Test if model is compiled with correct optimizer and loss."""
        self.assertIsNotNone(self.model.model.optimizer)
        self.assertIsInstance(self.model.model.optimizer, tf.keras.optimizers.Adam)
        
        # Check if loss and metrics are set
        self.assertEqual(self.model.model.loss, 'huber')
        self.assertEqual(len(self.model.model.metrics), 2)  # mae and direction_accuracy

    def test_direction_accuracy_metric(self):
        """Test the direction accuracy metric calculation."""
        y_true = tf.constant([[1.0], [-1.0], [0.5], [-0.5]])
        y_pred = tf.constant([[2.0], [-2.0], [1.0], [-1.0]])
        
        accuracy = self.model._direction_accuracy(y_true, y_pred)
        self.assertEqual(accuracy.numpy(), 1.0)  # All predictions have correct direction

    def test_model_training(self):
        """Test model training with and without validation data."""
        # Test training with validation data
        history = self.model.train(
            self.X_train, self.y_train,
            X_val=self.X_val, y_val=self.y_val,
            epochs=2, batch_size=self.batch_size
        )
        
        self.assertIn('loss', history)
        self.assertIn('val_loss', history)
        self.assertIn('mae', history)
        self.assertIn('direction_accuracy', history)
        
        # Test training without validation data
        history = self.model.train(
            self.X_train, self.y_train,
            epochs=2, batch_size=self.batch_size
        )
        
        self.assertIn('loss', history)
        self.assertIn('mae', history)
        self.assertIn('direction_accuracy', history)

    def test_model_prediction(self):
        """Test model prediction functionality."""
        # Train model with minimal epochs
        self.model.train(self.X_train, self.y_train, epochs=1, batch_size=self.batch_size)
        
        # Test prediction
        predictions = self.model.predict(self.X_val)
        
        self.assertEqual(predictions.shape, (len(self.X_val), self.output_shape))
        self.assertTrue(np.all(np.isfinite(predictions)))  # Check for NaN or inf values

    def test_input_validation(self):
        """Test input validation for training data."""
        # Test with wrong input shape
        wrong_shape_X = np.random.random((self.n_samples, 30, 25))  # Wrong timesteps
        with self.assertRaises(ValueError):
            self.model.train(wrong_shape_X, self.y_train)
        
        # Test with wrong output shape
        wrong_shape_y = np.random.random((self.n_samples, 3))  # Wrong output dimension
        with self.assertRaises(ValueError):
            self.model.train(self.X_train, wrong_shape_y)

if __name__ == '__main__':
    unittest.main()
