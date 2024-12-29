import unittest
import numpy as np
import tensorflow as tf
import tempfile
import os
from pathlib import Path
from models.ml_model import MLModel

class TestMLModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.input_shape = (60, 25)  # (lookback_period, n_features)
        self.horizon = 5
        self.model = MLModel(input_shape=self.input_shape, horizon=self.horizon)
        
        # Create sample data
        self.batch_size = 32
        self.n_samples = 100
        self.X_train = np.random.random((self.n_samples, *self.input_shape))
        self.y_train = np.random.random((self.n_samples, self.horizon * 2))  # Returns and directions
        self.X_val = np.random.random((20, *self.input_shape))
        self.y_val = np.random.random((20, self.horizon * 2))
        
        # Create temporary directory for test artifacts
        self.test_dir = tempfile.mkdtemp()
        self.logs_dir = Path(self.test_dir) / 'logs'
        self.models_dir = Path(self.test_dir) / 'models'
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test artifacts."""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_model_initialization(self):
        """Test if model is initialized correctly with given input shape."""
        self.assertIsNotNone(self.model)
        self.assertIsInstance(self.model.model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(self.model.model.input_shape[1:], self.input_shape)
        
        # Check output shape
        self.assertEqual(self.model.model.output_shape[-1], self.horizon * 2)
        
        # Test initialization with invalid input shape
        with self.assertRaises(ValueError):
            MLModel(input_shape=(0, 25), horizon=self.horizon)
        
        # Test initialization with invalid horizon
        with self.assertRaises(ValueError):
            MLModel(input_shape=self.input_shape, horizon=0)

    def test_model_compilation(self):
        """Test if model is compiled with correct optimizer and loss."""
        self.assertIsNotNone(self.model.model.optimizer)
        self.assertIsInstance(self.model.model.optimizer, tf.keras.optimizers.Adam)
        
        # Check if loss and metrics are set
        self.assertTrue(callable(self.model.model.loss))  # Custom combined loss
        self.assertEqual(len(self.model.model.metrics), 2)  # mae and direction_accuracy
        
        # Test optimizer parameters
        self.assertEqual(self.model.model.optimizer.learning_rate.numpy(), 
                        self.model.config['learning_rate'])

    def test_direction_accuracy_metric(self):
        """Test the direction accuracy metric calculation."""
        # Test perfect direction prediction
        y_true = tf.constant([[1.0, 1.0], [-1.0, -1.0], [0.5, 0.5], [-0.5, -0.5]])
        y_pred = tf.constant([[2.0, 2.0], [-2.0, -2.0], [1.0, 1.0], [-1.0, -1.0]])
        accuracy = self.model._direction_accuracy(y_true, y_pred)
        self.assertEqual(accuracy.numpy(), 1.0)
        
        # Test opposite direction prediction
        y_pred_opposite = -y_pred
        accuracy = self.model._direction_accuracy(y_true, y_pred_opposite)
        self.assertEqual(accuracy.numpy(), 0.0)
        
        # Test mixed direction prediction
        y_pred_mixed = tf.constant([[2.0, 2.0], [2.0, 2.0], [-1.0, -1.0], [1.0, 1.0]])
        accuracy = self.model._direction_accuracy(y_true, y_pred_mixed)
        self.assertEqual(accuracy.numpy(), 0.5)

    def test_returns_loss(self):
        """Test the returns loss calculation."""
        y_true = tf.constant([[1.0, 0.0], [-1.0, 0.0]])
        y_pred = tf.constant([[1.0, 0.0], [-1.0, 0.0]])
        loss = self.model._returns_loss(y_true, y_pred)
        self.assertEqual(loss.numpy(), 0.0)
        
        # Test with different predictions
        y_pred_diff = tf.constant([[2.0, 0.0], [-2.0, 0.0]])
        loss = self.model._returns_loss(y_true, y_pred_diff)
        self.assertGreater(loss.numpy(), 0.0)

    def test_directions_loss(self):
        """Test the directions loss calculation."""
        y_true = tf.constant([[1.0, 1.0], [-1.0, -1.0]])
        y_pred = tf.constant([[1.0, 1.0], [-1.0, -1.0]])
        loss = self.model._directions_loss(y_true, y_pred)
        self.assertAlmostEqual(loss.numpy(), 0.0, places=5)
        
        # Test with different predictions
        y_pred_diff = tf.constant([[-1.0, -1.0], [1.0, 1.0]])
        loss = self.model._directions_loss(y_true, y_pred_diff)
        self.assertGreater(loss.numpy(), 0.0)

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
        
        # Test early stopping
        history = self.model.train(
            self.X_train, self.y_train,
            X_val=self.X_val, y_val=self.y_val,
            epochs=100,  # Large number of epochs
            batch_size=self.batch_size
        )
        self.assertLess(len(history['loss']), 100)  # Should stop early

    def test_model_prediction(self):
        """Test model prediction functionality."""
        # Train model with minimal epochs
        self.model.train(self.X_train, self.y_train, epochs=1, batch_size=self.batch_size)
        
        # Test prediction
        predictions = self.model.predict(self.X_val)
        
        self.assertEqual(predictions.shape, (len(self.X_val), self.horizon * 2))
        self.assertTrue(np.all(np.isfinite(predictions)))  # Check for NaN or inf values
        
        # Test prediction with single sample
        single_pred = self.model.predict(self.X_val[0:1])
        self.assertEqual(single_pred.shape, (1, self.horizon * 2))
        
        # Test prediction with empty input
        with self.assertRaises(ValueError):
            self.model.predict(np.array([]))

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
        
        # Test with mismatched samples
        wrong_samples_X = np.random.random((50, *self.input_shape))
        with self.assertRaises(ValueError):
            self.model.train(wrong_samples_X, self.y_train)
        
        # Test with NaN values
        X_with_nan = self.X_train.copy()
        X_with_nan[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.model.train(X_with_nan, self.y_train)
        
        # Test with infinite values
        X_with_inf = self.X_train.copy()
        X_with_inf[0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            self.model.train(X_with_inf, self.y_train)

    def test_model_save_load(self):
        """Test model saving and loading functionality."""
        # Train model
        self.model.train(self.X_train, self.y_train, epochs=1, batch_size=self.batch_size)
        
        # Save model
        model_path = self.models_dir / 'test_model.keras'
        self.model.model.save(model_path)
        
        # Load model
        loaded_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                '_combined_loss': self.model._combined_loss,
                '_direction_accuracy': self.model._direction_accuracy
            }
        )
        
        # Compare predictions
        original_pred = self.model.predict(self.X_val)
        loaded_pred = loaded_model.predict(self.X_val)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

if __name__ == '__main__':
    unittest.main()
