# Machine learning model implementation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional, Input, Attention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import logging
from .base_model import BaseModel


logger = logging.getLogger(__name__)


class MLModel(BaseModel):
    def __init__(self, config: 'Config'):
        try:
            super().__init__(config)
            self.model_type = config.model.model_type
            self.epochs = config.model.epochs
            self.batch_size = config.model.batch_size
            self.lookback_period = config.model.lookback_period
            self.build_model()
        except Exception as e:
            logger.exception("Error initializing MLModel")
            raise e
        
    def build_model(self) -> None:
        """Build and compile the model based on configuration."""
        try:
            input_shape = (self.lookback_period, len(self.config.model.features))
            
            if self.model_type == 'lstm':
                self.model = self._build_lstm_model(input_shape)
            elif self.model_type == 'gru':
                self.model = self._build_gru_model(input_shape)
            elif self.model_type == 'transformer':
                self.model = self._build_transformer_model(input_shape)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Use fixed learning rate from config
            self.model.compile(
                optimizer=Adam(learning_rate=self.config.model.learning_rate),
                loss=self._custom_loss,
                metrics=['mae', self._direction_accuracy]
            )
            
            logger.info(f"Built {self.model_type.upper()} model")
            logger.info(self.get_model_summary())
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
            
    def _build_lstm_model(self, input_shape: tuple) -> tf.keras.Model:
        model = Sequential()
        
        # First LSTM layer with input_shape=(timesteps, features)
        model.add(LSTM(64, input_shape=(self.config.model.lookback_period, 12), return_sequences=True, name="lstm"))
        model.add(LayerNormalization(name="layer_normalization"))
        model.add(Dropout(0.2, name="dropout"))
        
        # Second LSTM layer returns sequences
        model.add(LSTM(64, return_sequences=True, name="lstm_1"))
        model.add(LayerNormalization(name="layer_normalization_1"))
        model.add(Dropout(0.2, name="dropout_1"))
        
        # Third LSTM layer, returns final state
        model.add(LSTM(64, name="lstm_2"))
        model.add(LayerNormalization(name="layer_normalization_2"))
        model.add(Dropout(0.2, name="dropout_2"))
        
        # Dense layers
        model.add(Dense(128, activation='relu', name="dense"))
        model.add(Dense(64, activation='relu', name="dense_1"))
        model.add(Dropout(0.2, name="dropout_3"))
        
        # Final output layer for 12 signals (or however many outputs you need)
        model.add(Dense(self.config.model.prediction_horizon, activation='linear', name="dense_2"))
        
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def _build_gru_model(self, input_shape: tuple) -> Sequential:
        """Build a GRU model with bidirectional layers."""
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(GRU(units=64, return_sequences=True)),
            LayerNormalization(),
            Dropout(0.2),
            Bidirectional(GRU(units=32, return_sequences=False)),
            LayerNormalization(),
            Dropout(0.2),
            Dense(self.config.model.prediction_horizon)
        ])
        return model
        
    def _build_transformer_model(self, input_shape: tuple) -> Sequential:
        """Build a Transformer model."""
        model = Sequential([
            Input(shape=input_shape),
            # Add transformer layers here
            Dense(self.config.model.prediction_horizon)
        ])
        return model
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the ML model."""
        try:
            logger.info(f"Training model with {len(X_train)} samples for {self.epochs} epochs")
            
            # Validate input data types
            if not isinstance(X_train, np.ndarray):
                raise ValueError("X_train must be a NumPy array.")
            if not isinstance(y_train, np.ndarray):
                raise ValueError("y_train must be a NumPy array.")
            if X_val is not None and not isinstance(X_val, np.ndarray):
                raise ValueError("X_val must be a NumPy array if provided.")
            if y_val is not None and not isinstance(y_val, np.ndarray):
                raise ValueError("y_val must be a NumPy array if provided.")
            
            # Validate shapes
            if len(X_train.shape) != 3:
                raise ValueError(f"X_train must be 3D (samples, timesteps, features), but got {X_train.shape}")
            if len(y_train.shape) != 2:
                raise ValueError(f"y_train must be 2D (samples, prediction_horizon), but got {y_train.shape}")
            if X_val is not None and len(X_val.shape) != 3:
                raise ValueError(f"X_val must be 3D (samples, timesteps, features), but got {X_val.shape}")
            if y_val is not None and len(y_val.shape) != 2:
                raise ValueError(f"y_val must be 2D (samples, prediction_horizon), but got {y_val.shape}")

            # Debugging: Log the shapes of the data
            logger.debug(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            if X_val is not None and y_val is not None:
                logger.debug(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

            # Log training configurations
            logger.info(f"Epochs: {self.epochs}, Batch Size: {self.batch_size}")
            logger.info(f"Validation Data: {'Provided' if X_val is not None and y_val is not None else 'Not Provided'}")

            # Define callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    min_delta=1e-5
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,
                    patience=5,
                    min_lr=1e-7,
                    min_delta=1e-5
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'logs/best_model.keras',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False
                )
            ]
            
            if X_val is None or y_val is None:
                callbacks = callbacks[:-1]  # Remove ModelCheckpoint if no validation data

            # Train the model
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                callbacks=callbacks,
                verbose=1
            )

            self.is_trained = True
            return history.history

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with uncertainty estimation."""
        try:
            # Validate input data
            is_valid, message = self.validate_data(X)
            if not is_valid:
                raise ValueError(message)
                
            if not self.check_is_trained():
                raise ValueError("Model must be trained before making predictions")
                
            # Make predictions with Monte Carlo Dropout
            predictions = []
            for _ in range(10):  # Number of Monte Carlo samples
                pred = self.model(X, training=True)  # Enable dropout during inference
                predictions.append(pred)
                
            # Calculate mean and standard deviation
            predictions = np.stack(predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # Add uncertainty information to the predictions
            return mean_pred, std_pred
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
            
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model with detailed metrics."""
        try:
            if not self.check_is_trained():
                raise ValueError("Model must be trained before evaluation")
                
            # Get predictions with uncertainty
            y_pred, y_std = self.predict(X_test)
            
            # Calculate metrics
            metrics = {}
            
            # Basic metrics
            metrics.update(self.model.evaluate(X_test, y_test, return_dict=True))
            
            # Direction accuracy
            pred_direction = np.sign(y_pred)
            true_direction = np.sign(y_test)
            metrics['direction_accuracy'] = np.mean(pred_direction == true_direction)
            
            # Uncertainty metrics
            metrics['mean_uncertainty'] = np.mean(y_std)
            metrics['max_uncertainty'] = np.max(y_std)
            
            # Custom metrics for financial predictions
            metrics['rmse'] = np.sqrt(np.mean((y_pred - y_test) ** 2))
            metrics['mape'] = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
        
    def _custom_loss(self, y_true, y_pred):
        """Custom loss function combining MSE and directional accuracy with return optimization."""
        # MSE component using tf.keras backend
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Direction component
        direction_true = tf.sign(y_true)
        direction_pred = tf.sign(y_pred)
        direction_loss = tf.cast(tf.not_equal(direction_true, direction_pred), tf.float32)
        
        # Return magnitude component - penalize more for missing big moves
        return_magnitude = tf.abs(y_true)
        magnitude_loss = tf.reduce_mean(return_magnitude * tf.square(y_true - y_pred))
        
        # Combine losses with weighting
        return mse_loss + 0.2 * direction_loss + 0.3 * magnitude_loss
        
    def _direction_accuracy(self, y_true, y_pred):
        """Calculate directional accuracy metric."""
        direction_true = tf.sign(y_true)
        direction_pred = tf.sign(y_pred)
        return tf.reduce_mean(tf.cast(tf.equal(direction_true, direction_pred), tf.float32))

    def validate_data(self, X: np.ndarray, y: np.ndarray = None) -> tuple:
        """Validate input data."""
        try:
            if len(X.shape) != 3:
                return False, "Input data must be 3-dimensional (samples, timesteps, features)"
            
            if X.shape[1] != self.config.model.lookback_period:
                return False, f"Input timesteps must be {self.config.model.lookback_period}"
            
            if X.shape[2] != len(self.config.model.features):
                return False, f"Input features must be {len(self.config.model.features)}"
            
            if y is not None:
                if len(y.shape) != 2:
                    return False, "Target data must be 2-dimensional (samples, prediction_horizon)"
                
                if y.shape[0] != X.shape[0]:
                    return False, "Number of samples must match between X and y"
                
                if y.shape[1] != self.config.model.prediction_horizon:
                    return False, f"Target horizon must be {self.config.model.prediction_horizon}"
                
            return True, "Data validation successful"
            
        except Exception as e:
            return False, f"Data validation error: {str(e)}"

    def check_is_trained(self) -> bool:
        """Check if the model has been trained."""
        return hasattr(self, 'is_trained') and self.is_trained

    def get_model_summary(self) -> str:
        """Get model summary as string."""
        try:
            string_list = []
            self.model.summary(print_fn=lambda x: string_list.append(x))
            summary_str = '\n'.join(string_list)
            return summary_str.encode('ascii', 'ignore').decode('ascii')
        except Exception as e:
            return f"Error getting model summary: {str(e)}"