# Machine learning model implementation
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, Input, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Tuple, Dict, Any
import logging
import numpy as np
from datetime import datetime
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class MLModel:
    """Enhanced LSTM model for financial time series prediction."""
    def __init__(self, input_shape: Tuple[int, int], horizon: int, config=None):
        self.input_shape = input_shape
        self.horizon = horizon
        self.output_shape = horizon * 2  # Changed to match the target shape (returns and directions)
        self.config = config or Config.get_default_config().model
        
        # Build the model
        self.model = self._build_model(input_shape, self.output_shape)
        self.history = None
        
    def _build_model(self, input_shape: Tuple[int, ...], output_shape: int) -> Model:
        """Build the LSTM model architecture with enhanced attention and feature extraction."""
        inputs = Input(shape=input_shape, name='input_layer')
        
        # Normalize input
        x = LayerNormalization()(inputs)
        
        # LSTM blocks with residual connections
        lstm_units = self.config['lstm_units']
        attention_heads = self.config['attention_heads']
        key_dim = self.config['attention_key_dim']
        
        for i, (units, heads) in enumerate(zip(lstm_units, attention_heads)):
            # LSTM layer
            lstm = LSTM(units, return_sequences=True)(x)
            lstm = LayerNormalization()(lstm)
            
            # Multi-head attention
            attention = MultiHeadAttention(num_heads=heads, key_dim=key_dim)(lstm, lstm)
            x = tf.keras.layers.Add()([lstm, attention])
            x = Dropout(self.config['dropout_rate'])(x)
        
        # Final LSTM layer
        lstm_final = LSTM(lstm_units[-1], return_sequences=True)(x)
        lstm_final = LayerNormalization()(lstm_final)
        
        # Parallel feature extraction
        avg_pool = GlobalAveragePooling1D()(lstm_final)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(lstm_final)
        
        # Combine features
        x = tf.keras.layers.Concatenate()([avg_pool, max_pool])
        
        # Dense layers with residual connections
        dense_units = self.config['dense_units']
        dense_layers = []
        
        for units in dense_units:
            dense = Dense(units, activation='relu')(x)
            dense = LayerNormalization()(dense)
            dense = Dropout(self.config['dropout_rate'])(dense)
            dense_layers.append(dense)
            x = dense
        
        # Skip connections
        concat = tf.keras.layers.Concatenate()(dense_layers)
        
        # Split into returns and directions branches
        returns_branch = Dense(128, activation='relu')(concat)
        returns_branch = LayerNormalization()(returns_branch)
        returns_branch = Dropout(self.config['dropout_rate'])(returns_branch)
        returns_output = Dense(self.horizon, activation='linear', name='returns_output')(returns_branch)
        
        directions_branch = Dense(128, activation='relu')(concat)
        directions_branch = LayerNormalization()(directions_branch)
        directions_branch = Dropout(self.config['dropout_rate'])(directions_branch)
        directions_output = Dense(self.horizon, activation='tanh', name='directions_output')(directions_branch)
        
        # Combine outputs
        outputs = tf.keras.layers.Concatenate()([returns_output, directions_output])
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Custom optimizer with gradient clipping
        optimizer = Adam(
            learning_rate=self.config['learning_rate'],
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss=self._combined_loss,
            metrics=['mae', self._direction_accuracy]
        )
        
        logger.info("Built enhanced LSTM model with improved feature extraction")
        model.summary(print_fn=logger.info)
        
        return model
        
    def _direction_accuracy(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate direction prediction accuracy."""
        # Split predictions and true values into returns and directions
        pred_returns = y_pred[:, :self.horizon]
        true_returns = y_true[:, :self.horizon]
        pred_directions = y_pred[:, self.horizon:]
        true_directions = y_true[:, self.horizon:]
        
        # Calculate direction accuracy from both returns and explicit directions
        returns_direction_acc = tf.reduce_mean(tf.cast(tf.equal(tf.sign(pred_returns), tf.sign(true_returns)), tf.float32))
        explicit_direction_acc = tf.reduce_mean(tf.cast(tf.equal(tf.sign(pred_directions), tf.sign(true_directions)), tf.float32))
        
        # Return the average of both accuracies
        return (returns_direction_acc + explicit_direction_acc) / 2.0
        
    def _returns_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Custom loss function for returns prediction."""
        # Extract returns from the combined target
        true_returns = y_true[:, :self.horizon]
        pred_returns = y_pred[:, :self.horizon]
        return tf.keras.losses.huber(true_returns, pred_returns)
    
    def _directions_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Custom loss function for directions prediction."""
        # Extract directions from the combined target
        true_directions = y_true[:, self.horizon:]
        pred_directions = y_pred[:, self.horizon:]
        return tf.keras.losses.binary_crossentropy(
            tf.nn.sigmoid(true_directions),
            tf.nn.sigmoid(pred_directions)
        )
    
    def _combined_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Combined loss function for both returns and directions."""
        returns_loss = self._returns_loss(y_true, y_pred)
        directions_loss = self._directions_loss(y_true, y_pred)
        return returns_loss + directions_loss
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, epochs: int = None, batch_size: int = None) -> Dict[str, list]:
        """Train the model with enhanced validation and monitoring."""
        try:
            # Use config values if not provided
            epochs = epochs or self.config['epochs']
            batch_size = batch_size or self.config['batch_size']
            
            # Validate input shapes
            if len(X_train.shape) != 3:
                raise ValueError(f"X_train must be 3D (samples, timesteps, features), got shape {X_train.shape}")
            if len(y_train.shape) != 2:
                raise ValueError(f"y_train must be 2D (samples, targets), got shape {y_train.shape}")
            
            # Validate target shape
            if y_train.shape[1] != self.horizon * 2:
                raise ValueError(f"y_train must have shape (samples, {self.horizon * 2}), got shape {y_train.shape}")
            
            # Validate matching samples
            if len(X_train) != len(y_train):
                raise ValueError(f"X_train and y_train must have same number of samples. Got {len(X_train)} vs {len(y_train)}")
            
            # Validate validation data if provided
            if X_val is not None and y_val is not None:
                if X_val.shape[1:] != X_train.shape[1:]:
                    raise ValueError(f"X_val must have same shape as X_train except for samples dimension")
                if y_val.shape[1:] != y_train.shape[1:]:
                    raise ValueError(f"y_val must have same shape as y_train except for samples dimension")
            
            logger.info(f"Starting training with configuration:")
            logger.info(f"  Training samples: {len(X_train)}")
            logger.info(f"  Validation samples: {len(X_val) if X_val is not None else 'None'}")
            logger.info(f"  Timesteps: {X_train.shape[1]}")
            logger.info(f"  Features: {X_train.shape[2]}")
            logger.info(f"  Target shape: {y_train.shape[1]} (returns and directions)")
            logger.info(f"  Batch size: {batch_size}")
            logger.info(f"  Epochs: {epochs}")
            
            # Create base directories using pathlib.Path
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            root_dir = Path(r"C:\Users\jliet\source\repos\Python\Python Projects\Trading_Bot")
            
            # Create directory paths using pathlib.Path
            logs_dir = root_dir / 'logs'
            training_dir = logs_dir / 'training' / timestamp
            checkpoints_dir = root_dir / 'models' / 'checkpoints' / timestamp
            tensorboard_dir = training_dir / 'tensorboard'
            
            # Create all directories
            logs_dir.mkdir(exist_ok=True)
            training_dir.mkdir(parents=True, exist_ok=True)
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Created directories:")
            logger.info(f"  Logs: {logs_dir}")
            logger.info(f"  Training: {training_dir}")
            logger.info(f"  Checkpoints: {checkpoints_dir}")
            logger.info(f"  TensorBoard: {tensorboard_dir}")
            
            # Configure file logging
            log_file = training_dir / "training.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8', errors='replace')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
            
            try:
                # Enhanced callbacks with improved logging
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss' if X_val is not None else 'loss',
                        patience=self.config['early_stopping_patience'],
                        restore_best_weights=True,
                        mode='min',
                        verbose=1
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss' if X_val is not None else 'loss',
                        factor=0.5,
                        patience=self.config['reduce_lr_patience'],
                        min_lr=self.config['min_lr'],
                        mode='min',
                        verbose=1
                    ),
                    tf.keras.callbacks.CSVLogger(
                        str(training_dir / 'training_metrics.csv'),
                        separator=',',
                        append=True
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=str(checkpoints_dir / 'model-{epoch:02d}-{val_loss:.4f}.keras'),
                        monitor='val_loss' if X_val is not None else 'loss',
                        save_best_only=True,
                        save_weights_only=False,
                        mode='min',
                        verbose=1
                    ),
                    tf.keras.callbacks.TensorBoard(
                        log_dir=str(tensorboard_dir),
                        histogram_freq=1,
                        write_graph=True,
                        write_images=True,
                        update_freq='epoch',
                        profile_batch=0
                    )
                ]
                
                # Save model architecture
                try:
                    with open(checkpoints_dir / 'model_architecture.json', 'w', encoding='utf-8') as f:
                        f.write(self.model.to_json())
                    logger.info(f"Saved initial model architecture to {checkpoints_dir / 'model_architecture.json'}")
                except Exception as e:
                    logger.warning(f"Could not save model architecture: {str(e)}")

                logger.info(f"Training {'with provided validation data' if X_val is not None else 'without validation data'}. Logs will be saved to {training_dir}")

                # Train the model
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val) if X_val is not None else None,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1
                )

                if history is None:
                    logger.error("Model training failed: fit() returned None")
                    return None

                # Log training results
                final_epoch = len(history.history['loss'])
                best_epoch = np.argmin(history.history['val_loss' if X_val is not None else 'loss']) + 1

                logger.info(f"Training completed after {final_epoch} epochs")
                logger.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
                if X_val is not None:
                    logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
                logger.info(f"Best model found at epoch {best_epoch}")
                logger.info(f"Best training loss: {min(history.history['loss']):.4f}")
                if X_val is not None:
                    logger.info(f"Best validation loss: {min(history.history['val_loss']):.4f}")

                # Save training history
                try:
                    history_file = training_dir / 'training_history.npy'
                    np.save(history_file, history.history)
                    logger.info(f"Training history saved to {history_file}")
                except Exception as e:
                    logger.warning(f"Could not save training history: {str(e)}")

                # Save final model
                try:
                    final_model_path = checkpoints_dir / 'final_model.keras'
                    self.model.save(final_model_path)
                    logger.info(f"Final model saved to {final_model_path}")
                except Exception as e:
                    logger.warning(f"Could not save final model: {str(e)}")

                return history.history

            except Exception as e:
                logger.error(f"Error during training: {str(e)}")
                return None

            finally:
                # Clean up file handler
                logger.removeHandler(file_handler)
                file_handler.close()
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return None
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for the input data."""
        try:
            # Make predictions
            predictions = self.model.predict(X, verbose=0)
            
            # Log prediction shapes and statistics
            logger.info(f"Generated predictions with shape: {predictions.shape}")
            
            # Split predictions into returns and directions
            returns = predictions[:, :self.horizon]
            directions = predictions[:, self.horizon:]
            
            # Log prediction statistics
            logger.info("Prediction statistics:")
            logger.info(f"  Returns - Mean: {returns.mean():.4f}, Std: {returns.std():.4f}")
            logger.info(f"  Directions - Mean: {directions.mean():.4f}, Std: {directions.std():.4f}")
            
            # Combine predictions
            combined_predictions = np.concatenate([returns, directions], axis=1)
            
            return combined_predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
