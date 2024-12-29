from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    def __init__(self, config: 'Config'):
        self.config = config
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def build_model(self) -> None:
        """Build and compile the model."""
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 100, batch_size: int = 32) -> Dict:
        """Train the model and return training history."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        pass
    
    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model performance."""
        pass
    
    def save_model(self, path: str) -> None:
        """Save the model to disk."""
        try:
            if hasattr(self.model, 'save'):
                self.model.save(path)
                logger.info(f"Model saved to {path}")
            else:
                logger.warning("Model does not support saving")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, path: str) -> None:
        """Load the model from disk."""
        try:
            if hasattr(self.model, 'load'):
                self.model.load(path)
                self.is_trained = True
                logger.info(f"Model loaded from {path}")
            else:
                logger.warning("Model does not support loading")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def validate_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        # Validate input data shape and format.
        
        # Args:
        #     X (np.ndarray): Input features data.
        #     y (np.ndarray, optional): Target data. Defaults to None.
        
        # Returns:
        #     Tuple[bool, str]: A tuple containing a boolean indicating if validation passed and a message.
        try:
            logger.debug(f"X shape: {X.shape}, y shape: {y.shape if y is not None else None}")

            if len(X.shape) != 3:
                return False, "Input data must be 3D (samples, timesteps, features)"
            
            # Check if the number of features in X matches the expected number
            if X.shape[2] != len(self.config.model.features):
                return False, f"Expected {len(self.config.model.features)} features, got {X.shape[2]}"
            
            # If y is provided, perform additional checks
            if y is not None:
                # Check if y has the expected 2D shape (samples, prediction_horizon)
                if len(y.shape) != 2:
                    return False, "Target data must be 2D (samples, prediction_horizon)"
                
                # Check if the prediction horizon in y matches the expected value
                if y.shape[1] != self.config.model.prediction_horizon:
                    return False, f"Expected prediction horizon of {self.config.model.prediction_horizon}, got {y.shape[1]}"
            
            return True, "Data validation passed"
        
        except Exception as e:
            return False, f"Error validating data: {str(e)}"
    
    def check_is_trained(self) -> bool:
        """Check if the model is trained."""
        if not self.is_trained:
            logger.warning("Model is not trained yet")
            return False
        else:
            logger.info("Model is trained")
        return True
    
    def get_model_summary(self) -> str:
        try:
            string_list = []
            self.model.summary(print_fn=lambda x: string_list.append(x))
            return '\n'.join(string_list)
        except Exception as e:
            return f"Error getting model summary: {str(e)}"
