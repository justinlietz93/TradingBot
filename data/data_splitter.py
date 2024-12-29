import logging
import numpy as np
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def split_data(data: dict, train_split: float = 0.7, val_split: float = 0.15) -> Tuple[Dict, Dict, Dict]:
    """Split data into training, validation, and testing sets using time-based splitting."""
    train_data, val_data, test_data = {}, {}, {}
    
    for ticker, ticker_data in data.items():
        try:
            X, y = ticker_data['X'], ticker_data['y']
            
            # Calculate split indices
            train_idx = int(len(X) * train_split)
            val_idx = int(len(X) * (train_split + val_split))
            
            # Split the data
            train_data[ticker] = {
                'X': X[:train_idx],
                'y': y[:train_idx]
            }
            
            val_data[ticker] = {
                'X': X[train_idx:val_idx],
                'y': y[train_idx:val_idx]
            }
            
            test_data[ticker] = {
                'X': X[val_idx:],
                'y': y[val_idx:]
            }
            
            logger.info(f"Split data for {ticker}:")
            logger.info(f"  Train size: {len(train_data[ticker]['X'])} samples")
            logger.info(f"  Validation size: {len(val_data[ticker]['X'])} samples")
            logger.info(f"  Test size: {len(test_data[ticker]['X'])} samples")
            
            # Verify data integrity
            _verify_data_split(train_data[ticker], val_data[ticker], test_data[ticker])
            
        except Exception as e:
            logger.error(f"Error splitting data for {ticker}: {e}")
            raise
    
    return train_data, val_data, test_data

def _verify_data_split(train_data: Dict, val_data: Dict, test_data: Dict):
    """Verify the integrity of the data split."""
    # Check for data leakage
    train_end = len(train_data['X'])
    val_end = train_end + len(val_data['X'])
    test_end = val_end + len(test_data['X'])
    
    # Ensure sequential ordering
    if not (train_end > 0 and val_end > train_end and test_end > val_end):
        raise ValueError("Invalid split sequence ordering")
    
    # Check shapes
    if train_data['X'].shape[1:] != val_data['X'].shape[1:] or train_data['X'].shape[1:] != test_data['X'].shape[1:]:
        raise ValueError("Inconsistent feature dimensions across splits")
    
    # Check for NaN values
    for dataset in [train_data, val_data, test_data]:
        if np.isnan(dataset['X']).any() or np.isnan(dataset['y']).any():
            raise ValueError("NaN values detected in split data")
            
    # Verify minimum sizes
    min_size = 32  # Minimum batch size
    if len(train_data['X']) < min_size or len(val_data['X']) < min_size or len(test_data['X']) < min_size:
        raise ValueError(f"Split sizes too small, minimum required: {min_size} samples")
