from sklearn.model_selection import train_test_split
import logging
import numpy as np

logger = logging.getLogger(__name__)

def split_data(data):
    """Split 3D sequence data into train and test sets while maintaining the temporal structure.
    
    Args:
        data (dict): Dictionary containing sequences for each ticker
            Each ticker has 'X' (3D array of shape [samples, timesteps, features])
            and 'y' (target values)
        
    Returns:
        tuple: (train_data, test_data) dictionaries containing sequences for each ticker
    """
    train_data = {}
    test_data = {}
    
    for ticker, ticker_data in data.items():
        logger.debug(f"Ticker: {ticker}, keys: {list(ticker_data.keys())}")

        if 'X' not in ticker_data or 'y' not in ticker_data:
            raise ValueError(f"Missing 'X' or 'y' in data for ticker {ticker}")

        X = ticker_data['X']  # Shape: (samples, timesteps, features)
        y = ticker_data['y']  # Shape: (samples,) or (samples, prediction_horizon)

        # Check data types
        if not isinstance(X, np.ndarray):
            raise ValueError(f"'X' for {ticker} must be a NumPy array but got {type(X)}")
        if not isinstance(y, np.ndarray):
            raise ValueError(f"'y' for {ticker} must be a NumPy array but got {type(y)}")

        # Validate shapes
        if len(X.shape) != 3:
            raise ValueError(f"'X' for {ticker} must be 3D but got shape {X.shape}")
        if len(y.shape) not in [1, 2]:
            raise ValueError(f"'y' for {ticker} must be 1D or 2D but got shape {y.shape}")


        logger.info(f"Sequence shape for {ticker}: X={X.shape}, y={y.shape}")
        
        # Calculate split index (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        
        # Split the sequences maintaining temporal order
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(f"Training sequences shape: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Test sequences shape: X={X_test.shape}, y={y_test.shape}")
        
        train_data[ticker] = {'X': X_train, 'y': y_train}
        test_data[ticker] = {'X': X_test, 'y': y_test}
    
    if not train_data or not test_data:
        raise ValueError("No valid data after splitting")
    
    return train_data, test_data 