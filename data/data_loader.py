import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from scipy.stats import zscore
import yfinance as yf
import logging
from config.config import Config

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, tickers, start_date, end_date):
        self.config = Config.get_default_config()
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.scalers = {}
        self.original_data = None

        logger.info(f"[DEBUG] DataLoader initialized with config: {self.config}")
        logger.info(f"[DEBUG] DataLoader initialized with tickers: {self.tickers}")
        logger.info(f"[DEBUG] DataLoader initialized with start_date: {self.start_date}")
        logger.info(f"[DEBUG] DataLoader initialized with end_date: {self.end_date}")
        
    def fetch_data(self):
        data = {}
        for ticker in self.tickers:
            ticker_data = yf.download(ticker, start=self.start_date, end=self.end_date)
            
            # Reset the index to remove the MultiIndex structure
            ticker_data.reset_index(inplace=True)
            
            # Flatten the columns by removing the ticker level
            ticker_data.columns = ticker_data.columns.droplevel(1)
            
            # Rename columns to lowercase for consistency
            ticker_data.columns = [col.lower() for col in ticker_data.columns]
            
            # Ensure 'close' column is present
            if 'close' not in ticker_data.columns:
                raise ValueError(f"'close' column not found in data for {ticker}")
            
            logger.info(f"[DEBUG] Columns before preprocessing: {ticker_data.columns}")
            X, y = self.preprocess_data(ticker_data, lookback_period=60)
            
            if X is None or y is None:
                logger.warning(f"Skipping {ticker} due to preprocessing failure")
                continue
            
            # Store the 3D sequences directly without flattening
            data[ticker] = {'X': X, 'y': y}
        
        if not data:
            raise ValueError("No valid data after preprocessing")
        
        return data

    def preprocess_data(self, data, lookback_period):
        logger.info(f"[DEBUG] Preprocessing data for lookback period: {lookback_period}")
        logger.info(f"[DEBUG] Columns in data before filtering: {data.columns}")
        logger.info(f"[DEBUG] Starting preprocess with {len(data)} rows")

        # Handle missing values
        data = data.ffill().bfill()
        logger.info(f"[DEBUG] Columns after handling missing values: {data.columns}")
        logger.info(f"[DEBUG] After handling missing values, rows = {len(data)}")

        # Add technical indicators
        data = self._add_technical_indicators(data)
        logger.info(f"[DEBUG] Columns after adding technical indicators: {data.columns}")
        logger.info(f"[DEBUG] After adding technical indicators, rows = {len(data)}")

        data = self._add_derived_features(data)
        logger.info(f"[DEBUG] Columns after adding derived features: {data.columns}")
        logger.info(f"[DEBUG] After adding derived features, rows = {len(data)}")
        logger.info(f"[DEBUG] 'returns' column present: {'returns' in data.columns}")

        # Validate required columns
        required_columns = self.config.model.features
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}. Generating fallback values or dropping non-critical features.")
            for col in missing_columns:
                if col == 'close':
                    raise ValueError("Critical feature 'close' is missing after preprocessing.")
                elif col == 'returns':
                    data['returns'] = data['close'].pct_change()
                    data['returns'].fillna(0, inplace=True)
                elif col == 'rsi':
                    data['rsi'] = self._calculate_rsi(data['close'])
                elif col == 'macd':
                    data['macd'] = self._calculate_macd(data['close'])
                elif col in ['bb_upper', 'bb_lower']:
                    data['bb_upper'], data['bb_lower'] = self._calculate_bollinger_bands(data['close'])
                elif col == 'volatility':
                    data['volatility'] = data['returns'].rolling(window=20).std()
                elif col == 'volume_ma':
                    if 'volume' in data.columns:
                        data['volume_ma'] = data['volume'].rolling(window=20).mean().ffill()
                    else:
                        logger.warning("'volume' column not found. Skipping volume moving average calculation.")
                elif col == 'open':
                    logger.warning(f"Generating fallback value for missing 'open' feature based on previous day's 'close' price.")
                    data['open'] = data['close'].shift(1)
                elif col in ['high', 'low']:
                    logger.warning(f"Forward filling missing '{col}' feature with 'close' price.")
                    data[col] = data['close']
                elif col == 'volume':
                    logger.warning("'volume' column not found. Filling with zeros.")
                    data['volume'] = 0
                else:
                    raise ValueError(f"Missing required feature: {col}")
        logger.info(f"[DEBUG] Columns after validating required columns: {data.columns}")
        logger.info(f"[DEBUG] After validating required columns, rows = {len(data)}")








        # At this point 'returns_lag' is still in data.columns
        logger.info(f"[DEBUG] Features from config: {self.config.model.features}")
        # Select only available features in config.model.features
        available_features = [col for col in self.config.model.features if col in data.columns]

        # Keep target column separate
        target_data = data[self.config.model.target] if self.config.model.target in data.columns else None

        # Select only features
        data = data[available_features]

        # Create sequences from features and target
        if target_data is not None:
            X, y = self._create_sequences(data.values, target_data.values)
        else:
            # Handle case where no target data is available
            logger.warning("No target data available for sequence creation")
            X, y = None, None

        return X, y
    
    def _scale_features(self, data):
        """Scale features using MinMaxScaler."""
        scaler = MinMaxScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(data.drop(columns=['ticker'])), columns=data.columns.drop('ticker'), index=data.index)
        self.scalers[data['ticker'].iloc[0]] = scaler
        return data_scaled
    
    def _calculate_rsi(self, close, window=14):
        """Calculate Relative Strength Index (RSI)."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, close, short_period=12, long_period=26, signal_period=9):
        """Calculate Moving Average Convergence Divergence (MACD)."""
        ema_short = close.ewm(span=short_period, adjust=False).mean()
        ema_long = close.ewm(span=long_period, adjust=False).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_histogram
    
    def _calculate_bollinger_bands(self, close, window=20, num_std=2):
        """Calculate Bollinger Bands."""
        rolling_mean = close.rolling(window=window).mean()
        rolling_std = close.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        try:
            logger.info(f"[DEBUG] Calculating RSI for 'close' column")
            df['rsi'] = self._calculate_rsi(df['close'])
            logger.info(f"[DEBUG] Calculating MACD for 'close' column")
            df['macd'] = self._calculate_macd(df['close'])
            logger.info(f"[DEBUG] Calculating Bollinger Bands for 'close' column")
            df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error in adding technical indicators: {str(e)}")
            raise
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataframe."""
        try:
            df['returns'] = df['close'].pct_change()
            df['returns'].fillna(0, inplace=True)
            logger.info(f"[DEBUG] Calculated 'returns' column: {df['returns'].head()}")

            # Calculate 'returns_lag' column
            df['returns_lag'] = df['returns'].shift(1).fillna(0)
            logger.info(f"[DEBUG] Calculated 'returns_lag' column: {df['returns_lag'].head()}")

            df['volatility'] = df['returns'].rolling(window=20).std()
            
            if 'volume' in df.columns:
                df['volume_ma'] = df['volume'].rolling(window=20).mean().ffill()
            else:
                logger.warning("'volume' column not found. Skipping volume moving average calculation.")
            
            logger.info(f"[DEBUG] Columns after adding derived features: {df.columns}")
            return df
            
        except Exception as e:
            logger.error(f"Error in adding derived features: {str(e)}")
            raise
    
    def prepare_ml_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for machine learning model."""
        try:
            logger.info("Preparing features for ML model")
            
            # Convert feature names to lowercase for consistency
            features = [str(f).lower() for f in self.config.model.features]
            target = str(self.config.model.target).lower()
            
            # Ensure all required features are present
            missing_features = set(features) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing features in data: {missing_features}")
            
            # Select features and target
            feature_data = df[features].values
            target_data = df[target].values
            
            # Create sequences with proper 3D shape
            num_samples = len(feature_data) - self.config.model.lookback_period
            X = np.zeros((num_samples, self.config.model.lookback_period, len(features)))
            y = np.zeros((num_samples))
            
            for i in range(num_samples):
                X[i] = feature_data[i:i + self.config.model.lookback_period]
                y[i] = target_data[i + self.config.model.lookback_period]
            
            # Split data
            split_idx = int(len(X) * self.config.model.train_split)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f"[DEBUG] X_train shape: {X_train.shape}")
            logger.info(f"[DEBUG] y_train shape: {y_train.shape}")
            logger.info(f"[DEBUG] X_test shape: {X_test.shape}")
            logger.info(f"[DEBUG] y_test shape: {y_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing ML data: {str(e)}")
            raise
    
    def get_unscaled_data(self) -> pd.DataFrame:
        """Get the unscaled data for trading execution."""
        if self.original_data is None:
            raise ValueError("No data available. Call fetch_data first.")
        return self.original_data
    
    def _create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        X, y = [], []
        
        for i in range(len(features) - self.config.model.lookback_period - self.config.model.prediction_horizon + 1):
            X.append(features[i:(i + self.config.model.lookback_period)])
            y.append(target[i + self.config.model.lookback_period:i + self.config.model.lookback_period + self.config.model.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        split_idx = int(len(X) * self.config.model.train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"[DEBUG] y_train shape: {y_train.shape}")
        logger.info(f"[DEBUG] y_train info:\n{pd.DataFrame(y_train.reshape(-1, y_train.shape[-1])).info()}")
        
        logger.info(f"[DEBUG] y_test shape: {y_test.shape}")
        logger.info(f"[DEBUG] y_test info:\n{pd.DataFrame(y_test.reshape(-1, y_test.shape[-1])).info()}")
        
        logger.info(f"[DEBUG] Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test

def load_data(tickers, start_date, end_date):
    loader = DataLoader(tickers, start_date, end_date)
    data = loader.fetch_data()
    return data