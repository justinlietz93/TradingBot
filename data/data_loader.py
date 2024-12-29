"""
Data loading and preprocessing functionality.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import logging
import pandas_ta as ta
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and preprocessing financial data."""
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data for all tickers."""
        for ticker in self.tickers:
            try:
                logger.info(f"Loading data for {ticker}...")
                stock = yf.Ticker(ticker)
                df = stock.history(start=self.start_date, end=self.end_date, interval='1d')
                
                if df.empty:
                    logger.error(f"No data found for {ticker}")
                    continue
                    
                logger.info(f"Successfully loaded {len(df)} samples for {ticker}")
                self.data[ticker] = self.preprocess_data(df, ticker)
                
                # Verify data size after preprocessing
                if len(self.data[ticker]) < 1000:
                    logger.warning(f"Insufficient data for {ticker} after preprocessing. Got {len(self.data[ticker])} samples, need at least 1000.")
                    del self.data[ticker]
                    continue
                
            except Exception as e:
                logger.error(f"Error loading data for {ticker}: {str(e)}")
                continue
                
        if not self.data:
            raise ValueError("No valid data loaded for any ticker")
            
        return self.data

    def preprocess_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Preprocess raw data with improved NaN handling."""
        try:
            # Validate input data
            self.validate_data(df)
            
            # Convert column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Forward fill NaN values in price data
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = df[price_cols].ffill()
            
            # Calculate returns after filling NaNs
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Handle volume data
            df['volume'] = df['volume'].fillna(df['volume'].mean())
            df['volume'] = df['volume'].astype(np.float64)
            
            # Add technical indicators with error handling
            df = self._add_technical_indicators(df)
            
            # Scale features
            df = self._scale_features(df)
            
            # Calculate target variables
            df['target_next_return'] = df['returns'].shift(-1)
            df['target_direction'] = np.sign(df['target_next_return'])
            
            # Fill remaining NaN values with forward fill then backward fill
            df = df.ffill().bfill()
            
            # Drop rows where critical values are still NaN
            critical_cols = ['returns', 'target_next_return', 'target_direction']
            initial_size = len(df)
            df = df.dropna(subset=critical_cols)
            final_size = len(df)
            
            if final_size < initial_size:
                logger.info(f"Dropped {initial_size - final_size} rows with NaN values in critical columns for {ticker}")
            
            if final_size < 100:  # Minimum required for any meaningful analysis
                logger.warning(f"Too few samples remaining for {ticker} after preprocessing: {final_size}")
                return pd.DataFrame()  # Return empty DataFrame to signal invalid data
            
            logger.info(f"Successfully preprocessed data for {ticker}: {final_size} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data for {ticker}: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using rolling standardization."""
        # List of features to scale
        price_cols = ['open', 'high', 'low', 'close']
        return_cols = ['returns', 'log_returns']
        indicator_cols = [col for col in df.columns if col not in price_cols + return_cols + ['volume', 'target_next_return', 'target_direction']]
        
        # Scale price features
        for col in price_cols:
            df[f'{col}_scaled'] = (df[col] - df[col].rolling(window=20).mean()) / df[col].rolling(window=20).std()
        
        # Scale volume
        df['volume_scaled'] = (df['volume'] - df['volume'].rolling(window=20).mean()) / df['volume'].rolling(window=20).std()
        
        # Scale technical indicators
        for col in indicator_cols:
            if df[col].std() != 0:  # Only scale if there's variation
                df[f'{col}_scaled'] = (df[col] - df[col].rolling(window=20).mean()) / df[col].rolling(window=20).std()
        
        return df

    def validate_data(self, df: pd.DataFrame):
        """Validate input data."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if df.isnull().any().any():
            logging.warning("Data contains NaN values, will be dropped during preprocessing")

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using pandas_ta."""
        # Convert volume to float64 to avoid dtype issues
        df['volume'] = df['volume'].astype(np.float64)
        
        # Price-based features
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Trend indicators
        df['ema_5'] = df.ta.ema(length=5)
        df['ema_10'] = df.ta.ema(length=10)
        df['ema_20'] = df.ta.ema(length=20)
        df['ema_50'] = df.ta.ema(length=50)
        
        # Momentum indicators
        df['rsi'] = df.ta.rsi(length=14)
        stoch = df.ta.stoch(length=14)
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']
        df['willr'] = df.ta.willr(length=14)
        df['roc'] = df.ta.roc(length=12)
        
        # Volatility indicators
        bbands = df.ta.bbands(length=20)
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_middle'] = bbands['BBM_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['atr'] = df.ta.atr(length=14)
        
        # Volume indicators
        df['volume_ema'] = df.ta.ema(close=df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_ema']
        df['obv'] = df.ta.obv()
        df['cmf'] = df.ta.cmf(length=20)
        
        # Calculate MFI with proper type conversion
        try:
            # Convert price and volume data to float64 for MFI calculation
            high = df['high'].astype(np.float64)
            low = df['low'].astype(np.float64)
            close = df['close'].astype(np.float64)
            volume = df['volume'].astype(np.float64)
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate raw money flow
            raw_money_flow = typical_price * volume
            
            # Calculate positive and negative money flow
            positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
            
            # Calculate money flow ratio
            positive_flow_sum = positive_flow.rolling(window=14).sum()
            negative_flow_sum = negative_flow.rolling(window=14).sum()
            money_flow_ratio = positive_flow_sum / negative_flow_sum
            
            # Calculate MFI
            df['mfi'] = 100 - (100 / (1 + money_flow_ratio))
            df['mfi'] = df['mfi'].fillna(50)  # Fill NaN with neutral value
            
        except Exception as e:
            logger.warning(f"Error calculating MFI: {str(e)}")
            df['mfi'] = 50  # Set to neutral value on error
        
        # MACD
        macd = df.ta.macd()
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
        
        # Custom features
        df['price_momentum'] = df['close'].pct_change(5)
        df['volume_momentum'] = df['volume'].pct_change(5)
        df['volatility_14'] = df['log_return'].rolling(window=14).std() * np.sqrt(252)
        
        # Cross-sectional features
        df['ema_ratio_fast'] = df['ema_5'] / df['ema_10']
        df['ema_ratio_slow'] = df['ema_10'] / df['ema_20']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Drop any columns with all NaN values
        df = df.dropna(axis=1, how='all')
        
        # Forward fill any remaining NaN values
        df = df.ffill().fillna(0)
        
        return df

    def create_sequences(self, data: pd.DataFrame, lookback_period: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input with enhanced feature selection."""
        try:
            # Define core features that should be available for all stocks
            core_scaled_features = [
                'open_scaled', 'high_scaled', 'low_scaled', 'close_scaled',
                'volume_scaled', 'ema_5_scaled', 'ema_10_scaled', 'ema_20_scaled',
                'ema_50_scaled', 'macd_scaled', 'macd_signal_scaled', 'macd_hist_scaled',
                'atr_scaled', 'obv_scaled', 'cmf_scaled', 'price_momentum_scaled',
                'volume_momentum_scaled', 'volatility_14_scaled', 'ema_ratio_fast_scaled',
                'ema_ratio_slow_scaled'
            ]
            
            # Core normalized features
            core_normalized_features = [
                'rsi', 'stoch_k', 'stoch_d', 'willr', 'bb_position', 'mfi'
            ]
            
            # Log feature information
            logger.info(f"Using {len(core_scaled_features) + len(core_normalized_features)} consistent features for all stocks")
            logger.info(f"Creating sequences with lookback_period={lookback_period}, horizon={horizon}")
            logger.info(f"Data shape: {data.shape}")
            
            # Verify all required features are present
            missing_scaled = [f for f in core_scaled_features if f not in data.columns]
            missing_normalized = [f for f in core_normalized_features if f not in data.columns]
            
            if missing_scaled or missing_normalized:
                logger.warning(f"Missing features: {missing_scaled + missing_normalized}")
                # Add missing features with zeros
                for feature in missing_scaled + missing_normalized:
                    data[feature] = 0.0
            
            # Combine all features
            feature_columns = core_scaled_features + core_normalized_features
            
            # Create sequences
            sequences = []
            targets = []
            skipped = 0
            nan_count = 0
            
            for i in range(len(data) - lookback_period - horizon + 1):
                # Extract sequence
                sequence = data[feature_columns].iloc[i:i+lookback_period].values
                
                # Extract target (next 'horizon' returns and directions)
                returns = data['target_next_return'].iloc[i+lookback_period:i+lookback_period+horizon].values
                directions = data['target_direction'].iloc[i+lookback_period:i+lookback_period+horizon].values
                target = np.concatenate([returns, directions])
                
                # Check for NaN values
                if np.isnan(sequence).any() or np.isnan(target).any():
                    nan_count += 1
                    continue
                    
                # Check sequence validity
                if len(sequence) != lookback_period or len(target) != horizon * 2:
                    skipped += 1
                    continue
                    
                sequences.append(sequence)
                targets.append(target)
            
            # Convert to numpy arrays
            X = np.array(sequences)
            y = np.array(targets)
            
            # Log sequence creation results
            logger.info(f"Created {len(sequences)} valid sequences")
            logger.info(f"Skipped {skipped} invalid sequences")
            logger.info(f"Found {nan_count} sequences with NaN values")
            
            # Verify shapes
            if len(X) > 0:
                logger.info(f"First valid sequence shape: {X[0].shape}")
                logger.info(f"First valid target shape: {y[0].shape}")
                logger.info(f"Final shapes - X: {X.shape}, y: {y.shape}")
            else:
                raise ValueError("No valid sequences created")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise

    def prepare_data(self, lookback_period: int, horizon: int) -> Dict[str, Dict[str, np.ndarray]]:
        """Prepare data for all tickers with enhanced validation."""
        prepared_data = {}
        min_required_samples = 1000  # Minimum samples needed for reliable training
        
        for ticker in self.tickers:
            try:
                if ticker not in self.data:
                    logger.warning(f"Loading data for {ticker}...")
                    self.load_data()
                
                df = self.data.get(ticker)
                if df is None:
                    logger.error(f"No data available for {ticker}")
                    continue
                
                # Create sequences with validation
                try:
                    X, y = self.create_sequences(df, lookback_period, horizon)
                    
                    # Verify sequence creation
                    if len(X) < lookback_period + horizon:
                        logger.error(f"Insufficient sequences created for {ticker}")
                        continue
                        
                    # Verify no NaN values
                    if np.isnan(X).any() or np.isnan(y).any():
                        logger.error(f"NaN values found in sequences for {ticker}")
                        continue
                        
                    # Verify target shape
                    expected_target_shape = horizon * 2  # Both returns and directions
                    if y.shape[1] != expected_target_shape:
                        logger.error(f"Invalid target shape for {ticker}. Expected {expected_target_shape}, got {y.shape[1]}")
                        continue
                    
                    prepared_data[ticker] = {
                        'X': X,
                        'y': y
                    }
                    
                    logger.info(f"Successfully prepared data for {ticker}:")
                    logger.info(f"  Sequences: {len(X)}")
                    logger.info(f"  Features: {X.shape[2]}")
                    logger.info(f"  Target shape: {y.shape[1]} (returns and directions)")
                    logger.info(f"  Prediction horizon: {horizon}")
                    
                except Exception as e:
                    logger.error(f"Error creating sequences for {ticker}: {str(e)}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error preparing data for {ticker}: {str(e)}")
                continue
        
        if not prepared_data:
            raise ValueError("No valid data prepared for any ticker")
        
        return prepared_data