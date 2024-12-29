"""
Trading Bot Parameters Configuration

This module contains all configurable parameters used throughout the trading bot.
Parameters are organized by their functional area.
"""

# Data Parameters
DATA_PARAMS = {
    'sequence_length': 60,  # Number of time steps in each sequence
    'prediction_horizon': 5,  # Number of steps to predict ahead
    'train_split': 0.7,  # Proportion of data for training
    'val_split': 0.15,  # Proportion of data for validation
    'test_split': 0.15,  # Proportion of data for testing
    'batch_size': 32,  # Batch size for training
}

# Model Parameters
MODEL_PARAMS = {
    'lstm_units': [128, 64],  # Number of units in LSTM layers
    'dense_units': [32, 16],  # Number of units in dense layers
    'dropout_rate': 0.2,  # Dropout rate for regularization
    'learning_rate': 0.001,  # Learning rate for optimizer
    'epochs': 100,  # Number of training epochs
    'early_stopping_patience': 10,  # Patience for early stopping
}

# Trading Strategy Parameters
STRATEGY_PARAMS = {
    'bull_threshold': 0.55,  # Threshold for bullish signals
    'bear_threshold': 0.45,  # Threshold for bearish signals
    'trend_threshold': 0.1,  # Minimum trend strength required
    'return_threshold': 0.002,  # Minimum expected return
    'direction_threshold': 0.6,  # Minimum direction probability
    'confidence_threshold': 0.7,  # Minimum prediction confidence
}

# Risk Management Parameters
RISK_PARAMS = {
    'max_position_size': 0.1,  # Maximum position size as fraction of portfolio
    'stop_loss_pct': 0.02,  # Stop loss percentage
    'take_profit_pct': 0.04,  # Take profit percentage
    'max_drawdown': 0.2,  # Maximum allowable drawdown
    'risk_per_trade': 0.01,  # Risk per trade as fraction of portfolio
}

# Technical Indicators
TECHNICAL_INDICATORS = {
    'sma': [20, 50, 200],  # Simple Moving Average periods
    'ema': [12, 26],  # Exponential Moving Average periods
    'rsi': 14,  # Relative Strength Index period
    'macd': {'fast': 12, 'slow': 26, 'signal': 9},  # MACD parameters
    'bollinger': {'period': 20, 'std_dev': 2},  # Bollinger Bands parameters
}

# Logging Parameters
LOGGING_PARAMS = {
    'log_level': 'INFO',
    'trading_log_file': 'logs/trading/trading.log',
    'training_log_file': 'logs/training/training.log',
    'backtest_log_file': 'logs/backtest/backtest.log',
    'error_log_file': 'logs/errors/error.log',
}

# Backtest Parameters
BACKTEST_PARAMS = {
    'initial_capital': 100000,  # Initial capital for backtesting
    'commission_rate': 0.001,  # Commission rate per trade
    'slippage': 0.0001,  # Slippage assumption
    'trade_frequency': 'daily',  # Trading frequency
}

# Environment Parameters
ENV_PARAMS = {
    'random_seed': 42,  # Random seed for reproducibility
    'num_workers': 4,  # Number of workers for data loading
    'use_gpu': True,  # Whether to use GPU for training
    'precision': 'float32',  # Numerical precision
}

# Feature Engineering Parameters
FEATURE_PARAMS = {
    'price_features': ['open', 'high', 'low', 'close', 'volume'],
    'technical_features': ['sma', 'ema', 'rsi', 'macd', 'bollinger'],
    'target_column': 'close',  # Target column for prediction
    'normalize_method': 'minmax',  # Method for feature normalization
}

# Market Data Parameters
MARKET_PARAMS = {
    'tickers': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],  # Tickers to trade
    'timeframe': '1d',  # Data timeframe
    'start_date': '2010-01-01',  # Start date for historical data
    'data_source': 'yfinance',  # Data source
}
