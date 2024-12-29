"""
Global parameters for the Trading Bot project.
All configurable parameters should be defined here to make it easier to modify them.
"""

# Data Parameters
LOOKBACK_PERIOD = 60  # Number of time steps to look back for predictions
PREDICTION_HORIZON = 5  # Number of time steps to predict into the future
TRAIN_TEST_SPLIT = 0.8  # Proportion of data to use for training
VALIDATION_SPLIT = 0.2  # Proportion of training data to use for validation
BATCH_SIZE = 32  # Batch size for training
MAX_EPOCHS = 100  # Maximum number of epochs for training

# Model Parameters
LEARNING_RATE = 0.001  # Learning rate for the optimizer
DROPOUT_RATE = 0.2  # Dropout rate for regularization
LSTM_UNITS = 128  # Number of LSTM units in each layer
DENSE_UNITS = 64  # Number of dense units in each layer
L2_REGULARIZATION = 0.01  # L2 regularization factor
EARLY_STOPPING_PATIENCE = 10  # Number of epochs to wait before early stopping

# Trading Parameters
INITIAL_BALANCE = 100000  # Initial balance for trading simulation
TRANSACTION_FEE = 0.001  # Transaction fee as a percentage
STOP_LOSS = 0.02  # Stop loss percentage
TAKE_PROFIT = 0.04  # Take profit percentage
POSITION_SIZE = 0.1  # Position size as a fraction of portfolio
MAX_POSITIONS = 5  # Maximum number of simultaneous positions

# Data Processing Parameters
PRICE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
TECHNICAL_INDICATORS = [
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Middle', 'BB_Lower',
    'EMA_9', 'SMA_20', 'SMA_50'
]
FEATURE_COLUMNS = PRICE_COLUMNS + TECHNICAL_INDICATORS

# File Paths
ROOT_DIR = r'C:\Users\jliet\source\repos\Python\Python Projects\Trading_Bot'
DATA_DIR = f'{ROOT_DIR}/data'
MODELS_DIR = f'{ROOT_DIR}/models'
LOGS_DIR = f'{ROOT_DIR}/logs'
RESULTS_DIR = f'{ROOT_DIR}/results'

# Logging Parameters
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Stock Universe
STOCK_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
TIME_PERIOD = '1d'  # Time period for data collection (1d = daily)
START_DATE = '2010-01-01'
END_DATE = '2023-12-31'

# Model Training Configuration
MODEL_CONFIG = {
    'learning_rate': LEARNING_RATE,
    'dropout_rate': DROPOUT_RATE,
    'lstm_units': LSTM_UNITS,
    'dense_units': DENSE_UNITS,
    'l2_regularization': L2_REGULARIZATION,
    'batch_size': BATCH_SIZE,
    'epochs': MAX_EPOCHS,
    'early_stopping_patience': EARLY_STOPPING_PATIENCE
}

# Trading Strategy Parameters
STRATEGY_PARAMS = {
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'macd_threshold': 0,
    'bb_threshold': 0.1,  # Percentage from Bollinger Band to trigger signals
    'volume_threshold': 1.5  # Volume increase factor to confirm signals
}

# Risk Management Parameters
RISK_PARAMS = {
    'max_portfolio_risk': 0.02,  # Maximum portfolio risk (2%)
    'position_risk': 0.01,  # Risk per position (1%)
    'max_drawdown': 0.15,  # Maximum drawdown before stopping (15%)
    'stop_loss': STOP_LOSS,
    'take_profit': TAKE_PROFIT
}

# Performance Metrics
METRICS = [
    'total_return',
    'annual_return',
    'sharpe_ratio',
    'max_drawdown',
    'win_rate',
    'profit_factor',
    'avg_trade_return'
] 