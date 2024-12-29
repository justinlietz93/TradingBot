"""
Configuration handling for the trading bot.
"""
import json
from datetime import datetime

def validate_config(config):
    """Validate configuration parameters."""
    # Check required sections
    required_sections = ['data', 'model', 'trading', 'logging']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    # Validate data section
    if not isinstance(config['data']['tickers'], list):
        raise ValueError("Tickers must be a list")
    
    try:
        datetime.strptime(config['data']['start_date'], '%Y-%m-%d')
        datetime.strptime(config['data']['end_date'], '%Y-%m-%d')
    except ValueError:
        raise ValueError("Invalid date format. Use YYYY-MM-DD")
    
    if config['data']['start_date'] > config['data']['end_date']:
        raise ValueError("Start date must be before end date")

    # Validate model section
    model_config = config['model']
    required_model_params = [
        'type', 'features', 'lookback_period', 'prediction_horizon',
        'train_split', 'val_split', 'epochs', 'batch_size',
        'learning_rate', 'dropout_rate', 'early_stopping_patience',
        'reduce_lr_patience', 'min_lr', 'lstm_units', 'dense_units',
        'attention_heads', 'attention_key_dim'
    ]
    
    for param in required_model_params:
        if param not in model_config:
            raise ValueError(f"Missing required model parameter: {param}")
    
    if model_config['type'] not in ['lstm', 'gru']:
        raise ValueError("Invalid model type")
    
    if not model_config['features']:
        raise ValueError("Features list cannot be empty")
    
    if not isinstance(model_config['lookback_period'], int):
        raise ValueError("Lookback period must be an integer")
    
    if not isinstance(model_config['prediction_horizon'], int):
        raise ValueError("Prediction horizon must be an integer")
    
    if not 0 < model_config['train_split'] <= 1:
        raise ValueError("Train split must be between 0 and 1")
            
    if not 0 < model_config['val_split'] <= 1:
        raise ValueError("Validation split must be between 0 and 1")
            
    if model_config['train_split'] + model_config['val_split'] > 1:
        raise ValueError("Train split + validation split must be <= 1")
    
    if not isinstance(model_config['lstm_units'], list):
        raise ValueError("LSTM units must be a list")
    
    if not isinstance(model_config['dense_units'], list):
        raise ValueError("Dense units must be a list")
    
    if not isinstance(model_config['attention_heads'], list):
        raise ValueError("Attention heads must be a list")
    
    if len(model_config['attention_heads']) != len(model_config['lstm_units']) - 1:
        raise ValueError("Number of attention heads must match number of LSTM layers - 1")

    # Validate trading section
    if config['trading']['initial_capital'] <= 0:
        raise ValueError("Initial capital must be positive")
    
    if config['trading']['max_positions'] <= 0:
        raise ValueError("Maximum positions must be positive")
    
    if not 0 < config['trading']['position_size'] <= 1:
        raise ValueError("Position size must be between 0 and 1")
    
    if config['trading']['stop_loss_pct'] <= 0:
        raise ValueError("Stop loss percentage must be positive")

    # Validate logging section
    valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if config['logging']['level'] not in valid_log_levels:
        raise ValueError(f"Invalid log level. Must be one of {valid_log_levels}")
    
    try:
        with open(config['logging']['log_file'], 'a') as f:
            pass
    except:
        raise ValueError("Invalid log file path")

    return True

class Config:
    """Configuration class for the trading bot."""
    def __init__(self, config_path=None):
        """Initialize configuration."""
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
                validate_config(config)
                self.data = config['data']
                self.model = config['model']
                self.trading = config['trading']
                self.logging = config['logging']

    @staticmethod
    def get_default_config():
        """Get default configuration."""
        config = Config()
        config.data = {
            'tickers': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
            'start_date': '2010-01-01',
            'end_date': datetime.now().strftime('%Y-%m-%d')
        }
        config.model = {
            'type': 'lstm',
            'features': [
                'close', 'volume', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'atr', 'volatility', 'volume_ma', 'volume_std',
                'open', 'high', 'low', 'returns_lag',
                'ema_fast', 'ema_slow', 'adx', 'obv',
                'stoch_k', 'stoch_d', 'williams_r'
            ],
            'lookback_period': 20,
            'prediction_horizon': 5,
            'train_split': 0.7,
            'val_split': 0.15,
            'epochs': 5,
            'batch_size': 64,
            'learning_rate': 0.001,
            'dropout_rate': 0.2,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5,
            'min_lr': 1e-6,
            'lstm_units': [128, 64, 32],
            'dense_units': [64, 32, 16],
            'attention_heads': [4, 2],
            'attention_key_dim': 32
        }
        config.trading = {
            'initial_capital': 100000,
            'max_positions': 4,
            'position_size': 0.02,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.03,
            'risk_per_trade': 0.01
        }
        config.logging = {
            'level': 'INFO',
            'log_file': 'logs/trading_bot.log'
        }
        return config

    def save(self, config_path):
        """Save configuration to file."""
        config = {
            'data': self.data,
            'model': self.model,
            'trading': self.trading,
            'logging': self.logging
        }
        validate_config(config)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def update(self, path, value):
        """Update configuration value at path."""
        parts = path.split('.')
        if len(parts) == 1:
            if not hasattr(self, parts[0]):
                raise ValueError(f"Invalid configuration path: {path}")
            setattr(self, parts[0], value)
        else:
            section = getattr(self, parts[0])
            if not isinstance(section, dict):
                raise ValueError(f"Invalid configuration path: {path}")
            section[parts[1]] = value

    def validate(self):
        """Validate current configuration."""
        config = {
            'data': self.data,
            'model': self.model,
            'trading': self.trading,
            'logging': self.logging
        }
        try:
            validate_config(config)
            return True
        except Exception as e:
            print(f"Configuration validation failed: {str(e)}")
            return False
