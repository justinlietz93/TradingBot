# Configuration settings

import os
from dataclasses import dataclass, field
from typing import Dict, List
import logging

@dataclass
class ModelConfig:
    model_type: str
    features: List[str]
    target: str
    train_split: float = 0.8
    lookback_period: int = 60
    prediction_horizon: int = 5
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 10
    learning_rate: float = 0.001

@dataclass
class TradingConfig:
    symbol: str
    initial_capital: float
    risk_per_trade: float
    position_sizing_method: str
    max_positions: int
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    min_trade_size: int = 1

@dataclass
class Config:
    model: ModelConfig
    trading: TradingConfig
    data_dir: str = os.path.join(os.path.dirname(__file__), '../data')
    data: Dict[str, List[str]] = field(default_factory=lambda: {'tickers': []})
    
    @classmethod
    def get_default_config(cls) -> 'Config':
        model_config = ModelConfig(
            model_type='lstm',
            features=[
                'close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower',
                'volatility', 'volume_ma', 'open', 'high', 'low', 'returns_lag'
            ],
            target='returns',
            train_split=0.8,
            lookback_period=60,
            prediction_horizon=5,
            batch_size=32,
            epochs=100,
            early_stopping_patience=15,
            learning_rate=0.001
        )
        
        trading_config = TradingConfig(
            symbol='AAPL',
            initial_capital=100000.0,
            risk_per_trade=0.02,
            position_sizing_method='risk_based',
            max_positions=4,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            min_trade_size=1
        )
        
        data_config = {
            'tickers': ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        }
        logging.info(f"Default config tickers: {data_config['tickers']}")
        
        return cls(model=model_config, trading=trading_config, data=data_config)
        
    def validate(self) -> bool:
        """Validate configuration settings."""
        try:
            logging.info(f"Validating config tickers: {self.data['tickers']}")
            
            assert 0 < self.model.train_split < 1, "Train split must be between 0 and 1"
            assert self.model.lookback_period > 0, "Lookback period must be positive"
            assert self.model.prediction_horizon > 0, "Prediction horizon must be positive"
            assert self.model.batch_size > 0, "Batch size must be positive"
            assert self.model.epochs > 0, "Number of epochs must be positive"
            assert self.model.learning_rate > 0, "Learning rate must be positive"
            
            assert self.trading.initial_capital > 0, "Initial capital must be positive"
            assert self.trading.risk_per_trade > 0, "Risk per trade must be positive"
            assert self.trading.max_positions > 0, "Max positions must be positive"
            assert self.trading.stop_loss_pct > 0, "Stop loss must be positive"
            assert self.trading.take_profit_pct > 0, "Take profit must be positive"
            
            assert self.data['tickers'], "Tickers list must not be empty"
            for ticker in self.data['tickers']:
                assert isinstance(ticker, str), "Ticker must be a string"
                assert len(ticker) > 0, "Ticker must not be an empty string"
            
            return True
        except AssertionError as e:
            logging.error(f"Configuration validation failed: {str(e)}")
            return False
