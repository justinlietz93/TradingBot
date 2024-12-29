import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from strategies.ml_strategy import MLTradingStrategy
from config.config import Config

class MockConfig:
    """Mock configuration for testing."""
    class ModelConfig:
        def __init__(self):
            self.lookback_period = 60
            self.features = ['close', 'volume', 'rsi', 'macd']
            
    class TradingConfig:
        def __init__(self):
            self.initial_capital = 100000.0
            self.risk_per_trade = 0.02
            self.symbol = 'TEST'
            
    def __init__(self):
        self.model = self.ModelConfig()
        self.trading = self.TradingConfig()

class TestMLTradingStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = MockConfig()
        self.strategy = MLTradingStrategy(self.config)
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.data = pd.DataFrame({
            'close': np.random.random(100) * 100 + 50,
            'open': np.random.random(100) * 100 + 50,
            'high': np.random.random(100) * 100 + 50,
            'low': np.random.random(100) * 100 + 50,
            'volume': np.random.random(100) * 1000000
        }, index=dates)

    def test_feature_calculation(self):
        """Test technical indicator calculation."""
        features = self.strategy._calculate_features(self.data)
        
        # Check if all required features are calculated
        required_features = [
            'returns', 'returns_lag', 'volume_ma', 'volume_std',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'atr', 'volatility', 'ema_fast', 'ema_slow',
            'adx', 'obv', 'stoch_k', 'stoch_d', 'williams_r'
        ]
        
        for feature in required_features:
            self.assertIn(feature, features.columns)
            self.assertTrue(features[feature].notna().any())

    def test_signal_generation(self):
        """Test trading signal generation."""
        features = self.strategy._calculate_features(self.data)
        signals = self.strategy.generate_signals(features)
        
        # Check signal DataFrame structure
        self.assertIn('buy', signals.columns)
        self.assertIn('sell', signals.columns)
        
        # Check signal values
        self.assertTrue(all(signals['buy'].isin([0, 1])))
        self.assertTrue(all(signals['sell'].isin([0, 1])))
        
        # Check that we don't have simultaneous buy and sell signals
        self.assertTrue(all((signals['buy'] + signals['sell']) <= 1))

    def test_position_management(self):
        """Test position creation and updates."""
        # Test position creation
        current_price = 100.0
        current_date = datetime.now()
        
        # Test long position
        long_position = self.strategy.execute_trade(1, current_price, current_date)
        self.assertEqual(long_position['direction'], 1)
        self.assertLess(long_position['position_size'], self.strategy.max_position_size)
        
        # Test short position
        short_position = self.strategy.execute_trade(-1, current_price, current_date)
        self.assertEqual(short_position['direction'], -1)
        self.assertLess(short_position['position_size'], self.strategy.max_position_size)
        
        # Test position update
        # Test stop loss
        position = self.strategy.execute_trade(1, 100.0, current_date)
        stop_loss_price = position['stop_loss']
        updated_position = self.strategy.update_position(position, stop_loss_price, current_date + timedelta(days=1))
        self.assertIsNotNone(updated_position)
        self.assertIn('pnl', updated_position)
        
        # Test take profit
        position = self.strategy.execute_trade(1, 100.0, current_date)
        take_profit_price = position['take_profit']
        updated_position = self.strategy.update_position(position, take_profit_price, current_date + timedelta(days=1))
        self.assertIsNotNone(updated_position)
        self.assertIn('pnl', updated_position)

    def test_backtest(self):
        """Test backtesting functionality."""
        features = self.strategy._calculate_features(self.data)
        trades, metrics = self.strategy.backtest(features)
        
        # Check trades list
        self.assertIsInstance(trades, list)
        if trades:  # If any trades were made
            self.assertIn('entry_price', trades[0])
            self.assertIn('exit_price', trades[0])
            self.assertIn('pnl', trades[0])
        
        # Check metrics dictionary
        required_metrics = [
            'total_trades', 'win_rate', 'avg_return', 'total_return',
            'max_drawdown', 'sharpe_ratio', 'direction_accuracy'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))

    def test_risk_management(self):
        """Test risk management calculations."""
        # Test ATR calculation
        atr = self.strategy.calculate_atr(100.0)
        self.assertGreater(atr, 0)
        
        # Test position sizing
        position = self.strategy.execute_trade(1, 100.0, datetime.now())
        position_value = position['position_size'] * 100.0
        
        # Check if position size respects risk limits
        self.assertLess(position_value, self.config.trading.initial_capital * self.strategy.max_position_size)
        
        # Check stop loss and take profit levels
        self.assertLess(position['stop_loss'], position['entry_price'])  # For long position
        self.assertGreater(position['take_profit'], position['entry_price'])  # For long position

if __name__ == '__main__':
    unittest.main()
