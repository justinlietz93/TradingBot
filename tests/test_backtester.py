import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from backtesting.backtester import Backtester
from backtesting.position import Position
from backtesting.metrics import calculate_metrics

class TestBacktesting(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.price_data = pd.DataFrame({
            'close': np.random.random(100) * 100 + 50,
            'high': np.random.random(100) * 100 + 60,
            'low': np.random.random(100) * 100 + 40,
            'volume': np.random.random(100) * 1000000
        }, index=dates)
        
        # Create sample signals
        self.signals = pd.DataFrame({
            'buy_signal': np.random.choice([0, 1], size=100, p=[0.9, 0.1]),
            'sell_signal': np.random.choice([0, 1], size=100, p=[0.9, 0.1])
        }, index=dates)
        
        # Initialize backtester
        self.backtester = Backtester(
            initial_capital=100000,
            position_size=0.02,
            max_positions=5,
            stop_loss_pct=0.02,
            take_profit_pct=0.03
        )

    def test_backtester_initialization(self):
        """Test backtester initialization."""
        self.assertEqual(self.backtester.initial_capital, 100000)
        self.assertEqual(self.backtester.current_capital, 100000)
        self.assertEqual(len(self.backtester.positions), 0)
        self.assertEqual(len(self.backtester.closed_positions), 0)

    def test_position_creation(self):
        """Test position creation and management."""
        # Create a new position
        entry_price = 100
        position_size = 10
        stop_loss = 98
        take_profit = 103
        
        position = Position(
            ticker='AAPL',
            entry_price=entry_price,
            size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now()
        )
        
        self.assertEqual(position.ticker, 'AAPL')
        self.assertEqual(position.entry_price, entry_price)
        self.assertEqual(position.size, position_size)
        self.assertEqual(position.stop_loss, stop_loss)
        self.assertEqual(position.take_profit, take_profit)
        self.assertFalse(position.is_closed)

    def test_position_update(self):
        """Test position update logic."""
        position = Position(
            ticker='AAPL',
            entry_price=100,
            size=10,
            stop_loss=98,
            take_profit=103,
            entry_time=datetime.now()
        )
        
        # Test stop loss hit
        position.update(current_price=97)
        self.assertTrue(position.is_closed)
        self.assertEqual(position.exit_price, 97)
        self.assertLess(position.pnl, 0)
        
        # Test take profit hit
        position = Position(
            ticker='AAPL',
            entry_price=100,
            size=10,
            stop_loss=98,
            take_profit=103,
            entry_time=datetime.now()
        )
        position.update(current_price=104)
        self.assertTrue(position.is_closed)
        self.assertEqual(position.exit_price, 104)
        self.assertGreater(position.pnl, 0)

    def test_backtesting_run(self):
        """Test full backtesting run."""
        results = self.backtester.run(
            ticker='AAPL',
            price_data=self.price_data,
            signals=self.signals
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('equity_curve', results)
        self.assertIn('trades', results)
        self.assertIn('metrics', results)
        
        # Check equity curve
        self.assertEqual(len(results['equity_curve']), len(self.price_data))
        self.assertTrue(all(isinstance(x, (int, float)) for x in results['equity_curve']))
        
        # Check trades
        self.assertGreaterEqual(len(results['trades']), 0)
        if len(results['trades']) > 0:
            trade = results['trades'][0]
            self.assertIn('entry_price', trade)
            self.assertIn('exit_price', trade)
            self.assertIn('pnl', trade)
            self.assertIn('entry_time', trade)
            self.assertIn('exit_time', trade)

    def test_metrics_calculation(self):
        """Test performance metrics calculation."""
        # Create sample returns
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        equity_curve = (1 + returns).cumprod() * 100000
        
        metrics = calculate_metrics(
            returns=returns,
            equity_curve=equity_curve,
            risk_free_rate=0.02
        )
        
        # Check required metrics
        required_metrics = [
            'total_return',
            'annualized_return',
            'sharpe_ratio',
            'max_drawdown',
            'win_rate',
            'profit_factor'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))

    def test_risk_management(self):
        """Test risk management rules."""
        # Test maximum positions limit
        for _ in range(6):  # Try to open 6 positions (max is 5)
            self.backtester.open_position(
                ticker='AAPL',
                entry_price=100,
                size=10,
                stop_loss=98,
                take_profit=103
            )
        
        self.assertLessEqual(len(self.backtester.positions), 5)
        
        # Test position sizing
        position = self.backtester.open_position(
            ticker='AAPL',
            entry_price=100,
            size=self.backtester.calculate_position_size(100),
            stop_loss=98,
            take_profit=103
        )
        
        if position:
            # Check that position size doesn't exceed max risk per trade
            max_risk = position.size * (position.entry_price - position.stop_loss)
            self.assertLessEqual(max_risk, self.backtester.current_capital * 0.02)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty price data
        empty_data = pd.DataFrame()
        empty_signals = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            self.backtester.run('AAPL', empty_data, empty_signals)
        
        # Test with mismatched data lengths
        mismatched_signals = pd.DataFrame({
            'buy_signal': [0, 1],
            'sell_signal': [0, 1]
        })
        
        with self.assertRaises(ValueError):
            self.backtester.run('AAPL', self.price_data, mismatched_signals)
        
        # Test with invalid position parameters
        with self.assertRaises(ValueError):
            self.backtester.open_position(
                ticker='AAPL',
                entry_price=-100,  # Invalid negative price
                size=10,
                stop_loss=98,
                take_profit=103
            )

if __name__ == '__main__':
    unittest.main() 