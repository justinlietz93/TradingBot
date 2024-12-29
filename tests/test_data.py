import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data.data_loader import DataLoader
from data.data_splitter import split_data
from config.config import Config

class MockConfig:
    """Mock configuration for testing."""
    class ModelConfig:
        def __init__(self):
            self.lookback_period = 60
            self.features = ['close', 'volume', 'rsi', 'macd']
            self.target = 'returns'
            self.train_split = 0.8
            
    class TradingConfig:
        def __init__(self):
            self.symbol = 'TEST'
            
    def __init__(self):
        self.model = self.ModelConfig()
        self.trading = self.TradingConfig()
        self.data = {'tickers': ['AAPL', 'GOOGL']}

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = MockConfig()
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        self.sample_data = {}
        for ticker in ['AAPL', 'GOOGL']:
            self.sample_data[ticker] = pd.DataFrame({
                'close': np.random.random(200) * 100 + 50,
                'open': np.random.random(200) * 100 + 50,
                'high': np.random.random(200) * 100 + 50,
                'low': np.random.random(200) * 100 + 50,
                'volume': np.random.random(200) * 1000000
            }, index=dates)

    def test_data_loader_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader(['AAPL', 'GOOGL'], '2023-01-01', '2023-12-31')
        self.assertEqual(loader.tickers, ['AAPL', 'GOOGL'])
        self.assertIsInstance(loader.start_date, str)
        self.assertIsInstance(loader.end_date, str)

    def test_data_preprocessing(self):
        """Test data preprocessing functionality."""
        loader = DataLoader(['AAPL'], '2023-01-01', '2023-12-31')
        
        # Test with sample data
        processed_data = loader._preprocess_data(self.sample_data['AAPL'], 'AAPL')
        
        # Check if required columns exist
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        for col in required_columns:
            self.assertIn(col, processed_data.columns)
        
        # Check for NaN values
        self.assertFalse(processed_data.isnull().any().any())

    def test_sequence_creation(self):
        """Test sequence creation for LSTM input."""
        loader = DataLoader(['AAPL'], '2023-01-01', '2023-12-31')
        data = self.sample_data['AAPL']
        
        # Create sequences
        X, y = loader._create_sequences(data, lookback_period=60, horizon=5)
        
        # Check shapes
        self.assertEqual(len(X), len(y))
        self.assertEqual(X.shape[1], 60)  # lookback_period
        self.assertEqual(y.shape[1], 5)   # prediction_horizon

    def test_data_splitting(self):
        """Test data splitting functionality."""
        # Prepare data dictionary
        data = {}
        for ticker in ['AAPL', 'GOOGL']:
            sequences = np.random.random((100, 60, 12))  # 100 samples, 60 timesteps, 12 features
            targets = np.random.random((100, 5))        # 100 samples, 5 target values
            data[ticker] = {
                'X': sequences,
                'y': targets
            }
        
        # Split data
        train_data, test_data = split_data(data, train_split=0.8)
        
        # Check if all tickers are present
        self.assertEqual(set(train_data.keys()), set(test_data.keys()))
        
        for ticker in ['AAPL', 'GOOGL']:
            # Check train data
            self.assertIn('X', train_data[ticker])
            self.assertIn('y', train_data[ticker])
            
            # Check test data
            self.assertIn('X', test_data[ticker])
            self.assertIn('y', test_data[ticker])
            
            # Check shapes
            self.assertEqual(len(train_data[ticker]['X']), len(train_data[ticker]['y']))
            self.assertEqual(len(test_data[ticker]['X']), len(test_data[ticker]['y']))
            
            # Check split ratio
            total_samples = len(train_data[ticker]['X']) + len(test_data[ticker]['X'])
            train_ratio = len(train_data[ticker]['X']) / total_samples
            self.assertAlmostEqual(train_ratio, 0.8, places=1)

    def test_data_validation(self):
        """Test data validation checks."""
        loader = DataLoader(['AAPL'], '2023-01-01', '2023-12-31')
        
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'close': [None] * 10,
            'volume': range(10)
        })
        
        with self.assertRaises(ValueError):
            loader._validate_data(invalid_data)
        
        # Test with missing columns
        incomplete_data = pd.DataFrame({
            'close': range(10)
        })
        
        with self.assertRaises(ValueError):
            loader._validate_data(incomplete_data)

    def test_feature_engineering(self):
        """Test feature engineering functionality."""
        loader = DataLoader(['AAPL'], '2023-01-01', '2023-12-31')
        data = self.sample_data['AAPL']
        
        # Add technical indicators
        features = loader._add_technical_indicators(data)
        
        # Check if technical indicators are calculated
        technical_indicators = ['rsi', 'macd', 'bb_upper', 'bb_lower']
        for indicator in technical_indicators:
            self.assertIn(indicator, features.columns)
            self.assertTrue(features[indicator].notna().any())

if __name__ == '__main__':
    unittest.main()
