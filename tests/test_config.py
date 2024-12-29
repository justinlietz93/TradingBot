import unittest
import os
import tempfile
import json
from config.config import Config, validate_config

class TestConfig(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample valid configuration
        self.valid_config = {
            "data": {
                "tickers": ["AAPL", "GOOGL", "MSFT"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "data_source": "yahoo"
            },
            "model": {
                "type": "lstm",
                "lookback_period": 60,
                "prediction_horizon": 5,
                "features": ["close", "volume", "rsi", "macd"],
                "target": "returns",
                "train_split": 0.8,
                "batch_size": 32,
                "epochs": 50,
                "learning_rate": 0.001
            },
            "trading": {
                "initial_capital": 100000,
                "position_size": 0.02,
                "max_positions": 5,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.03,
                "risk_free_rate": 0.02
            },
            "logging": {
                "level": "INFO",
                "log_file": "trading_bot.log"
            }
        }

    def test_config_initialization(self):
        """Test configuration initialization."""
        # Write valid config to temporary file
        config_path = os.path.join(self.temp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        # Load config
        config = Config(config_path)
        
        # Check if all sections are present
        self.assertTrue(hasattr(config, 'data'))
        self.assertTrue(hasattr(config, 'model'))
        self.assertTrue(hasattr(config, 'trading'))
        self.assertTrue(hasattr(config, 'logging'))
        
        # Check specific values
        self.assertEqual(config.data['tickers'], ["AAPL", "GOOGL", "MSFT"])
        self.assertEqual(config.model['lookback_period'], 60)
        self.assertEqual(config.trading['initial_capital'], 100000)
        self.assertEqual(config.logging['level'], "INFO")

    def test_config_validation(self):
        """Test configuration validation."""
        # Test with valid config
        self.assertTrue(validate_config(self.valid_config))
        
        # Test with missing required section
        invalid_config = self.valid_config.copy()
        del invalid_config['model']
        with self.assertRaises(ValueError):
            validate_config(invalid_config)
        
        # Test with invalid data type
        invalid_config = self.valid_config.copy()
        invalid_config['model']['lookback_period'] = "60"  # Should be int
        with self.assertRaises(ValueError):
            validate_config(invalid_config)
        
        # Test with invalid value range
        invalid_config = self.valid_config.copy()
        invalid_config['trading']['position_size'] = 1.5  # Should be <= 1
        with self.assertRaises(ValueError):
            validate_config(invalid_config)

    def test_config_data_section(self):
        """Test data section validation."""
        # Test with invalid ticker format
        invalid_config = self.valid_config.copy()
        invalid_config['data']['tickers'] = "AAPL"  # Should be list
        with self.assertRaises(ValueError):
            validate_config(invalid_config)
        
        # Test with invalid date format
        invalid_config = self.valid_config.copy()
        invalid_config['data']['start_date'] = "2023/01/01"  # Invalid format
        with self.assertRaises(ValueError):
            validate_config(invalid_config)
        
        # Test with end date before start date
        invalid_config = self.valid_config.copy()
        invalid_config['data']['start_date'] = "2023-12-31"
        invalid_config['data']['end_date'] = "2023-01-01"
        with self.assertRaises(ValueError):
            validate_config(invalid_config)

    def test_config_model_section(self):
        """Test model section validation."""
        # Test with invalid model type
        invalid_config = self.valid_config.copy()
        invalid_config['model']['type'] = "invalid_model"
        with self.assertRaises(ValueError):
            validate_config(invalid_config)
        
        # Test with invalid feature list
        invalid_config = self.valid_config.copy()
        invalid_config['model']['features'] = []  # Empty list
        with self.assertRaises(ValueError):
            validate_config(invalid_config)
        
        # Test with invalid training parameters
        invalid_config = self.valid_config.copy()
        invalid_config['model']['train_split'] = 1.5  # Should be <= 1
        with self.assertRaises(ValueError):
            validate_config(invalid_config)

    def test_config_trading_section(self):
        """Test trading section validation."""
        # Test with invalid initial capital
        invalid_config = self.valid_config.copy()
        invalid_config['trading']['initial_capital'] = -100000
        with self.assertRaises(ValueError):
            validate_config(invalid_config)
        
        # Test with invalid position parameters
        invalid_config = self.valid_config.copy()
        invalid_config['trading']['max_positions'] = 0
        with self.assertRaises(ValueError):
            validate_config(invalid_config)
        
        # Test with invalid risk parameters
        invalid_config = self.valid_config.copy()
        invalid_config['trading']['stop_loss_pct'] = -0.02
        with self.assertRaises(ValueError):
            validate_config(invalid_config)

    def test_config_logging_section(self):
        """Test logging section validation."""
        # Test with invalid log level
        invalid_config = self.valid_config.copy()
        invalid_config['logging']['level'] = "INVALID_LEVEL"
        with self.assertRaises(ValueError):
            validate_config(invalid_config)
        
        # Test with invalid log file path
        invalid_config = self.valid_config.copy()
        invalid_config['logging']['log_file'] = "/invalid/path/log.txt"
        with self.assertRaises(ValueError):
            validate_config(invalid_config)

    def test_config_save_load(self):
        """Test configuration save and load functionality."""
        # Save config to file
        config_path = os.path.join(self.temp_dir, 'test_config.json')
        config = Config(None)
        config.data = self.valid_config['data']
        config.model = self.valid_config['model']
        config.trading = self.valid_config['trading']
        config.logging = self.valid_config['logging']
        config.save(config_path)
        
        # Load config from file
        loaded_config = Config(config_path)
        
        # Compare original and loaded configs
        self.assertEqual(config.data, loaded_config.data)
        self.assertEqual(config.model, loaded_config.model)
        self.assertEqual(config.trading, loaded_config.trading)
        self.assertEqual(config.logging, loaded_config.logging)

    def test_config_update(self):
        """Test configuration update functionality."""
        config = Config(None)
        config.data = self.valid_config['data']
        
        # Update single value
        config.update('data.start_date', '2023-02-01')
        self.assertEqual(config.data['start_date'], '2023-02-01')
        
        # Update nested dictionary
        new_model_config = {
            'type': 'gru',
            'lookback_period': 30
        }
        config.update('model', new_model_config)
        self.assertEqual(config.model['type'], 'gru')
        self.assertEqual(config.model['lookback_period'], 30)
        
        # Test invalid update
        with self.assertRaises(ValueError):
            config.update('invalid.path', 'value')

    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory and its contents
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)

if __name__ == '__main__':
    unittest.main() 