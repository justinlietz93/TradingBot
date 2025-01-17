# Trading Bot

## In development

A sophisticated machine learning-based trading bot that uses LSTM neural networks with attention mechanisms to predict stock price movements and execute trades.

## Features

- Advanced data collection and preprocessing from Yahoo Finance
- Comprehensive technical indicator calculation and feature engineering
- Enhanced LSTM model with attention mechanism and residual connections
- Hybrid trading strategy combining ML predictions with technical analysis
- Robust backtesting engine with position management
- Dynamic risk management with adaptive stop-loss and take-profit
- Detailed performance metrics and visualization
- Highly configurable trading parameters
- Real-time logging and monitoring

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading_bot.git
cd trading_bot
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.json` to configure:
- Trading pairs and timeframes
- Model hyperparameters
- Trading strategy parameters
- Risk management settings
- Logging preferences

## Usage

1. Run the trading bot:
```bash
python main.py
```

2. Monitor the logs:
- Check `logs/trading_bot.log` for general execution logs
- Check `logs/training/*.log` for model training logs
- Check `logs/backtest/*.log` for backtest results

## Testing

The project includes comprehensive unit tests and integration tests:

1. Run all tests with coverage reporting:
```bash
python tests/run_tests.py
```

2. View the coverage report:
```bash
open coverage_report/index.html  # On Windows: start coverage_report/index.html
```

Test suite includes:
- Model architecture and training tests
- Data processing and feature engineering tests
- Strategy implementation and signal generation tests
- Backtesting engine and performance metrics tests
- Configuration validation and error handling tests

## Project Structure

```
trading_bot/
│
├── config/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   └── logging_config.py   # Logging configuration
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py      # Data fetching and preprocessing
│   ├── feature_engineer.py # Feature engineering
│   └── data_validator.py   # Data validation utilities
│
├── models/
│   ├── __init__.py
│   ├── base_model.py       # Abstract base class for models
│   ├── ml_model.py         # Enhanced LSTM model implementation
│   └── technical_model.py  # Technical analysis model
│
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py    # Abstract base class for strategies
│   ├── ml_strategy.py      # Hybrid ML-based strategy
│   └── technical_strategy.py # Technical analysis strategy
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py          # Performance metrics
│   ├── risk_manager.py     # Risk management utilities
│   ├── visualization.py    # Plotting and visualization
│   └── validators.py       # Input validation utilities
│
├── tests/
│   ├── __init__.py
│   ├── test_data/         # Test data files
│   ├── test_models.py     # Model tests
│   ├── test_strategies.py # Strategy tests
│   └── test_utils.py      # Utility tests
│
├── logs/
│   ├── trading/           # Trading execution logs
│   ├── training/          # Model training logs
│   └── backtest/          # Backtest result logs
│
├── documentation/
│   ├── ARCHITECTURE.md    # System architecture
│   ├── Program_Flow.md    # Program execution flow
│   ├── codebase_analysis.md # Code documentation
│   ├── change_log.md      # Version history
│   └── api/              # API documentation
│
├── notebooks/
│   ├── analysis/         # Analysis notebooks
│   ├── research/         # Research notebooks
│   └── visualization/    # Visualization notebooks
│
├── requirements/
│   ├── base.txt         # Base dependencies
│   ├── dev.txt          # Development dependencies
│   └── test.txt         # Testing dependencies
│
├── scripts/
│   ├── setup.py         # Setup script
│   ├── install.py       # Installation script
│   └── run_tests.py     # Test runner
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── main.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure everything works
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
