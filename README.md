# Trading Bot

A machine learning-based trading bot that uses LSTM neural networks to predict stock price movements and execute trades.

## Features

- Data collection from Yahoo Finance
- Technical indicator calculation
- LSTM model with attention mechanism for price prediction
- Backtesting engine with position management
- Risk management with stop-loss and take-profit
- Performance metrics calculation
- Configurable trading parameters

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
- Trading pairs
- Time period
- Model parameters
- Trading parameters
- Risk management settings

## Usage

Run the trading bot:
```bash
python main.py
```

## Testing

The project includes comprehensive unit tests for all major components. To run the tests:

1. Run all tests with coverage reporting:
```bash
python tests/run_tests.py
```

2. View the coverage report in your browser:
```bash
open coverage_report/index.html  # On Windows: start coverage_report/index.html
```

The test suite includes:
- Model architecture tests
- Data processing tests
- Strategy implementation tests
- Backtesting engine tests
- Configuration validation tests

## Project Structure

```
trading_bot/
├── config/
│   ├── config.json
│   └── config.py
├── data/
│   ├── data_loader.py
│   └── data_splitter.py
├── models/
│   └── ml_model.py
├── strategies/
│   └── ml_strategy.py
├── backtesting/
│   ├── backtester.py
│   ├── position.py
│   └── metrics.py
├── tests/
│   ├── test_models.py
│   ├── test_data.py
│   ├── test_strategies.py
│   ├── test_backtester.py
│   └── test_config.py
├── logs/
├── main.py
└── requirements.txt
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure everything works
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
