import numpy as np
from config.config import Config
from data.data_loader import DataLoader, load_data
from models.ml_model import MLModel
from strategies.ml_strategy import MLTradingStrategy
from strategies.technical_strategy import TechnicalStrategy
from datetime import datetime, timedelta
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import signal
import sys
from data.data_splitter import split_data
import traceback
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

def setup_logging():
    """Set up logging configuration with both console and file output."""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/trading_bot_{timestamp}.log'
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

def plot_training_history(history: dict, save_path: str = None):
    """Plot training history metrics."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['mae'], label='Training MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_trading_results(data: pd.DataFrame, trades: list, save_path: str = None):
    """Plot trading results with buy/sell signals."""
    plt.figure(figsize=(15, 8))
    
    # Plot price
    plt.plot(data.index, data['close'], label='Price', alpha=0.7)
    
    # Plot buy signals
    buys = [(t['date'], t['price']) for t in trades if t['action'] == 'BUY']
    if buys:
        buy_dates, buy_prices = zip(*buys)
        plt.scatter(buy_dates, buy_prices, color='green', marker='^', 
                   label='Buy', alpha=0.7)
    
    # Plot sell signals
    sells = [(t['date'], t['price']) for t in trades if t['action'] == 'SELL']
    if sells:
        sell_dates, sell_prices = zip(*sells)
        plt.scatter(sell_dates, sell_prices, color='red', marker='v',
                   label='Sell', alpha=0.7)
    
    plt.title('Trading Results')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def run_backtest(strategy, data: pd.DataFrame, predictions: np.ndarray = None) -> tuple:
    """Run backtest for a strategy and return results."""
    signals = strategy.generate_signals(data, predictions)
    trades = strategy.execute_trades(signals, data)
    metrics = strategy.calculate_metrics(trades, data['close'].iloc[-1])
    return trades, metrics

def plot_portfolio_performance(data: pd.DataFrame, trades: list, metrics: dict, save_path: str = None):
    """Plot detailed portfolio performance analysis."""
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Price and Trades
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(data.index, data['close'], label='Price', alpha=0.7, color='blue')
    
    # Plot buy/sell points
    buys = [(t['date'], t['price']) for t in trades if t['action'] == 'BUY']
    sells = [(t['date'], t['price']) for t in trades if t['action'] in ['SELL', 'CLOSE']]
    
    if buys:
        buy_dates, buy_prices = zip(*buys)
        ax1.scatter(buy_dates, buy_prices, color='green', marker='^', 
                   label='Buy', alpha=0.7, s=100)
    
    if sells:
        sell_dates, sell_prices = zip(*sells)
        ax1.scatter(sell_dates, sell_prices, color='red', marker='v',
                   label='Sell', alpha=0.7, s=100)
    
    ax1.set_title('Trading Signals')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # Plot 2: Portfolio Value
    ax2 = plt.subplot(2, 2, 2)
    portfolio_values = calculate_portfolio_values(data, trades, metrics['initial_capital'])
    ax2.plot(data.index, portfolio_values, label='Portfolio Value', color='green')
    ax2.set_title('Portfolio Value Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Portfolio Value ($)')
    
    # Add horizontal line for initial capital
    ax2.axhline(y=metrics['initial_capital'], color='r', linestyle='--', 
                label='Initial Capital')
    ax2.legend()
    
    # Plot 3: Drawdown Analysis
    ax3 = plt.subplot(2, 2, 3)
    drawdowns = calculate_drawdowns(portfolio_values)
    ax3.fill_between(data.index, drawdowns * 100, 0, color='red', alpha=0.3)
    ax3.set_title('Drawdown Analysis')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown (%)')
    
    # Plot 4: Trade Distribution
    ax4 = plt.subplot(2, 2, 4)
    plot_trade_distribution(trades, ax4)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def calculate_portfolio_values(data: pd.DataFrame, trades: list, initial_capital: float) -> np.ndarray:
    """Calculate portfolio value over time."""
    portfolio_values = np.full(len(data), initial_capital)
    current_position = 0
    current_cash = initial_capital
    
    for trade in trades:
        idx = data.index.get_loc(trade['date'])
        
        if trade['action'] == 'BUY':
            current_position = trade['size']
            current_cash -= trade['cost']
        elif trade['action'] in ['SELL', 'CLOSE']:
            current_position = 0
            current_cash += trade['proceeds']
            
        # Update portfolio value from this point forward
        portfolio_values[idx:] = current_cash + \
            (current_position * data['close'].iloc[idx:])
    
    return portfolio_values

def calculate_drawdowns(portfolio_values: np.ndarray) -> np.ndarray:
    """Calculate drawdown percentage over time."""
    rolling_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (rolling_max - portfolio_values) / rolling_max
    return drawdowns

def plot_trade_distribution(trades: list, ax):
    """Plot trade profit distribution."""
    profits = []
    for trade in trades:
        if trade['action'] in ['SELL', 'CLOSE']:
            profit = trade.get('proceeds', 0) - trade.get('cost', 0)
            profits.append(profit)
    
    # Plot the histogram
    if profits:
        sns.histplot(profits, bins=20, ax=ax)
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_title('Trade Profit Distribution')
        ax.set_xlabel('Profit/Loss ($)')
        ax.set_ylabel('Frequency')

def signal_handler(signum, frame):
    print('\nReceived interrupt signal. Cleaning up...')
    sys.exit(0)

def main():
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Set up logging
    logger = setup_logging()
    logger.info("=== Trading Bot Simulation Started ===")
    
    try:
        # Initialize and validate configuration
        config = Config.get_default_config()
        if not config.validate():
            logger.error("Invalid configuration settings")
            return
            
        logger.info(f"Trading symbol: {config.trading.symbol}")
        logger.info(f"Initial capital: ${config.trading.initial_capital:,.2f}")
        
        # Define list of tickers and date range
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        start_date = '2020-01-01'
        end_date = '2023-05-31'
        
        # Load and preprocess data for all tickers
        data = load_data(tickers, start_date, end_date)
        logger.debug(f"Data keys: {data.keys()}")
        
        # Split data into train and test sets for each ticker
        train_data, test_data = split_data(data)

        # Validate train_data and test_data
        for ticker, data in train_data.items():
            if 'X' not in data or 'y' not in data:
                logger.error(f"Missing 'X' or 'y' in train_data for {ticker}")
                continue
            if not isinstance(data['X'], np.ndarray) or not hasattr(data['X'], 'shape'):
                logger.error(f"Invalid 'X' for {ticker} in train_data: {type(data['X'])}")
                continue
            if not isinstance(data['y'], (np.ndarray, pd.Series, list)):
                logger.error(f"Invalid 'y' for {ticker} in train_data: {type(data['y'])}")
                continue
            logger.debug(f"Valid train_data for {ticker}: X shape = {data['X'].shape}, y shape = {data['y'].shape}")

        for ticker, data in test_data.items():
            if 'X' not in data or 'y' not in data:
                logger.error(f"Missing 'X' or 'y' in test_data for {ticker}")
                continue
            if not isinstance(data['X'], np.ndarray) or not hasattr(data['X'], 'shape'):
                logger.error(f"Invalid 'X' for {ticker} in test_data: {type(data['X'])}")
                continue
            if not isinstance(data['y'], (np.ndarray, pd.Series, list)):
                logger.error(f"Invalid 'y' for {ticker} in test_data: {type(data['y'])}")
                continue
            logger.debug(f"Valid test_data for {ticker}: X shape = {data['X'].shape}, y shape = {data['y'].shape}")
        
        # Train models and run backtesting for each ticker
        if 'tickers' in config.data:
            for ticker in config.data['tickers']:
                logger.info(f"Processing {ticker}...")
                if ticker not in train_data:
                    logger.warning(f"Skipping {ticker} due to missing train data.")
                    continue
                
                try:
                    # Extract data
                    X_train = train_data[ticker]['X']
                    y_train = train_data[ticker]['y']
                    X_test = test_data[ticker]['X']
                    y_test = test_data[ticker]['y']

                    # Validate data shapes
                    logger.debug(f"X_train shape: {X_train.shape}")
                    logger.debug(f"y_train shape: {y_train.shape}")
                    logger.debug(f"X_test shape: {X_test.shape}")  
                    logger.debug(f"y_test shape: {y_test.shape}")
                    
                    # Train ML model
                    ml_model = MLModel(config)
                    ml_model.train(X_train, y_train, X_test, y_test)
                    
                    # Generate predictions
                    logger.info(f"X_test shape before predict: {X_test.shape}")
                    predictions = ml_model.predict(X_test)
                    
                    # Run ML strategy backtest
                    ml_strategy = MLTradingStrategy(config, ml_model)
                    ml_results = run_backtest(ml_strategy, X_test, predictions)
                    
                    # Run technical strategy backtest
                    tech_strategy = TechnicalStrategy(config)
                    tech_results = run_backtest(tech_strategy, X_test, predictions)
                    
                    # Evaluate and compare strategies
                    evaluate_strategy(ml_results, tech_results, ticker)
                    
                except Exception as e:
                    logger.error(f"Error in main for ticker {ticker}: {e}")
                    traceback.print_exc()
        else:
            logger.error("No tickers found in config.data")
            
        logger.info("=== Trading Bot Simulation Completed ===")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
