import os
from datetime import datetime
import logging
import signal
import sys
import pandas as pd
import numpy as np
from config.config import Config
from data.data_loader import DataLoader
from data.data_splitter import split_data
from models.ml_model import MLModel
from strategies.ml_strategy import MLTradingStrategy

def setup_logging():
    """Set up logging configuration."""
    # Create logs directory structure
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/fit', exist_ok=True)
    
    log_filename = f'logs/trading_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def process_ticker(ticker, data_loader, config):
    """Process a single ticker's training and backtesting."""
    try:
        logger.info(f"Processing {ticker}...")
        
        # Validate data availability
        if ticker not in data_loader.data:
            logger.error(f"No data available for {ticker}")
            return None, None
            
        # Prepare data sequences
        prepared_data = data_loader.prepare_data(
            lookback_period=config.model['lookback_period'],
            horizon=config.model['prediction_horizon']
        )
        
        if ticker not in prepared_data:
            logger.error(f"Failed to prepare data for {ticker}")
            return None, None
            
        ticker_data = prepared_data[ticker]
        
        # Split data into train, validation, and test sets
        train_data, val_data, test_data = split_data(
            {ticker: ticker_data},
            train_split=config.model['train_split'],
            val_split=config.model['val_split']
        )
        
        # Get data for the current ticker
        X_train, y_train = train_data[ticker]['X'], train_data[ticker]['y']
        X_val, y_val = val_data[ticker]['X'], val_data[ticker]['y']
        X_test, y_test = test_data[ticker]['X'], test_data[ticker]['y']
        
        # Initialize and train ML model
        input_shape = X_train.shape[1:]
        horizon = config.model['prediction_horizon']
        model = MLModel(input_shape=input_shape, horizon=horizon, config=config.model)
        
        logger.info(f"Training ML model for {ticker}...")
        history = model.train(
            X_train, y_train,
            X_val, y_val
        )
        
        # Log training metrics
        logger.info(f"Training history for {ticker}:")
        for metric, values in history.items():
            logger.info(f"  {metric}: {values[-1]:.4f}")
        
        # Make predictions on test set
        predictions = model.predict(X_test)
        
        # Initialize strategy with ML predictions
        strategy = MLTradingStrategy(config)
        
        # Prepare data for backtesting
        backtest_data = data_loader.data[ticker].iloc[-len(X_test):]
        
        # Run backtest with full predictions
        trades, metrics = strategy.backtest(backtest_data, predictions)
        
        if trades and metrics:
            logger.info(f"Backtest completed for {ticker}. Metrics:")
            logger.info(f"  - Total trades: {metrics['total_trades']}")
            logger.info(f"  - Win rate: {metrics['win_rate']:.2%}")
            logger.info(f"  - Average return: {metrics['avg_return']:.4%}")
            logger.info(f"  - Total return: {metrics['total_return']:.2%}")
            logger.info(f"  - Max drawdown: {metrics['max_drawdown']:.2%}")
            logger.info(f"  - Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"  - Direction accuracy: {metrics['direction_accuracy']:.2%}")
            return metrics, trades
        else:
            logger.warning(f"No valid backtest results for {ticker}")
            return None, None
        
    except Exception as e:
        logger.error(f"Error processing {ticker}: {str(e)}")
        return None, None

def main():
    logger.info("=== Trading Bot Simulation Started ===")
    try:
        # Load configuration
        config = Config.get_default_config()
        if not config.validate():
            logger.error("Invalid configuration settings.")
            return
        
        tickers = config.data['tickers']
        logger.info(f"Tickers: {tickers}")
        
        # Initialize data loader
        data_loader = DataLoader(
            tickers=tickers,
            start_date=config.data['start_date'],
            end_date=config.data['end_date']
        )
        
        # Load and preprocess data
        data_loader.load_data()
        
        # Process each ticker
        results = {}
        success_count = 0
        for ticker in tickers:
            try:
                metrics, trades = process_ticker(ticker, data_loader, config)
                if metrics and trades:
                    results[ticker] = {
                        'metrics': metrics,
                        'trades': trades
                    }
                    success_count += 1
                else:
                    logger.warning(f"No results available for {ticker}")
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {str(e)}")
                continue

        # Log final summary
        logger.info(f"Successfully processed {success_count} out of {len(tickers)} tickers")
        if success_count == 0:
            logger.error("No tickers were successfully processed")
            return
            
        logger.info("=== Trading Bot Simulation Completed ===")
        return results

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.info("=== Trading Bot Simulation Failed ===")
        raise

if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
