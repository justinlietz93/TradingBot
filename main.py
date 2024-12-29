"""
Main entry point for the trading bot.
"""
import os
from datetime import datetime
import logging
import signal
import sys
import pandas as pd
import numpy as np
import traceback
import argparse
import tensorflow as tf
from config.config import Config
from data.data_loader import DataLoader
from data.data_splitter import split_data
from models.ml_model import MLModel
from strategies.ml_strategy import MLTradingStrategy
from pathlib import Path

def setup_logging():
    """Set up logging configuration."""
    # Create logs directory structure
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/fit', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/trading_bot_{timestamp}.log'
    error_filename = f'logs/trading_bot_errors_{timestamp}.log'
    
    # Configure main logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Configure error logger
    error_logger = logging.getLogger('error_logger')
    error_logger.setLevel(logging.ERROR)
    error_handler = logging.FileHandler(error_filename, encoding='utf-8')
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(exc_info)s'))
    error_logger.addHandler(error_handler)
    
    return logging.getLogger(__name__), error_logger

logger, error_logger = setup_logging()

def process_ticker(ticker, data_loader, config, model_path=None):
    """Process a single ticker's training and backtesting."""
    try:
        logger.info(f"Processing {ticker}...")
        
        # Validate data availability
        if ticker not in data_loader.data:
            error_logger.error(f"No data available for {ticker}")
            return None, None
            
        # Prepare data sequences
        try:
            prepared_data = data_loader.prepare_data(
                lookback_period=config.model['lookback_period'],
                horizon=config.model['prediction_horizon']
            )
        except Exception as e:
            error_logger.error(f"Failed to prepare data for {ticker}: {str(e)}\n{traceback.format_exc()}")
            return None, None
        
        if ticker not in prepared_data:
            error_logger.error(f"Failed to prepare data for {ticker}: Ticker not found in prepared data")
            return None, None
            
        ticker_data = prepared_data[ticker]
        
        # Split data into train, validation, and test sets
        try:
            train_data, val_data, test_data = split_data(
                {ticker: ticker_data},
                train_split=config.model['train_split'],
                val_split=config.model['val_split']
            )
        except Exception as e:
            error_logger.error(f"Failed to split data for {ticker}: {str(e)}\n{traceback.format_exc()}")
            return None, None
        
        # Get data for the current ticker
        X_train, y_train = train_data[ticker]['X'], train_data[ticker]['y']
        X_val, y_val = val_data[ticker]['X'], val_data[ticker]['y']
        X_test, y_test = test_data[ticker]['X'], test_data[ticker]['y']
        
        # Initialize ML model
        try:
            input_shape = X_train.shape[1:]
            horizon = config.model['prediction_horizon']
            model = MLModel(input_shape=input_shape, horizon=horizon, config=config.model)
            
            if model_path:
                try:
                    # Load pre-trained model
                    logger.info(f"Loading pre-trained model from {model_path}")
                    model.model = tf.keras.models.load_model(model_path, compile=False)
                    model.model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=config.model['learning_rate']),
                        loss=model.combined_loss,
                        metrics=[model.direction_accuracy]
                    )
                    logger.info("Successfully loaded and compiled pre-trained model")
                except Exception as e:
                    error_logger.error(f"Failed to load pre-trained model: {str(e)}\n{traceback.format_exc()}")
                    return None, None
            else:
                # Train new model
                logger.info(f"Training ML model for {ticker}...")
                history = model.train(
                    X_train, y_train,
                    X_val, y_val
                )
                
                # Check if training was successful
                if history is None:
                    error_logger.error(f"Training failed for {ticker}: No history returned")
                    return None, None
                
                # Log training metrics
                logger.info(f"Training history for {ticker}:")
                for metric, values in history.items():
                    logger.info(f"  {metric}: {values[-1]:.4f}")
            
            # Make predictions on test set
            try:
                predictions = model.predict(X_test)
                if predictions is None:
                    error_logger.error(f"Prediction failed for {ticker}: No predictions returned")
                    return None, None
            except Exception as e:
                error_logger.error(f"Failed to make predictions for {ticker}: {str(e)}\n{traceback.format_exc()}")
                return None, None
            
            # Initialize strategy with ML predictions
            try:
                strategy = MLTradingStrategy(config)
                
                # Get the original feature data for backtesting
                backtest_data = data_loader.data[ticker].iloc[-len(test_data[ticker]['X']):]
                backtest_results = strategy.backtest(backtest_data, predictions)
                
                if backtest_results is None:
                    error_logger.error(f"Backtesting failed for {ticker}: No results returned")
                    return None, None
                
                metrics = backtest_results['metrics']
                trades = backtest_results['trades']
                equity_curve = backtest_results['equity_curve']
                
                if trades and metrics:
                    logger.info(f"Backtest completed for {ticker}. Metrics:")
                    logger.info(f"  - Total trades: {metrics['total_trades']}")
                    logger.info(f"  - Win rate: {metrics['win_rate']:.2%}")
                    logger.info(f"  - Total return: {metrics['total_return']:.2%}")
                    logger.info(f"  - Max drawdown: {metrics['max_drawdown']:.2%}")
                    logger.info(f"  - Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
                    logger.info(f"  - Profit factor: {metrics['profit_factor']:.2f}")
                    return metrics, trades
                else:
                    error_logger.error(f"No valid backtest results for {ticker}")
                    return None, None
            except Exception as e:
                error_logger.error(f"Failed to run backtest for {ticker}: {str(e)}\n{traceback.format_exc()}")
                return None, None
            
        except Exception as e:
            error_logger.error(f"Failed to train model for {ticker}: {str(e)}\n{traceback.format_exc()}")
            return None, None
        
    except Exception as e:
        error_logger.error(f"Error processing {ticker}: {str(e)}\n{traceback.format_exc()}")
        return None, None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--mode', choices=['train', 'backtest'], required=True, help='Mode of operation')
    parser.add_argument('--model_path', help='Path to pre-trained model for backtesting')
    args = parser.parse_args()
    
    logger.info("=== Trading Bot Simulation Started ===")
    
    try:
        # Load configuration
        config = Config.get_default_config()
        if not config.validate():
            logger.error("Invalid configuration settings")
            return
            
        logger.info(f"Tickers: {config.data['tickers']}")
        
        success_count = 0
        error_count = 0
        
        # Initialize data loader
        data_loader = DataLoader(
            tickers=config.data['tickers'],
            start_date=config.data['start_date'],
            end_date=config.data['end_date']
        )
        
        # Load all data first
        logger.info("Loading data for all tickers...")
        try:
            data = data_loader.load_data()
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            return
            
        # Prepare sequences for all tickers
        logger.info("Preparing sequences for all tickers...")
        try:
            prepared_data = data_loader.prepare_data(
                lookback_period=config.model['lookback_period'],
                horizon=config.model['prediction_horizon']
            )
        except Exception as e:
            logger.error(f"Failed to prepare data: {str(e)}")
            return
        
        for ticker in config.data['tickers']:
            try:
                logger.info(f"Processing {ticker}...")
                
                if ticker not in prepared_data:
                    logger.error(f"No prepared data available for {ticker}")
                    error_count += 1
                    continue
                
                # Process ticker with appropriate mode
                metrics, trades = process_ticker(ticker, data_loader, config, args.model_path if args.mode == 'backtest' else None)
                
                if metrics is not None and trades is not None:
                    success_count += 1
                else:
                    error_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                error_count += 1
                continue
        
        logger.info("=== Trading Bot Simulation Completed ===")
        logger.info(f"Successfully processed: {success_count}/{len(config.data['tickers'])} tickers")
        logger.info(f"Failed to process: {error_count}/{len(config.data['tickers'])} tickers")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))
        main()
    except Exception as e:
        error_logger.error(f"Fatal error: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)
