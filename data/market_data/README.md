# Market Data

This directory contains historical market data used for training, testing, and backtesting the trading bot.

## Structure

- `raw/`: Raw data downloaded from data sources
- `processed/`: Cleaned and preprocessed data
- `features/`: Engineered features and technical indicators

## Data Sources

- Primary: Yahoo Finance (via yfinance)
- Alternative: Alpha Vantage

## Data Format

Data is stored in CSV format with the following columns:
- Date
- Open
- High
- Low
- Close
- Volume
- Adjusted Close

## Usage

Data in this directory is automatically managed by the data pipeline. Manual modifications are not recommended.
