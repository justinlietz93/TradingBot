"""
Backtesting functionality for the trading bot.
"""
from datetime import datetime
import pandas as pd
import numpy as np

class Position:
    """Class representing a trading position."""
    def __init__(self, ticker, entry_price, size, stop_loss, take_profit, entry_time):
        self.ticker = ticker
        self.entry_price = entry_price
        self.size = size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = entry_time
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0
        self.is_closed = False

    def update(self, current_price, current_time=None):
        """Update position status based on current price."""
        if self.is_closed:
            return

        # Check stop loss and take profit
        if current_price <= self.stop_loss or current_price >= self.take_profit:
            self.close(current_price, current_time or datetime.now())

    def close(self, exit_price, exit_time):
        """Close the position."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.pnl = (self.exit_price - self.entry_price) * self.size
        self.is_closed = True

class Backtester:
    """Class for backtesting trading strategies."""
    def __init__(self, initial_capital, position_size, max_positions, stop_loss_pct, take_profit_pct):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.positions = []
        self.closed_positions = []
        self.equity_curve = []

    def calculate_position_size(self, price):
        """Calculate position size based on current capital and risk parameters."""
        return round(self.current_capital * self.position_size / price, 2)

    def open_position(self, ticker, entry_price, size, stop_loss, take_profit):
        """Open a new position."""
        if len(self.positions) >= self.max_positions:
            return None

        if entry_price <= 0 or size <= 0:
            raise ValueError("Invalid position parameters")

        position = Position(
            ticker=ticker,
            entry_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now()
        )
        self.positions.append(position)
        return position

    def update_positions(self, current_price, current_time):
        """Update all open positions."""
        for position in self.positions[:]:  # Copy list to avoid modification during iteration
            position.update(current_price, current_time)
            if position.is_closed:
                self.current_capital += position.pnl
                self.closed_positions.append(position)
                self.positions.remove(position)

    def run(self, ticker, price_data, signals):
        """Run backtesting simulation."""
        if price_data.empty or signals.empty:
            raise ValueError("Empty price data or signals")

        if len(price_data) != len(signals):
            raise ValueError("Price data and signals must have the same length")

        self.equity_curve = [self.initial_capital]
        trades = []

        for i in range(len(price_data)):
            current_price = price_data.iloc[i]['close']
            current_time = price_data.index[i]

            # Update existing positions
            self.update_positions(current_price, current_time)

            # Check for new signals
            if signals.iloc[i]['buy_signal'] and len(self.positions) < self.max_positions:
                size = self.calculate_position_size(current_price)
                stop_loss = current_price * (1 - self.stop_loss_pct)
                take_profit = current_price * (1 + self.take_profit_pct)
                
                position = self.open_position(
                    ticker=ticker,
                    entry_price=current_price,
                    size=size,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                if position:
                    trades.append({
                        'entry_price': position.entry_price,
                        'entry_time': position.entry_time,
                        'size': position.size,
                        'stop_loss': position.stop_loss,
                        'take_profit': position.take_profit
                    })

            # Record equity
            total_equity = self.current_capital + sum(
                pos.size * current_price for pos in self.positions
            )
            self.equity_curve.append(total_equity)

        # Close any remaining positions at the end
        final_price = price_data.iloc[-1]['close']
        final_time = price_data.index[-1]
        for position in self.positions[:]:
            position.close(final_price, final_time)
            self.current_capital += position.pnl
            self.closed_positions.append(position)
            trades.append({
                'entry_price': position.entry_price,
                'entry_time': position.entry_time,
                'exit_price': position.exit_price,
                'exit_time': position.exit_time,
                'pnl': position.pnl
            })

        # Calculate metrics
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        metrics = {
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor()
        }

        return {
            'equity_curve': self.equity_curve,
            'trades': trades,
            'metrics': metrics
        }

    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown from equity curve."""
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        return abs(drawdowns.min())

    def _calculate_win_rate(self):
        """Calculate win rate from closed positions."""
        if not self.closed_positions:
            return 0
        winning_trades = sum(1 for pos in self.closed_positions if pos.pnl > 0)
        return winning_trades / len(self.closed_positions)

    def _calculate_profit_factor(self):
        """Calculate profit factor from closed positions."""
        if not self.closed_positions:
            return 0
        gross_profit = sum(pos.pnl for pos in self.closed_positions if pos.pnl > 0)
        gross_loss = abs(sum(pos.pnl for pos in self.closed_positions if pos.pnl < 0))
        return gross_profit / gross_loss if gross_loss != 0 else float('inf') 