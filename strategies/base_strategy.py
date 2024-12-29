from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    def __init__(self, config: 'Config'):
        self.config = config
        self.positions = []
        self.cash = config.trading.initial_capital
        self.max_position_size = 0.05  # Maximum 5% of portfolio in single position
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        self.max_positions = 5  # Maximum number of concurrent positions
        self.risk_per_trade = 0.01  # Risk 1% of portfolio per trade
        self.peak_portfolio_value = self.cash
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, predictions: Optional[np.ndarray] = None) -> pd.Series:
        """Generate trading signals based on strategy logic."""
        pass
    
    def calculate_position_size(self, price: float, volatility: float) -> int:
        """Calculate position size based on risk management rules."""
        try:
            # Calculate risk amount (% of current portfolio)
            risk_amount = self.cash * self.risk_per_trade
            
            # Handle zero or very low volatility
            min_volatility = 0.001  # Minimum volatility threshold
            position_volatility = max(volatility, min_volatility)
            
            # Calculate position size based on volatility and risk
            position_size = risk_amount / (price * position_volatility)
            
            # Apply position size limits
            max_size = int(self.cash * self.max_position_size / price)
            position_size = min(position_size, max_size)
            
            # Ensure minimum position size
            min_size = 1
            position_size = max(min_size, int(position_size))
            
            # Adjust position size based on current drawdown
            drawdown = self.calculate_drawdown()
            if drawdown > 0.1:
                position_size = int(position_size * (1 - drawdown))
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0
            
    def calculate_drawdown(self) -> float:
        """Calculate the current drawdown of the portfolio."""
        if not self.positions:
            return 0
        
        current_value = self.cash + sum(p['size'] * p['entry_price'] for p in self.positions)
        peak_value = max(current_value, self.peak_portfolio_value)
        self.peak_portfolio_value = peak_value
        
        drawdown = (peak_value - current_value) / peak_value
        return drawdown
        
    def check_stop_loss_take_profit(self, position: Dict, current_price: float) -> bool:
        """Check if position should be closed due to stop-loss or take-profit."""
        try:
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            # Calculate current return
            current_return = (current_price - entry_price) / entry_price
            
            # Check stop loss
            if current_price <= stop_loss:
                logger.info(f"Stop loss triggered at ${current_price:.2f} (Entry: ${entry_price:.2f})")
                return True
                
            # Check take profit
            if current_price >= take_profit:
                logger.info(f"Take profit triggered at ${current_price:.2f} (Entry: ${entry_price:.2f})")
                return True
                
            # Trailing stop loss
            if current_return > 0.02:  # If position is profitable
                new_stop = entry_price + (current_price - entry_price) * 0.5  # Move stop to 50% of gains
                position['stop_loss'] = max(position['stop_loss'], new_stop)
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking stop loss/take profit: {str(e)}")
            return False
            
    def execute_trades(self, signals: pd.Series, data: pd.DataFrame) -> List[Dict]:
        """Execute trades based on signals and return trade history."""
        trades = []
        current_position = None
        
        for i in range(len(signals)):
            price = data['close'].iloc[i]
            date = data.index[i]
            signal = signals.iloc[i]
            
            # Close position if we have a sell signal or hit stop loss/take profit
            if current_position is not None:
                entry_price = current_position['entry_price']
                position_type = current_position['type']
                
                # Calculate returns
                if position_type == 'long':
                    returns = (price - entry_price) / entry_price
                else:  # short
                    returns = (entry_price - price) / entry_price
                
                # Check if we should close the position
                should_close = (
                    (signal == -1 and position_type == 'long') or
                    (signal == 1 and position_type == 'short') or
                    (returns <= -self.stop_loss_pct) or
                    (returns >= self.take_profit_pct)
                )
                
                if should_close:
                    current_position['exit_price'] = price
                    current_position['exit_date'] = date
                    current_position['returns'] = returns
                    trades.append(current_position)
                    current_position = None
            
            # Open new position if we have a signal and no current position
            if current_position is None and signal != 0:
                position_type = 'long' if signal == 1 else 'short'
                volatility = data['volatility'].iloc[i] if 'volatility' in data.columns else 0.01
                position_size = self.calculate_position_size(price, volatility)
                
                current_position = {
                    'type': position_type,
                    'entry_price': price,
                    'entry_date': date,
                    'size': position_size
                }
        
        # Close any remaining position at the end
        if current_position is not None:
            price = data['close'].iloc[-1]
            entry_price = current_position['entry_price']
            position_type = current_position['type']
            
            if position_type == 'long':
                returns = (price - entry_price) / entry_price
            else:  # short
                returns = (entry_price - price) / entry_price
            
            current_position['exit_price'] = price
            current_position['exit_date'] = data.index[-1]
            current_position['returns'] = returns
            trades.append(current_position)
        
        return trades

    def calculate_metrics(self, trades: List[Dict], final_price: float) -> Dict:
        """Calculate trading performance metrics."""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'total_return': 0.0
            }
        
        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade['returns'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate returns
        returns = [trade['returns'] for trade in trades]
        avg_return = np.mean(returns)
        total_return = np.prod([1 + r for r in returns]) - 1
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        if len(returns) > 1:
            returns_std = np.std(returns)
            sharpe_ratio = np.sqrt(252) * (avg_return / returns_std) if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        cumulative_returns = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / running_max
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return
        }