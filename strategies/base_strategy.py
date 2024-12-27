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
        """Execute trades based on signals and update portfolio."""
        trades = []
        
        try:
            for i, signal in enumerate(signals):
                price = data['close'].iloc[i]
                volatility = data['volatility'].iloc[i]
                
                # Check stop loss/take profit for existing positions
                for position in self.positions[:]:
                    if self.check_stop_loss_take_profit(position, price):
                        proceeds = position['size'] * price
                        self.cash += proceeds
                        self.positions.remove(position)
                        trades.append({
                            'date': data.index[i],
                            'action': 'CLOSE',
                            'size': position['size'],
                            'price': price,
                            'proceeds': proceeds,
                            'reason': 'SL/TP'
                        })
                        logger.info(f"Position closed at ${price:.2f}, Portfolio: ${self.cash:.2f}")
                
                if signal == 1:  # Buy signal
                    # Check if we have enough cash and are under position limit
                    if len(self.positions) < self.max_positions:
                        size = self.calculate_position_size(price, volatility)
                        cost = size * price
                        
                        if cost <= self.cash and size > 0:
                            self.cash -= cost
                            self.positions.append({
                                'size': size,
                                'entry_price': price,
                                'entry_date': data.index[i],
                                'stop_loss': price * (1 - self.stop_loss_pct),
                                'take_profit': price * (1 + self.take_profit_pct)
                            })
                            trades.append({
                                'date': data.index[i],
                                'action': 'BUY',
                                'size': size,
                                'price': price,
                                'cost': cost
                            })
                            logger.info(f"Buy signal: {size} shares at ${price:.2f}, Portfolio: ${self.cash:.2f}")
                            
                            # Adjust stop loss and take profit based on volatility
                            last_position = self.positions[-1]
                            last_position['stop_loss'] = price * (1 - self.stop_loss_pct * (1 + volatility))
                            last_position['take_profit'] = price * (1 + self.take_profit_pct * (1 - volatility))
                            
                elif signal == -1:  # Sell signal
                    for position in self.positions[:]:
                        proceeds = position['size'] * price
                        self.cash += proceeds
                        self.positions.remove(position)
                        trades.append({
                            'date': data.index[i],
                            'action': 'SELL',
                            'size': position['size'],
                            'price': price,
                            'proceeds': proceeds
                        })
                        logger.info(f"Sell signal: {position['size']} shares at ${price:.2f}, Portfolio: ${self.cash:.2f}")
                        
            return trades
            
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
            return trades
    
    def calculate_metrics(self, trades: List[Dict], final_price: float) -> Dict:
        """Calculate trading performance metrics."""
        try:
            metrics = {}
            
            # Basic metrics
            metrics['total_trades'] = len(trades)
            metrics['final_cash'] = self.cash
            metrics['positions'] = len(self.positions)
            
            # Calculate final portfolio value
            portfolio_value = self.cash + sum(pos['size'] * final_price for pos in self.positions)
            metrics['portfolio_value'] = portfolio_value
            
            # Calculate returns
            initial_capital = self.config.trading.initial_capital
            total_return = (portfolio_value - initial_capital) / initial_capital
            metrics['total_return'] = total_return
            
            if trades:
                # Win rate
                profitable_trades = sum(1 for t in trades if t.get('proceeds', 0) > t.get('cost', 0))
                metrics['win_rate'] = profitable_trades / len(trades)
                
                # Calculate profit/loss for each trade
                trade_returns = []
                for trade in trades:
                    if trade['action'] == 'SELL' or trade['action'] == 'CLOSE':
                        trade_return = (trade['proceeds'] - trade.get('cost', 0)) / trade.get('cost', 1)
                        trade_returns.append(trade_return)
                
                if trade_returns:
                    # Average return per trade
                    metrics['avg_trade_return'] = np.mean(trade_returns)
                    
                    # Sharpe ratio (assuming risk-free rate of 0.02)
                    excess_returns = np.array(trade_returns) - 0.02
                    if len(excess_returns) > 1:
                        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
                        metrics['sharpe_ratio'] = sharpe_ratio
                    
                    # Maximum drawdown
                    cumulative_returns = np.cumprod(1 + np.array(trade_returns))
                    running_max = np.maximum.accumulate(cumulative_returns)
                    drawdowns = (running_max - cumulative_returns) / running_max
                    metrics['max_drawdown'] = np.max(drawdowns)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}