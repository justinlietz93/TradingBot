"""
Machine learning-based trading strategy.
"""
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class MLTradingStrategy:
    """Trading strategy using machine learning predictions."""
    def __init__(self, config):
        """Initialize strategy with enhanced risk management."""
        self.config = config
        self.max_position_size = 0.02  # Reduced from 0.03 for more conservative sizing
        self.risk_per_trade = 0.005  # Reduced from 0.01 for tighter risk control
        self.min_risk_reward = 2.0  # Increased from 1.5 for better reward/risk ratio
        self.atr_periods = 14  # Periods for ATR calculation
        self.volatility_threshold = 0.01  # Reduced from 0.015 for tighter volatility filter
        self.max_positions = 3  # Reduced from 4 for more focused positions
        self.initial_capital = config.trading['initial_capital']  # Get initial capital from config

    def calculate_atr(self, price):
        """Calculate Average True Range for position sizing."""
        return price * self.volatility_threshold

    def execute_trade(self, signal, current_price, current_time):
        """Execute a trade with enhanced risk management."""
        # Calculate ATR for position sizing
        atr = self.calculate_atr(current_price)
        
        # Enhanced stop loss and take profit calculations
        if signal == 1:  # Long position
            stop_loss = current_price - (1.5 * atr)  # Tighter stop loss
            take_profit = current_price + (3.0 * atr)  # Higher take profit for better R:R
        else:  # Short position
            stop_loss = current_price + (1.5 * atr)
            take_profit = current_price - (3.0 * atr)
        
        # Calculate position size based on risk
        risk_amount = self.initial_capital * self.risk_per_trade
        price_risk = abs(current_price - stop_loss)
        position_size = min(
            risk_amount / price_risk,
            self.initial_capital * self.max_position_size / current_price
        )
        
        # Scale position size based on conviction
        if abs(current_price - take_profit) / abs(current_price - stop_loss) >= self.min_risk_reward:
            position_size = position_size  # Full size for high R:R trades
        else:
            position_size = position_size * 0.7  # Reduced size for lower R:R trades
        
        return {
            'direction': signal,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'entry_time': current_time,
            'risk_reward_ratio': abs(current_price - take_profit) / abs(current_price - stop_loss)
        }

    def update_position(self, position, current_price, current_time):
        """Update position status."""
        if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
            pnl = (current_price - position['entry_price']) * position['position_size']
            return {
                'exit_price': current_price,
                'exit_time': current_time,
                'pnl': pnl
            }
        return None

    def generate_signals(self, features, ml_predictions):
        """Generate trading signals with enhanced filters and confirmation logic."""
        try:
            # Convert features to DataFrame if it's a dictionary
            if isinstance(features, dict):
                features = pd.DataFrame(features)
            
            # Ensure we have an index
            if features.index is None:
                features.index = range(len(features))
            
            # Debug logging
            logger.info(f"Available columns in features DataFrame: {features.columns.tolist()}")
            
            # Initialize signals DataFrame with zeros
            signals = pd.DataFrame(0, index=features.index, columns=['buy', 'sell'])
            
            # Split ML predictions into returns and directions
            horizon = len(ml_predictions[0]) // 2
            predicted_returns = ml_predictions[:, :horizon]
            predicted_directions = ml_predictions[:, horizon:]
            
            # Calculate weighted predictions across horizon
            weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])  # Higher weight for near-term predictions
            weights = weights / weights.sum()  # Normalize weights
            
            weighted_returns = np.average(predicted_returns, axis=1, weights=weights)
            weighted_directions = np.average(predicted_directions, axis=1, weights=weights)
            
            # Calculate prediction confidence
            return_std = np.std(predicted_returns, axis=1)
            direction_std = np.std(predicted_directions, axis=1)
            
            # Convert to pandas series for easier manipulation
            weighted_returns = pd.Series(weighted_returns, index=features.index)
            weighted_directions = pd.Series(weighted_directions, index=features.index)
            return_std = pd.Series(return_std, index=features.index)
            direction_std = pd.Series(direction_std, index=features.index)
            
            # Calculate multiple timeframe trends
            ema_fast = pd.Series(features['ema_5_scaled'], index=features.index)
            ema_slow = pd.Series(features['ema_10_scaled'], index=features.index)
            ema_medium = pd.Series(features['ema_20_scaled'], index=features.index)
            trend_primary = (ema_fast - ema_slow)
            trend_secondary = (ema_medium - ema_slow)  # Secondary trend
            
            # Calculate short-term momentum
            close = pd.Series(features['close_scaled'], index=features.index)
            momentum_5 = close.pct_change(5)
            momentum_10 = close.pct_change(10)
            momentum_20 = close.pct_change(20)
            
            # Enhanced momentum indicators
            rsi = pd.Series(features['rsi'], index=features.index)
            rsi_ma = rsi.rolling(window=10).mean()
            rsi_trend = rsi > rsi_ma
            
            macd = pd.Series(features['macd_scaled'], index=features.index)
            macd_signal = pd.Series(features['macd_signal_scaled'], index=features.index)
            macd_hist = pd.Series(features['macd_hist_scaled'], index=features.index)
            macd_hist_ma = macd_hist.rolling(window=5).mean()
            
            # Volume analysis
            volume = pd.Series(features['volume_scaled'], index=features.index)
            volume_ma = pd.Series(features['volume_ema_scaled'], index=features.index)
            volume_std = volume.rolling(window=20).std()
            volume_ratio = volume / volume_ma
            volume_surge = volume_ratio > 1.5
            volume_trend = volume > (volume_ma + 0.5 * volume_std)
            
            # Volatility filters
            volatility = pd.Series(features['volatility_14_scaled'], index=features.index)
            vol_ma = volatility.rolling(window=20).mean()
            vol_std = volatility.rolling(window=20).std()
            vol_ratio = volatility / vol_ma
            
            # Adaptive volatility thresholds
            vol_upper = vol_ma + (2 * vol_std)
            vol_lower = vol_ma * 0.8
            vol_filter = (volatility > vol_lower) & (volatility < vol_upper)
            
            # ML prediction trends
            returns_ma = weighted_returns.rolling(window=5).mean()
            returns_std = weighted_returns.rolling(window=10).std()
            direction_ma = weighted_directions.rolling(window=5).mean()
            
            # ML confidence scores
            return_confidence = 1 - (return_std / returns_ma.abs().clip(lower=1e-6))
            return_confidence = return_confidence.clip(lower=-1, upper=1)
            direction_confidence = 1 - direction_std
            direction_confidence = direction_confidence.clip(lower=0, upper=1)
            
            # Conviction scores with ML confidence
            bull_conviction = (
                (trend_primary > 0.005).astype(int) * 1.0 +
                (momentum_5 > 0).astype(int) * 0.8 +
                (momentum_10 > 0).astype(int) * 0.6 +
                (momentum_20 > 0).astype(int) * 0.4 +
                (rsi_trend).astype(int) * 0.7 +
                (macd > macd_signal).astype(int) * 0.8 +
                (macd_hist > 0).astype(int) * 0.6 +
                (volume_surge).astype(int) * 0.5 +
                (weighted_directions > 0.5).astype(int) * 1.2 +  # Higher weight for ML signals
                (return_confidence > 0.6).astype(int) * 0.8 +
                (direction_confidence > 0.7).astype(int) * 0.8
            )
            
            bear_conviction = (
                (trend_primary < -0.005).astype(int) * 1.0 +
                (momentum_5 < 0).astype(int) * 0.8 +
                (momentum_10 < 0).astype(int) * 0.6 +
                (momentum_20 < 0).astype(int) * 0.4 +
                (~rsi_trend).astype(int) * 0.7 +
                (macd < macd_signal).astype(int) * 0.8 +
                (macd_hist < 0).astype(int) * 0.6 +
                (volume_surge).astype(int) * 0.5 +
                (weighted_directions < -0.5).astype(int) * 1.2 +  # Higher weight for ML signals
                (return_confidence > 0.6).astype(int) * 0.8 +
                (direction_confidence > 0.7).astype(int) * 0.8
            )
            
            # Calculate adaptive thresholds based on market conditions with more lenient values
            bull_threshold = min(0.5, bull_conviction.mean() + 0.1 * bull_conviction.std())  # Reduced from 0.8
            bear_threshold = min(0.5, bear_conviction.mean() + 0.1 * bear_conviction.std())  # Reduced from 0.8
            trend_threshold = min(0.003, trend_primary.abs().mean() + 0.1 * trend_primary.abs().std())  # Reduced from 0.005
            return_threshold = min(0.02, returns_std.mean())  # Reduced from 0.03
            direction_threshold = min(0.1, weighted_directions.abs().mean() + 0.1 * weighted_directions.abs().std())  # Reduced from 0.2
            confidence_threshold = max(0.01, return_confidence.mean() * 0.5)  # Much more lenient confidence threshold
            
            # Calculate market condition indicators with more lenient filters
            volatility_filter = returns_std < returns_std.rolling(20).mean() * 3.0  # Increased from 2.5
            momentum_filter = (
                (macd_hist.abs() > macd_hist.abs().rolling(20).mean() * 0.2) &  # Reduced from 0.3
                (volume > volume.rolling(20).mean() * 0.2)  # Reduced from 0.3
            )
            trend_filter = (
                (trend_primary.abs() > trend_primary.abs().rolling(20).mean() * 0.2) &  # Reduced from 0.3
                (trend_secondary.abs() > trend_secondary.abs().rolling(20).mean() * 0.2)  # Reduced from 0.3
            )
            
            # Core conditions for buy signals with detailed logging
            conditions = {
                'bull_conviction': bull_conviction > bull_threshold,
                'trend_primary': trend_primary > trend_threshold,
                'rsi_range': (rsi > 20) & (rsi < 80),  # Widened from 30-70
                'macd_signal': macd > macd_signal,
                'volume_trend': volume_trend,
                'vol_filter': vol_filter,
                'returns_trend': weighted_returns > returns_ma,
                'direction_signal': weighted_directions > direction_threshold,
                'confidence': return_confidence > confidence_threshold
            }
            
            # Log each condition
            for name, condition in conditions.items():
                logger.info(f"Buy - {name}: {condition.sum()}/{len(condition)} ({condition.mean()*100:.1f}%)")
            
            # Require only a subset of conditions for buy signals
            buy_signals = (
                conditions['bull_conviction'] &  # Required
                conditions['rsi_range'] &  # Required
                (
                    (conditions['trend_primary'] & conditions['macd_signal']) |  # Technical signals
                    (conditions['returns_trend'] & conditions['direction_signal']) |  # ML signals
                    (conditions['volume_trend'] & conditions['vol_filter'])  # Volume signals
                )
            )
            
            # Core conditions for sell signals with detailed logging
            sell_conditions = {
                'bear_conviction': bear_conviction > bear_threshold,
                'trend_primary': trend_primary < -trend_threshold,
                'rsi_range': (rsi > 20) & (rsi < 80),  # Widened from 30-70
                'macd_signal': macd < macd_signal,
                'volume_trend': volume_trend,
                'vol_filter': vol_filter,
                'returns_trend': weighted_returns < returns_ma,
                'direction_signal': weighted_directions < -direction_threshold,
                'confidence': return_confidence > confidence_threshold
            }
            
            # Log each condition
            for name, condition in sell_conditions.items():
                logger.info(f"Sell - {name}: {condition.sum()}/{len(condition)} ({condition.mean()*100:.1f}%)")
            
            # Require only a subset of conditions for sell signals
            sell_signals = (
                sell_conditions['bear_conviction'] &  # Required
                sell_conditions['rsi_range'] &  # Required
                (
                    (sell_conditions['trend_primary'] & sell_conditions['macd_signal']) |  # Technical signals
                    (sell_conditions['returns_trend'] & sell_conditions['direction_signal']) |  # ML signals
                    (sell_conditions['volume_trend'] & sell_conditions['vol_filter'])  # Volume signals
                )
            )
            
            # Update signals DataFrame
            signals.loc[buy_signals, 'buy'] = 1
            signals.loc[sell_signals, 'sell'] = 1
            
            # Ensure no overlapping signals
            overlapping = (signals['buy'] == 1) & (signals['sell'] == 1)
            signals.loc[overlapping, ['buy', 'sell']] = 0
            
            # Log signal distribution
            total_signals = signals['buy'].sum() + signals['sell'].sum()
            logger.info(f"Generated {total_signals} total signals:")
            logger.info(f"  Buy signals: {signals['buy'].sum()}")
            logger.info(f"  Sell signals: {signals['sell'].sum()}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise

    def backtest(self, features, ml_predictions):
        """Run backtesting simulation."""
        try:
            # Generate signals
            signals = self.generate_signals(features, ml_predictions)
            
            # Initialize tracking variables
            current_capital = self.initial_capital
            positions = []
            trades = []
            equity_curve = [current_capital]
            
            # Get price data
            prices = features['close'].to_numpy()
            dates = features.index
            
            # Run simulation
            for i in range(len(features)):
                current_price = prices[i]
                current_time = dates[i]
                
                # Update existing positions
                for pos in positions[:]:  # Copy list to avoid modification during iteration
                    if current_price <= pos['stop_loss'] or current_price >= pos['take_profit']:
                        # Close position
                        exit_info = self.update_position(pos, current_price, current_time)
                        if exit_info:
                            current_capital += exit_info['pnl']
                            pos.update(exit_info)
                            trades.append(pos)
                            positions.remove(pos)
                
                # Check for new signals
                if signals.iloc[i]['buy'] and len(positions) < self.max_positions:
                    # Open long position
                    trade = self.execute_trade(1, current_price, current_time)
                    positions.append(trade)
                elif signals.iloc[i]['sell'] and len(positions) < self.max_positions:
                    # Open short position
                    trade = self.execute_trade(-1, current_price, current_time)
                    positions.append(trade)
                
                # Calculate current equity
                position_value = sum(
                    pos['position_size'] * current_price for pos in positions
                )
                total_equity = current_capital + position_value
                equity_curve.append(total_equity)
            
            # Close any remaining positions at the end
            final_price = prices[-1]
            final_time = dates[-1]
            for pos in positions[:]:
                exit_info = self.update_position(pos, final_price, final_time)
                if exit_info:
                    current_capital += exit_info['pnl']
                    pos.update(exit_info)
                    trades.append(pos)
            
            # Calculate performance metrics
            returns = pd.Series(equity_curve).pct_change().dropna()
            metrics = {
                'total_return': (current_capital - self.initial_capital) / self.initial_capital,
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(equity_curve),
                'win_rate': self._calculate_win_rate(trades),
                'profit_factor': self._calculate_profit_factor(trades),
                'total_trades': len(trades)
            }
            
            # Validate backtest results
            is_valid = (
                len(trades) >= 5 and  # Minimum number of trades
                metrics['total_trades'] > 0 and
                not np.isnan(metrics['sharpe_ratio']) and
                not np.isinf(metrics['sharpe_ratio']) and
                metrics['max_drawdown'] < 0.5  # Maximum allowed drawdown
            )
            
            logger.info("Backtesting completed successfully")
            logger.info(f"Total trades: {len(trades)}")
            logger.info(f"Final capital: ${current_capital:,.2f}")
            logger.info(f"Total return: {metrics['total_return']:.2%}")
            logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"Max drawdown: {metrics['max_drawdown']:.2%}")
            logger.info(f"Win rate: {metrics['win_rate']:.2%}")
            logger.info(f"Profit factor: {metrics['profit_factor']:.2f}")
            
            if not is_valid:
                logger.warning("Backtest results did not meet validation criteria")
                return None
            
            return {
                'equity_curve': equity_curve,
                'trades': trades,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error during backtesting: {str(e)}")
            raise

    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown from equity curve."""
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        return abs(drawdowns.min())

    def _calculate_win_rate(self, trades):
        """Calculate win rate from closed trades."""
        if not trades:
            return 0
        winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        return winning_trades / len(trades)

    def _calculate_profit_factor(self, trades):
        """Calculate profit factor from closed trades."""
        if not trades:
            return 0
        gross_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
        gross_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')