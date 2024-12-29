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
            signals = pd.DataFrame(index=features.index)
            
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
            ema_fast = features['ema_5']
            ema_slow = features['ema_10']
            trend_primary = (ema_fast - ema_slow) / ema_slow
            
            # Calculate short-term momentum
            close = features['close']
            momentum_5 = close.pct_change(5)
            momentum_10 = close.pct_change(10)
            momentum_20 = close.pct_change(20)
            
            # Enhanced momentum indicators
            rsi = features['rsi']
            rsi_ma = rsi.rolling(window=10).mean()
            rsi_trend = rsi > rsi_ma
            
            macd = features['macd']
            macd_signal = features['macd_signal']
            macd_hist = features['macd_hist']
            macd_hist_ma = macd_hist.rolling(window=5).mean()
            
            # Volume analysis
            volume = features['volume']
            volume_ma = features['volume_ema']
            volume_std = volume.rolling(window=20).std()
            volume_ratio = features['volume_ratio']
            volume_surge = volume_ratio > 1.5
            volume_trend = volume > (volume_ma + 0.5 * volume_std)
            
            # Volatility filters
            volatility = features['volatility_14']
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
            return_confidence = 1 - (return_std / returns_ma.abs())
            direction_confidence = 1 - direction_std
            
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
            
            # Generate signals with enhanced conditions and conviction scores
            signals['buy'] = (
                (bull_conviction > 6.0) &  # Adjusted threshold for weighted scores
                (trend_primary > 0.005) &
                (trend_primary.diff() > 0) &
                (rsi > 30) & (rsi < 70) &
                (macd > macd_signal) &
                (macd_hist > 0) &
                (macd_hist > macd_hist_ma) &
                volume_trend &
                vol_filter &
                (weighted_returns > returns_ma + 0.5 * returns_std) &
                (weighted_directions > 0.7) &
                (direction_ma > direction_ma.shift(1)) &
                (return_confidence > 0.6) &  # Add confidence thresholds
                (direction_confidence > 0.7)
            ).astype(int)
            
            signals['sell'] = (
                (bear_conviction > 6.0) &  # Adjusted threshold for weighted scores
                (trend_primary < -0.005) &
                (trend_primary.diff() < 0) &
                (rsi > 30) & (rsi < 70) &
                (macd < macd_signal) &
                (macd_hist < 0) &
                (macd_hist < macd_hist_ma) &
                volume_trend &
                vol_filter &
                (weighted_returns < returns_ma - 0.5 * returns_std) &
                (weighted_directions < -0.7) &
                (direction_ma < direction_ma.shift(1)) &
                (return_confidence > 0.6) &  # Add confidence thresholds
                (direction_confidence > 0.7)
            ).astype(int)
            
            # Log signal statistics
            buy_signals = signals['buy'].sum()
            sell_signals = signals['sell'].sum()
            logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
            logger.info(f"Average bull conviction: {bull_conviction.mean():.2f}")
            logger.info(f"Average bear conviction: {bear_conviction.mean():.2f}")
            logger.info(f"Average return confidence: {return_confidence.mean():.2f}")
            logger.info(f"Average direction confidence: {direction_confidence.mean():.2f}")
            
            # Log signal distribution
            signal_dates = signals[signals['buy'] | signals['sell']].index
            if len(signal_dates) > 0:
                logger.info("Signal distribution:")
                logger.info(f"  First signal: {signal_dates[0]}")
                logger.info(f"  Last signal: {signal_dates[-1]}")
                logger.info(f"  Average signals per month: {len(signal_dates) / (len(features) / 21):.2f}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise

    def backtest(self, features, ml_predictions):
        """Run backtesting simulation."""
        signals = self.generate_signals(features, ml_predictions)
        trades = []
        positions = []
        equity = [self.initial_capital]
        current_capital = self.initial_capital
        
        for i in range(len(features)):
            current_price = features['close'].iloc[i]
            current_time = features.index[i]
            
            # Update existing positions
            for pos in positions[:]:
                update = self.update_position(pos, current_price, current_time)
                if update:
                    pos.update(update)
                    current_capital += update['pnl']
                    trades.append(pos)
                    positions.remove(pos)
            
            # Check for new signals
            if signals['buy'].iloc[i] and len(positions) < self.max_positions:
                trade = self.execute_trade(1, current_price, current_time)
                positions.append(trade)
            elif signals['sell'].iloc[i] and len(positions) < self.max_positions:
                trade = self.execute_trade(-1, current_price, current_time)
                positions.append(trade)
            
            # Record equity
            total_equity = current_capital + sum(
                pos['position_size'] * current_price for pos in positions
            )
            equity.append(total_equity)
        
        # Calculate metrics
        returns = pd.Series(equity).pct_change().dropna()
        metrics = {
            'total_trades': len(trades),
            'win_rate': len([t for t in trades if t.get('pnl', 0) > 0]) / len(trades) if trades else 0,
            'avg_return': returns.mean() if len(returns) > 0 else 0,
            'total_return': (equity[-1] - equity[0]) / equity[0],
            'max_drawdown': self._calculate_max_drawdown(equity),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
            'direction_accuracy': len([t for t in trades if (t.get('pnl', 0) > 0) == (t['direction'] == 1)]) / len(trades) if trades else 0
        }
        
        return trades, metrics

    def _calculate_max_drawdown(self, equity):
        """Calculate maximum drawdown from equity curve."""
        peak = equity[0]
        max_dd = 0
        
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd