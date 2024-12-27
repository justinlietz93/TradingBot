from typing import Optional
import pandas as pd
import numpy as np
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class TechnicalStrategy(BaseStrategy):
    def __init__(self, config: 'Config'):
        super().__init__(config)
        self.rsi_oversold = 35
        self.rsi_overbought = 65
        self.sma_short = 10
        self.sma_long = 30
        self.bb_std = 2
        self.min_trend_strength = 0.02
        self.momentum_window = 10
        
    def generate_signals(self, data: pd.DataFrame, predictions: Optional[np.ndarray] = None) -> pd.Series:
        """Generate trading signals based on technical indicators."""
        signals = pd.Series(index=data.index, data=0)
        
        try:
            # Calculate additional technical indicators
            sma_short = data['close'].rolling(window=self.sma_short).mean()
            sma_long = data['close'].rolling(window=self.sma_long).mean()
            momentum = data['returns'].rolling(window=self.momentum_window).mean()
            
            # Calculate volatility for dynamic thresholds
            volatility = data['returns'].rolling(window=20).std()
            
            # Combine multiple technical signals
            for i in range(len(data)):
                # Skip if we don't have enough data for indicators
                if i < self.sma_long:
                    continue
                    
                # Get current indicator values
                rsi = data['rsi'].iloc[i]
                macd = data['macd'].iloc[i]
                bb_upper = data['bb_upper'].iloc[i]
                bb_lower = data['bb_lower'].iloc[i]
                current_price = data['close'].iloc[i]
                current_momentum = momentum.iloc[i]
                current_volatility = volatility.iloc[i]
                
                # Calculate trend direction and strength
                trend_up = sma_short.iloc[i] > sma_long.iloc[i]
                trend_strength = abs(sma_short.iloc[i] - sma_long.iloc[i]) / sma_long.iloc[i]
                
                # Dynamic RSI thresholds based on volatility
                rsi_oversold = self.rsi_oversold * (1 + current_volatility)
                rsi_overbought = self.rsi_overbought * (1 - current_volatility)
                
                # Adaptive trend strength threshold
                trend_strength_threshold = self.min_trend_strength * (1 - current_volatility) * 0.5
                
                # Generate buy signals with relaxed conditions
                if ((rsi < rsi_oversold or current_price < bb_lower) and  # Oversold condition
                    (macd > 0 or trend_up) and  # Either MACD or trend is positive
                    trend_strength > trend_strength_threshold):  # Reduced trend strength requirement
                    signals.iloc[i] = 1
                    
                # Generate sell signals with relaxed conditions
                elif ((rsi > rsi_overbought or current_price > bb_upper) and  # Overbought condition
                      (macd < 0 or not trend_up) and  # Either MACD or trend is negative  
                      trend_strength > trend_strength_threshold):  # Reduced trend strength requirement
                    signals.iloc[i] = -1
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating technical signals: {str(e)}")
            return signals
            
    def calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> tuple:
        """Calculate support and resistance levels using moving averages."""
        try:
            highs = data['high'].rolling(window=window).max()
            lows = data['low'].rolling(window=window).min()
            
            resistance = highs.rolling(window=window).mean()
            support = lows.rolling(window=window).mean()
            
            return support, resistance
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            return None, None
            
    def detect_divergence(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Detect RSI divergence patterns."""
        try:
            divergence = pd.Series(index=data.index, data=0)
            
            for i in range(window, len(data)):
                price_window = data['close'].iloc[i-window:i]
                rsi_window = data['rsi'].iloc[i-window:i]
                
                # Bullish divergence: price making lower lows but RSI making higher lows
                if (price_window.iloc[-1] < price_window.min() and 
                    rsi_window.iloc[-1] > rsi_window.min()):
                    divergence.iloc[i] = 1
                    
                # Bearish divergence: price making higher highs but RSI making lower highs
                elif (price_window.iloc[-1] > price_window.max() and 
                      rsi_window.iloc[-1] < rsi_window.max()):
                    divergence.iloc[i] = -1
                    
            return divergence
            
        except Exception as e:
            logger.error(f"Error detecting divergence: {str(e)}")
            return pd.Series(index=data.index, data=0)
            
    def calculate_volatility_bands(self, data: pd.DataFrame, window: int = 20, num_std: float = 2) -> tuple:
        """Calculate volatility-based trading bands."""
        try:
            # Calculate ATR-based bands
            tr1 = data['high'] - data['low']
            tr2 = abs(data['high'] - data['close'].shift())
            tr3 = abs(data['low'] - data['close'].shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=window).mean()
            
            middle = data['close'].rolling(window=window).mean()
            upper = middle + (atr * num_std)
            lower = middle - (atr * num_std)
            
            return lower, middle, upper
            
        except Exception as e:
            logger.error(f"Error calculating volatility bands: {str(e)}")
            return None, None, None