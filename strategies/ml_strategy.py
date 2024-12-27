from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import logging
from .base_strategy import BaseStrategy
from .technical_strategy import TechnicalStrategy

logger = logging.getLogger(__name__)

class MLTradingStrategy(BaseStrategy):
    def __init__(self, config: 'Config', ml_model: 'MLModel'):
        super().__init__(config)
        self.ml_model = ml_model
        self.prediction_threshold = 0.6  # Increased confidence threshold
        self.technical_strategy = TechnicalStrategy(config)
        self.volatility_threshold = 0.02  # Reduced for more conservative entry
        self.momentum_window = 10  # Increased for more stable momentum
        self.profit_target_multiplier = 1.5  # More realistic profit target
        self.max_position_size = 0.05  # Maximum 5% of portfolio per position
        self.stop_loss_atr_multiplier = 2.0  # Use ATR for dynamic stop-loss
        
    def generate_signals(self, data: pd.DataFrame, predictions: Optional[np.ndarray] = None) -> pd.Series:
        """Generate trading signals using enhanced ML strategy."""
        if predictions is None:
            raise ValueError("Predictions are required for ML strategy")
            
        signals = pd.Series(index=data.index, data=0)
        
        try:
            # Get technical signals for confirmation
            tech_signals = self.technical_strategy.generate_signals(data)
            
            # Calculate volatility and momentum
            returns = data['returns']
            volatility = returns.rolling(window=20).std()
            momentum = returns.rolling(window=self.momentum_window).mean()
            
            # Calculate ATR for dynamic stop-loss
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            # Dynamic thresholds based on market conditions
            vol_adjusted_threshold = self.prediction_threshold * (1 + volatility)
            
            for i in range(len(predictions)):
                if i < 20:  # Skip initial periods for indicators to warm up
                    continue
                    
                pred_returns = predictions[i]
                current_price = data['close'].iloc[i]
                rsi = data['rsi'].iloc[i]
                current_volatility = volatility.iloc[i]
                current_momentum = momentum.iloc[i]
                current_atr = atr.iloc[i]
                
                if current_volatility > self.volatility_threshold * 2:  # Skip high volatility periods
                    continue
                
                # Enhanced ML signal generation with confidence measure
                pred_mean = np.mean(pred_returns)
                pred_std = np.std(pred_returns)
                confidence = abs(pred_mean) / (pred_std + 1e-6)
                
                # Calculate trend strength with multiple timeframes
                sma_short = data['close'].rolling(window=10).mean().iloc[i]
                sma_medium = data['close'].rolling(window=20).mean().iloc[i]
                sma_long = data['close'].rolling(window=30).mean().iloc[i]
                
                # Multi-timeframe trend strength
                trend_short = (sma_short - sma_medium) / sma_medium
                trend_long = (sma_medium - sma_long) / sma_long
                trend_strength = (trend_short + trend_long) / 2
                
                # Generate buy signals with confirmation
                if (pred_mean > vol_adjusted_threshold.iloc[i] and 
                    confidence > 1.5 and 
                    trend_strength > 0 and 
                    current_momentum > 0 and
                    tech_signals.iloc[i] >= 0):  # Technical confirmation or neutral
                    
                    # Dynamic position sizing based on confidence and volatility
                    position_size = min(
                        self.max_position_size * confidence / (current_volatility * 10),
                        self.max_position_size
                    )
                    
                    # Dynamic stop-loss and take-profit levels
                    stop_loss = current_price - (current_atr * self.stop_loss_atr_multiplier)
                    take_profit = current_price + (current_atr * self.stop_loss_atr_multiplier * self.profit_target_multiplier)
                    
                    signals.iloc[i] = 1
                    
                # Generate sell signals with confirmation
                elif (pred_mean < -vol_adjusted_threshold.iloc[i] and 
                      confidence > 1.5 and 
                      trend_strength < 0 and 
                      current_momentum < 0 and
                      tech_signals.iloc[i] <= 0):  # Technical confirmation or neutral
                    
                    signals.iloc[i] = -1
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating ML signals: {str(e)}")
            return signals
            
    def update_model(self, new_data: pd.DataFrame):
        """Update the ML model with new data."""
        try:
            # Prepare new data
            X_new = self.prepare_features(new_data)
            y_new = self.prepare_targets(new_data)
            
            # Update model (assuming online learning capability)
            if hasattr(self.ml_model, 'partial_fit'):
                self.ml_model.partial_fit(X_new, y_new)
                logger.info("Model updated with new data")
            else:
                logger.warning("Model does not support online learning")
                
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            
    def analyze_prediction_quality(self, predictions: np.ndarray, actual_returns: np.ndarray) -> Dict:
        """Analyze the quality of ML predictions with enhanced metrics."""
        try:
            metrics = {}
            
            # Calculate prediction accuracy
            pred_direction = np.sign(predictions)
            actual_direction = np.sign(actual_returns)
            direction_accuracy = np.mean(pred_direction == actual_direction)
            
            metrics['direction_accuracy'] = direction_accuracy
            
            # Calculate prediction error metrics
            mse = np.mean((predictions - actual_returns) ** 2)
            mae = np.mean(np.abs(predictions - actual_returns))
            
            metrics['mse'] = mse
            metrics['mae'] = mae
            
            # Calculate correlation
            correlation = np.corrcoef(predictions.flatten(), actual_returns.flatten())[0, 1]
            metrics['correlation'] = correlation
            
            # Calculate additional metrics
            metrics['rmse'] = np.sqrt(mse)
            metrics['mean_prediction'] = np.mean(predictions)
            metrics['prediction_std'] = np.std(predictions)
            metrics['actual_std'] = np.std(actual_returns)
            
            # Calculate hit ratio (percentage of profitable trades)
            profitable_predictions = np.sum((predictions > 0) & (actual_returns > 0)) + \
                                   np.sum((predictions < 0) & (actual_returns < 0))
            total_predictions = len(predictions)
            metrics['hit_ratio'] = profitable_predictions / total_predictions
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing predictions: {str(e)}")
            return {}