"""
Trading strategy based on ML predictions.

This module implements a trading strategy that:
1. Uses ML model probability predictions
2. Trades when confidence exceeds threshold
3. Includes realistic latency
4. Accounts for transaction costs
5. Implements position sizing based on confidence

Strategy Logic:
--------------
For each tick t:
1. Get prediction probabilities: P(UP), P(NEUTRAL), P(DOWN)
2. If max(P) > threshold:
   - If P(UP) is max → BUY signal
   - If P(DOWN) is max → SELL signal
3. Execute trade after latency delay
4. Pay transaction costs (spread + commission)
5. Close position after holding period or stop loss

Position Sizing:
---------------
If use_confidence_sizing = True:
    position_size = base_size × (confidence - threshold) / (1 - threshold)
    
This scales position size linearly with confidence above threshold.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class TradingStrategy:
    """
    ML-based trading strategy with realistic execution.
    
    Attributes:
        config: StrategyConfig object
        positions: Current open positions
        trades: List of executed trades
    """
    
    def __init__(self, config):
        """
        Initialize trading strategy.
        
        Args:
            config: StrategyConfig object
        """
        self.config = config
        self.positions = []
        self.trades = []
        
    def generate_signals(self, predictions: np.ndarray, 
                        probabilities: np.ndarray) -> pd.DataFrame:
        """
        Generate trading signals from model predictions.
        
        Args:
            predictions: Predicted classes (0=DOWN, 1=NEUTRAL, 2=UP)
            probabilities: Probability matrix (n_samples, 3)
            
        Returns:
            DataFrame with signals:
                - signal: 1 (BUY), 0 (HOLD), -1 (SELL)
                - confidence: Maximum probability
                - position_size: Shares to trade
        """
        n = len(predictions)
        signals = np.zeros(n)
        confidences = np.max(probabilities, axis=1)
        position_sizes = np.zeros(n)
        
        for i in range(n):
            max_prob = confidences[i]
            
            # Only trade if confidence exceeds threshold
            if max_prob > self.config.entry_threshold:
                predicted_class = predictions[i]
                
                if predicted_class == 2:  # UP
                    signals[i] = 1  # BUY
                elif predicted_class == 0:  # DOWN
                    signals[i] = -1  # SELL
                else:  # NEUTRAL
                    signals[i] = 0  # HOLD
                
                # Calculate position size
                if signals[i] != 0:
                    if self.config.use_confidence_sizing:
                        # Scale by confidence
                        scaling_factor = (max_prob - self.config.entry_threshold) / \
                                       (1 - self.config.entry_threshold)
                        position_sizes[i] = min(
                            self.config.base_position_size * (1 + scaling_factor),
                            self.config.max_position_size
                        )
                    else:
                        position_sizes[i] = self.config.base_position_size
        
        df = pd.DataFrame({
            'signal': signals,
            'confidence': confidences,
            'position_size': position_sizes
        })
        
        return df
    
    def apply_latency(self, signals: pd.DataFrame, tick_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply execution latency to signals.
        
        Signal at time t is executed at time t + latency.
        
        Args:
            signals: DataFrame with signals
            tick_data: Original tick data with prices
            
        Returns:
            DataFrame with execution prices and adjusted signals
        """
        latency = self.config.execution_latency
        
        # Shift signals forward by latency
        signals['execution_signal'] = signals['signal'].shift(latency).fillna(0)
        signals['execution_position_size'] = signals['position_size'].shift(latency).fillna(0)
        
        # Get execution prices (after latency)
        signals['execution_price'] = tick_data['mid_price'].values
        signals['bid_price'] = tick_data['bid_price'].values
        signals['ask_price'] = tick_data['ask_price'].values
        signals['spread'] = tick_data['spread'].values
        
        return signals
    
    def calculate_transaction_costs(self, signal: float, position_size: float,
                                   bid: float, ask: float, spread: float) -> float:
        """
        Calculate transaction costs for a trade.
        
        Costs include:
        1. Commission: fixed per share
        2. Spread cost: pay ask when buying, receive bid when selling
        
        Args:
            signal: 1 (BUY) or -1 (SELL)
            position_size: Number of shares
            bid: Bid price
            ask: Ask price
            spread: Bid-ask spread
            
        Returns:
            Total transaction cost in dollars
        """
        # Commission
        commission = self.config.commission_per_share * position_size
        
        # Spread cost
        if self.config.use_spread_cost:
            # When buying, pay ask (half-spread above mid)
            # When selling, receive bid (half-spread below mid)
            # Cost is half-spread per share
            spread_cost = (spread / 2) * position_size
        else:
            spread_cost = 0
        
        total_cost = commission + spread_cost
        
        return total_cost
    
    def execute_trades(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Execute trades based on signals and track PnL.
        
        Args:
            signals: DataFrame with execution signals and prices
            
        Returns:
            DataFrame with trade execution details and PnL
        """
        trades = []
        position = None  # Current open position
        
        for i, row in signals.iterrows():
            signal = row['execution_signal']
            
            # Skip if no signal
            if signal == 0:
                continue
            
            # Close existing position if opposite signal
            if position is not None:
                # Check holding period
                holding_period = i - position['entry_idx']
                
                # Close if max holding period reached or opposite signal
                if holding_period >= self.config.max_holding_period or \
                   (signal * position['direction'] < 0):
                    
                    # Calculate PnL
                    if position['direction'] == 1:  # Long position
                        exit_price = row['bid_price']  # Sell at bid
                        pnl = (exit_price - position['entry_price']) * position['size']
                    else:  # Short position
                        exit_price = row['ask_price']  # Cover at ask
                        pnl = (position['entry_price'] - exit_price) * position['size']
                    
                    # Subtract exit costs
                    exit_cost = self.calculate_transaction_costs(
                        -position['direction'],
                        position['size'],
                        row['bid_price'],
                        row['ask_price'],
                        row['spread']
                    )
                    pnl -= exit_cost
                    
                    # Record trade
                    trades.append({
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'direction': position['direction'],
                        'size': position['size'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'entry_cost': position['entry_cost'],
                        'exit_cost': exit_cost,
                        'pnl': pnl,
                        'holding_period': holding_period
                    })
                    
                    position = None
            
            # Open new position if no current position
            if position is None and signal != 0:
                # Entry price
                if signal == 1:  # Buy
                    entry_price = row['ask_price']  # Pay ask
                else:  # Sell
                    entry_price = row['bid_price']  # Receive bid
                
                # Entry cost
                entry_cost = self.calculate_transaction_costs(
                    signal,
                    row['execution_position_size'],
                    row['bid_price'],
                    row['ask_price'],
                    row['spread']
                )
                
                position = {
                    'entry_idx': i,
                    'direction': signal,
                    'size': row['execution_position_size'],
                    'entry_price': entry_price,
                    'entry_cost': entry_cost
                }
        
        # Close any remaining position at end
        if position is not None:
            last_row = signals.iloc[-1]
            
            if position['direction'] == 1:
                exit_price = last_row['bid_price']
                pnl = (exit_price - position['entry_price']) * position['size']
            else:
                exit_price = last_row['ask_price']
                pnl = (position['entry_price'] - exit_price) * position['size']
            
            exit_cost = self.calculate_transaction_costs(
                -position['direction'],
                position['size'],
                last_row['bid_price'],
                last_row['ask_price'],
                last_row['spread']
            )
            pnl -= exit_cost
            
            trades.append({
                'entry_idx': position['entry_idx'],
                'exit_idx': len(signals) - 1,
                'direction': position['direction'],
                'size': position['size'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'entry_cost': position['entry_cost'],
                'exit_cost': exit_cost,
                'pnl': pnl,
                'holding_period': len(signals) - 1 - position['entry_idx']
            })
        
        return pd.DataFrame(trades)
    
    def get_strategy_summary(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for the strategy.
        
        Args:
            trades: DataFrame with executed trades
            
        Returns:
            Dictionary with strategy metrics
        """
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'avg_winner': 0,
                'avg_loser': 0,
                'profit_factor': 0
            }
        
        winners = trades[trades['pnl'] > 0]
        losers = trades[trades['pnl'] <= 0]
        
        summary = {
            'total_trades': len(trades),
            'total_pnl': trades['pnl'].sum(),
            'win_rate': len(winners) / len(trades) if len(trades) > 0 else 0,
            'avg_pnl': trades['pnl'].mean(),
            'avg_winner': winners['pnl'].mean() if len(winners) > 0 else 0,
            'avg_loser': losers['pnl'].mean() if len(losers) > 0 else 0,
            'profit_factor': abs(winners['pnl'].sum() / losers['pnl'].sum()) if len(losers) > 0 and losers['pnl'].sum() != 0 else np.inf,
            'avg_holding_period': trades['holding_period'].mean(),
            'total_entry_cost': trades['entry_cost'].sum(),
            'total_exit_cost': trades['exit_cost'].sum()
        }
        
        return summary
