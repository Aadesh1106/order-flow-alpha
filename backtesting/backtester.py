"""
Backtesting engine for trading strategies.

This module provides a realistic backtesting framework that:
1. Simulates strategy execution on historical data
2. Tracks portfolio value over time
3. Accounts for transaction costs and latency
4. Computes performance metrics
5. Compares against baseline strategies

Backtesting Principles:
----------------------
1. No look-ahead bias: Only use information available at time t
2. Realistic execution: Include latency and slippage
3. Transaction costs: Model commissions and spreads
4. Position tracking: Maintain accurate portfolio state
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import time


class Backtester:
    """
    Backtesting engine for trading strategies.
    
    Attributes:
        config: BacktestConfig object
        initial_capital: Starting capital
        portfolio_value: Current portfolio value
        cash: Available cash
        positions: Current positions
        equity_curve: Time series of portfolio values
    """
    
    def __init__(self, config):
        """
        Initialize backtester.
        
        Args:
            config: BacktestConfig object
        """
        self.config = config
        self.initial_capital = config.initial_capital
        self.reset()
    
    def reset(self):
        """Reset backtester state."""
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = []
        self.equity_curve = []
    
    def run_backtest(self, trades: pd.DataFrame, tick_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest on executed trades.
        
        Args:
            trades: DataFrame with trade details from strategy
            tick_data: Original tick data
            
        Returns:
            Dictionary with backtest results
        """
        start_time = time.time()
        print("\nRunning backtest...")
        
        self.reset()
        
        # Build equity curve
        equity_curve = np.full(len(tick_data), self.initial_capital, dtype=float)
        
        # Track cumulative PnL
        cumulative_pnl = 0
        
        # Process each trade
        for _, trade in trades.iterrows():
            entry_idx = int(trade['entry_idx'])
            exit_idx = int(trade['exit_idx'])
            pnl = trade['pnl']
            
            # Update cumulative PnL
            cumulative_pnl += pnl
            
            # Update equity curve from exit onwards
            equity_curve[exit_idx:] = self.initial_capital + cumulative_pnl
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame({
            'timestamp': tick_data['timestamp'].values,
            'equity': equity_curve,
            'returns': pd.Series(equity_curve).pct_change().fillna(0).values
        })
        
        # Calculate metrics
        final_equity = equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Sharpe ratio (annualized)
        returns = equity_df['returns'].values
        if len(returns) > 0 and np.std(returns) > 0:
            # Assume each tick is 1 second, scale to annual
            ticks_per_day = 23400  # 6.5 hours × 3600 seconds
            scaling_factor = np.sqrt(ticks_per_day * self.config.trading_days_per_year)
            sharpe = (np.mean(returns) - self.config.risk_free_rate / (ticks_per_day * self.config.trading_days_per_year)) / np.std(returns) * scaling_factor
        else:
            sharpe = 0
        
        results = {
            'equity_curve': equity_df,
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe,
            'total_trades': len(trades),
            'backtest_time': time.time() - start_time
        }
        
        print(f"✓ Backtest complete in {results['backtest_time']:.2f}s")
        print(f"  Initial capital: ${self.initial_capital:,.2f}")
        print(f"  Final equity:    ${final_equity:,.2f}")
        print(f"  Total return:    {total_return*100:+.2f}%")
        print(f"  Max drawdown:    {max_drawdown*100:.2f}%")
        print(f"  Sharpe ratio:    {sharpe:.2f}")
        print(f"  Total trades:    {len(trades)}")
        
        return results
    
    def run_baseline_strategy(self, tick_data: pd.DataFrame, 
                             test_indices: np.ndarray) -> Dict[str, Any]:
        """
        Run a simple baseline strategy for comparison.
        
        Baseline: Random trading with same frequency as ML strategy.
        
        Args:
            tick_data: Tick data
            test_indices: Indices of test set
            
        Returns:
            Dictionary with baseline results
        """
        print("\nRunning baseline strategy (random)...")
        
        # Generate random signals
        n_test = len(test_indices)
        rng = np.random.RandomState(42)
        
        # Random predictions with same class distribution
        random_preds = rng.choice([0, 1, 2], size=n_test, p=[0.33, 0.34, 0.33])
        
        # Random probabilities (but consistent with prediction)
        random_probs = np.zeros((n_test, 3))
        for i in range(n_test):
            pred = random_preds[i]
            # Make predicted class have highest probability
            probs = rng.dirichlet([1, 1, 1])
            # Boost predicted class
            probs[pred] += 0.5
            probs = probs / probs.sum()
            random_probs[i] = probs
        
        return random_preds, random_probs
    
    def compare_strategies(self, ml_results: Dict[str, Any],
                          baseline_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Compare ML strategy against baseline.
        
        Args:
            ml_results: Results from ML strategy
            baseline_results: Results from baseline strategy
            
        Returns:
            DataFrame with comparison
        """
        comparison = pd.DataFrame({
            'Metric': [
                'Total Return (%)',
                'Max Drawdown (%)',
                'Sharpe Ratio',
                'Total Trades',
                'Win Rate (%)',
                'Profit Factor'
            ],
            'ML Strategy': [
                ml_results.get('total_return_pct', 0),
                ml_results.get('max_drawdown_pct', 0),
                ml_results.get('sharpe_ratio', 0),
                ml_results.get('total_trades', 0),
                ml_results.get('win_rate', 0) * 100,
                ml_results.get('profit_factor', 0)
            ],
            'Baseline': [
                baseline_results.get('total_return_pct', 0),
                baseline_results.get('max_drawdown_pct', 0),
                baseline_results.get('sharpe_ratio', 0),
                baseline_results.get('total_trades', 0),
                baseline_results.get('win_rate', 0) * 100,
                baseline_results.get('profit_factor', 0)
            ]
        })
        
        # Calculate improvement
        comparison['Improvement'] = comparison['ML Strategy'] - comparison['Baseline']
        
        return comparison


def create_equity_curve(initial_capital: float, trades: pd.DataFrame,
                       n_ticks: int) -> np.ndarray:
    """
    Create equity curve from trades.
    
    Args:
        initial_capital: Starting capital
        trades: DataFrame with trades
        n_ticks: Total number of ticks
        
    Returns:
        Array with equity at each tick
    """
    equity = np.full(n_ticks, initial_capital, dtype=float)
    
    cumulative_pnl = 0
    for _, trade in trades.iterrows():
        exit_idx = int(trade['exit_idx'])
        cumulative_pnl += trade['pnl']
        equity[exit_idx:] = initial_capital + cumulative_pnl
    
    return equity
