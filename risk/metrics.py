"""
Risk and performance metrics module.

This module computes comprehensive risk and performance metrics:
1. Return metrics (total, annualized, CAGR)
2. Risk metrics (volatility, VaR, CVaR)
3. Risk-adjusted returns (Sharpe, Sortino, Calmar)
4. Drawdown analysis
5. Trade statistics

Mathematical Formulas:
---------------------
1. Sharpe Ratio:
   SR = (R_p - R_f) / σ_p
   where R_p = portfolio return, R_f = risk-free rate, σ_p = volatility

2. Sortino Ratio:
   Sortino = (R_p - R_f) / σ_downside
   where σ_downside = std of negative returns only

3. Maximum Drawdown:
   MDD = max_t [(RunningMax_t - Value_t) / RunningMax_t]

4. Calmar Ratio:
   Calmar = AnnualizedReturn / |MaxDrawdown|

5. Value at Risk (VaR):
   VaR_α = -Quantile(returns, α)
   
6. Conditional VaR (CVaR):
   CVaR_α = -E[returns | returns < -VaR_α]
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy import stats


class RiskMetrics:
    """
    Compute risk and performance metrics.
    
    Attributes:
        config: BacktestConfig object
    """
    
    def __init__(self, config):
        """
        Initialize risk metrics calculator.
        
        Args:
            config: BacktestConfig object
        """
        self.config = config
    
    def compute_all_metrics(self, equity_curve: pd.DataFrame,
                           trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute all risk and performance metrics.
        
        Args:
            equity_curve: DataFrame with timestamp, equity, returns
            trades: DataFrame with trade details
            
        Returns:
            Dictionary with all metrics
        """
        print("\nComputing risk metrics...")
        
        metrics = {}
        
        # Return metrics
        metrics.update(self._compute_return_metrics(equity_curve))
        
        # Risk metrics
        metrics.update(self._compute_risk_metrics(equity_curve))
        
        # Risk-adjusted metrics
        metrics.update(self._compute_risk_adjusted_metrics(equity_curve))
        
        # Drawdown metrics
        metrics.update(self._compute_drawdown_metrics(equity_curve))
        
        # Trade statistics
        if len(trades) > 0:
            metrics.update(self._compute_trade_statistics(trades))
        
        print("✓ Risk metrics computed")
        
        return metrics
    
    def _compute_return_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Compute return-based metrics."""
        initial = equity_curve['equity'].iloc[0]
        final = equity_curve['equity'].iloc[-1]
        
        total_return = (final - initial) / initial
        
        # Annualized return (assuming ticks are seconds)
        n_ticks = len(equity_curve)
        ticks_per_day = 23400  # 6.5 hours
        n_days = n_ticks / ticks_per_day
        n_years = n_days / self.config.trading_days_per_year
        
        if n_years > 0:
            annualized_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            annualized_return = 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100
        }
    
    def _compute_risk_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Compute risk-based metrics."""
        returns = equity_curve['returns'].values
        
        # Volatility (annualized)
        ticks_per_day = 23400
        scaling_factor = np.sqrt(ticks_per_day * self.config.trading_days_per_year)
        volatility = np.std(returns) * scaling_factor
        
        # Downside volatility (only negative returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_vol = np.std(negative_returns) * scaling_factor
        else:
            downside_vol = 0
        
        # Value at Risk (95% and 99%)
        var_95 = -np.percentile(returns, 5)
        var_99 = -np.percentile(returns, 1)
        
        # Conditional VaR (CVaR / Expected Shortfall)
        cvar_95 = -np.mean(returns[returns <= -var_95]) if len(returns[returns <= -var_95]) > 0 else 0
        cvar_99 = -np.mean(returns[returns <= -var_99]) if len(returns[returns <= -var_99]) > 0 else 0
        
        return {
            'volatility': volatility,
            'volatility_pct': volatility * 100,
            'downside_volatility': downside_vol,
            'downside_volatility_pct': downside_vol * 100,
            'var_95': var_95,
            'var_95_pct': var_95 * 100,
            'var_99': var_99,
            'var_99_pct': var_99 * 100,
            'cvar_95': cvar_95,
            'cvar_95_pct': cvar_95 * 100,
            'cvar_99': cvar_99,
            'cvar_99_pct': cvar_99 * 100
        }
    
    def _compute_risk_adjusted_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Compute risk-adjusted return metrics."""
        returns = equity_curve['returns'].values
        
        # Scaling for annualization
        ticks_per_day = 23400
        scaling_factor = np.sqrt(ticks_per_day * self.config.trading_days_per_year)
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Risk-free rate per tick
        rf_per_tick = self.config.risk_free_rate / (ticks_per_day * self.config.trading_days_per_year)
        
        # Sharpe Ratio
        if std_return > 0:
            sharpe = (mean_return - rf_per_tick) / std_return * scaling_factor
        else:
            sharpe = 0
        
        # Sortino Ratio
        negative_returns = returns[returns < rf_per_tick]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            if downside_std > 0:
                sortino = (mean_return - rf_per_tick) / downside_std * scaling_factor
            else:
                sortino = 0
        else:
            sortino = 0
        
        # Calmar Ratio (return / max drawdown)
        equity = equity_curve['equity'].values
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_dd = abs(np.min(drawdown))
        
        # Annualized return
        initial = equity[0]
        final = equity[-1]
        total_return = (final - initial) / initial
        n_ticks = len(equity)
        n_years = n_ticks / (ticks_per_day * self.config.trading_days_per_year)
        
        if n_years > 0:
            ann_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            ann_return = 0
        
        if max_dd > 0:
            calmar = ann_return / max_dd
        else:
            calmar = 0
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar
        }
    
    def _compute_drawdown_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Compute drawdown-related metrics."""
        equity = equity_curve['equity'].values
        
        # Running maximum
        running_max = np.maximum.accumulate(equity)
        
        # Drawdown series
        drawdown = (equity - running_max) / running_max
        
        # Maximum drawdown
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        
        # Drawdown duration
        # Find start of max drawdown (last peak before max dd)
        peak_idx = np.argmax(running_max[:max_dd_idx+1] == running_max[max_dd_idx])
        dd_duration = max_dd_idx - peak_idx
        
        # Average drawdown
        avg_dd = np.mean(drawdown[drawdown < 0]) if len(drawdown[drawdown < 0]) > 0 else 0
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'max_drawdown_duration': dd_duration,
            'avg_drawdown': avg_dd,
            'avg_drawdown_pct': avg_dd * 100,
            'drawdown_series': drawdown
        }
    
    def _compute_trade_statistics(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Compute trade-level statistics."""
        if len(trades) == 0:
            return {}
        
        # Win/loss statistics
        winners = trades[trades['pnl'] > 0]
        losers = trades[trades['pnl'] <= 0]
        
        n_winners = len(winners)
        n_losers = len(losers)
        n_total = len(trades)
        
        win_rate = n_winners / n_total if n_total > 0 else 0
        
        # PnL statistics
        total_pnl = trades['pnl'].sum()
        avg_pnl = trades['pnl'].mean()
        
        avg_winner = winners['pnl'].mean() if n_winners > 0 else 0
        avg_loser = losers['pnl'].mean() if n_losers > 0 else 0
        
        # Profit factor
        gross_profit = winners['pnl'].sum() if n_winners > 0 else 0
        gross_loss = abs(losers['pnl'].sum()) if n_losers > 0 else 0
        
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = np.inf if gross_profit > 0 else 0
        
        # Holding period
        avg_holding = trades['holding_period'].mean()
        
        # Largest winner/loser
        largest_winner = trades['pnl'].max()
        largest_loser = trades['pnl'].min()
        
        return {
            'total_trades': n_total,
            'winning_trades': n_winners,
            'losing_trades': n_losers,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'profit_factor': profit_factor,
            'avg_holding_period': avg_holding,
            'largest_winner': largest_winner,
            'largest_loser': largest_loser,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def print_metrics_summary(self, metrics: Dict[str, Any]):
        """
        Print formatted summary of metrics.
        
        Args:
            metrics: Dictionary with all metrics
        """
        print("\n" + "="*60)
        print("PERFORMANCE METRICS SUMMARY")
        print("="*60)
        
        print("\nRETURN METRICS:")
        print(f"  Total Return:       {metrics.get('total_return_pct', 0):>8.2f}%")
        print(f"  Annualized Return:  {metrics.get('annualized_return_pct', 0):>8.2f}%")
        
        print("\nRISK METRICS:")
        print(f"  Volatility (Ann.):  {metrics.get('volatility_pct', 0):>8.2f}%")
        print(f"  Max Drawdown:       {metrics.get('max_drawdown_pct', 0):>8.2f}%")
        print(f"  VaR (95%):          {metrics.get('var_95_pct', 0):>8.4f}%")
        print(f"  CVaR (95%):         {metrics.get('cvar_95_pct', 0):>8.4f}%")
        
        print("\nRISK-ADJUSTED RETURNS:")
        print(f"  Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):>8.2f}")
        print(f"  Sortino Ratio:      {metrics.get('sortino_ratio', 0):>8.2f}")
        print(f"  Calmar Ratio:       {metrics.get('calmar_ratio', 0):>8.2f}")
        
        if 'total_trades' in metrics:
            print("\nTRADE STATISTICS:")
            print(f"  Total Trades:       {metrics.get('total_trades', 0):>8d}")
            print(f"  Win Rate:           {metrics.get('win_rate_pct', 0):>8.2f}%")
            print(f"  Profit Factor:      {metrics.get('profit_factor', 0):>8.2f}")
            print(f"  Avg PnL:            ${metrics.get('avg_pnl', 0):>8.2f}")
            print(f"  Avg Winner:         ${metrics.get('avg_winner', 0):>8.2f}")
            print(f"  Avg Loser:          ${metrics.get('avg_loser', 0):>8.2f}")
        
        print("="*60)
