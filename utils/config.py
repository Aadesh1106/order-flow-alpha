"""
Configuration module for Order Flow Imbalance prediction system.

This module centralizes all configuration parameters to ensure consistency
across the entire system and make it easy to tune hyperparameters.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class SimulationConfig:
    """Configuration for HFT tick data simulation."""
    
    # Data generation
    n_ticks: int = 10000
    initial_price: float = 100.0
    tick_size: float = 0.01
    
    # Price dynamics
    drift: float = 0.0  # No drift for realistic microstructure
    volatility: float = 0.002  # Per-tick volatility
    
    # Spread dynamics
    min_spread: float = 0.01  # Minimum bid-ask spread
    max_spread: float = 0.05  # Maximum bid-ask spread
    spread_mean: float = 0.02
    spread_std: float = 0.005
    
    # Volume dynamics
    volume_mean: float = 1000.0
    volume_std: float = 300.0
    min_volume: float = 100.0
    
    # Depth dynamics
    depth_mean: float = 5000.0
    depth_std: float = 1500.0
    min_depth: float = 500.0
    
    # Autocorrelation parameters
    price_autocorr: float = 0.05  # Weak mean reversion
    ofi_autocorr: float = 0.3  # OFI has persistence
    
    # Random seed for reproducibility
    random_seed: int = 42


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Rolling window sizes
    volatility_window: int = 20
    volume_window: int = 10
    pressure_window: int = 15
    
    # Lagged features
    n_lags: int = 3  # Number of lagged features to include
    
    # Target definition
    price_change_threshold: float = 0.0001  # Threshold for UP/DOWN classification
    # If |price_change| < threshold → NEUTRAL
    # If price_change > threshold → UP
    # If price_change < -threshold → DOWN
    
    # Feature scaling
    scale_features: bool = True


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    
    # Train/test split
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Random seed
    random_seed: int = 42
    
    # Logistic Regression
    logistic_params: Dict[str, Any] = None
    
    # Random Forest
    rf_params: Dict[str, Any] = None
    
    # XGBoost
    xgb_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default model parameters."""
        if self.logistic_params is None:
            self.logistic_params = {
                'max_iter': 1000,
                'random_state': self.random_seed,
                'class_weight': 'balanced',  # Handle class imbalance
                'solver': 'lbfgs',
                'multi_class': 'multinomial'
            }
        
        if self.rf_params is None:
            self.rf_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 50,
                'min_samples_leaf': 20,
                'random_state': self.random_seed,
                'class_weight': 'balanced',
                'n_jobs': -1
            }
        
        if self.xgb_params is None:
            self.xgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_seed,
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'tree_method': 'hist'
            }


@dataclass
class StrategyConfig:
    """Configuration for trading strategy."""
    
    # Entry thresholds
    entry_threshold: float = 0.55  # Minimum probability to trade
    
    # Position sizing
    base_position_size: float = 100.0  # Base number of shares
    use_confidence_sizing: bool = True  # Scale position by confidence
    max_position_size: float = 500.0
    
    # Latency (in ticks)
    execution_latency: int = 2  # Delay between signal and execution
    
    # Transaction costs
    commission_per_share: float = 0.001  # $0.001 per share
    use_spread_cost: bool = True  # Pay half-spread on entry/exit
    
    # Risk management
    max_holding_period: int = 50  # Maximum ticks to hold position
    stop_loss_pct: float = 0.005  # 0.5% stop loss


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    # Initial capital
    initial_capital: float = 100000.0
    
    # Slippage model
    slippage_model: str = 'fixed'  # 'fixed' or 'proportional'
    fixed_slippage: float = 0.0  # Additional slippage in dollars
    proportional_slippage: float = 0.0001  # Proportional to price
    
    # Performance metrics
    risk_free_rate: float = 0.02  # Annual risk-free rate for Sharpe
    trading_days_per_year: int = 252
    
    # Baseline model for comparison
    baseline_type: str = 'random'  # 'random' or 'momentum'


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and visualization."""
    
    # Output directory
    output_dir: str = 'results'
    
    # Figure settings
    figure_dpi: int = 300
    figure_size: tuple = (12, 8)
    
    # Feature importance
    top_n_features: int = 15
    
    # ROC curve
    plot_roc: bool = True
    
    # Confusion matrix
    normalize_confusion: bool = True


class Config:
    """Master configuration class combining all configs."""
    
    def __init__(self):
        self.simulation = SimulationConfig()
        self.features = FeatureConfig()
        self.models = ModelConfig()
        self.strategy = StrategyConfig()
        self.backtest = BacktestConfig()
        self.evaluation = EvaluationConfig()
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        lines = ["Configuration:"]
        for attr_name in ['simulation', 'features', 'models', 'strategy', 'backtest', 'evaluation']:
            attr = getattr(self, attr_name)
            lines.append(f"\n{attr_name.upper()}:")
            lines.append(f"  {attr}")
        return "\n".join(lines)


# Global configuration instance
config = Config()
