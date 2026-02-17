"""
Main entry point for Order Flow Imbalance & Short-Term Price Prediction.

This script orchestrates the entire pipeline:
1. Data simulation
2. Feature engineering
3. Model training
4. Strategy execution
5. Backtesting
6. Evaluation and visualization

Usage:
    python main.py
"""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# Import modules
from utils.config import config
from data.simulation import generate_hft_data
from features.microstructure import engineer_features
from models.logistic import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from strategy.trading_strategy import TradingStrategy
from backtesting.backtester import Backtester
from risk.metrics import RiskMetrics
from evaluation.visualizer import create_all_visualizations, Visualizer


def split_data(df: pd.DataFrame, feature_names: list, config):
    """
    Split data into train/validation/test sets.
    
    Uses time-based splitting to avoid look-ahead bias.
    
    Args:
        df: DataFrame with features and target
        feature_names: List of feature column names
        config: ModelConfig object
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, 
                 train_idx, val_idx, test_idx)
    """
    n = len(df)
    
    # Time-based split
    train_end = int(n * config.train_ratio)
    val_end = int(n * (config.train_ratio + config.validation_ratio))
    
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)
    
    # Extract features and target
    X = df[feature_names].values
    y = df['target'].values
    
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_idx)} samples ({config.train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_idx)} samples ({config.validation_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_idx)} samples ({config.test_ratio*100:.0f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx


def train_all_models(X_train, y_train, X_val, y_val, feature_names, config):
    """
    Train all models.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        feature_names: List of feature names
        config: ModelConfig object
        
    Returns:
        Dictionary of trained models
    """
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    models = {}
    
    # 1. Logistic Regression
    lr_model = LogisticRegressionModel(
        params=config.logistic_params,
        scale_features=True
    )
    lr_model.train(X_train, y_train, X_val, y_val, feature_names)
    models['Logistic Regression'] = lr_model
    
    # 2. Random Forest
    rf_model = RandomForestModel(
        params=config.rf_params,
        scale_features=False
    )
    rf_model.train(X_train, y_train, X_val, y_val, feature_names)
    models['Random Forest'] = rf_model
    
    # 3. XGBoost
    xgb_model = XGBoostModel(
        params=config.xgb_params,
        scale_features=False
    )
    xgb_model.train(X_train, y_train, X_val, y_val, feature_names)
    models['XGBoost'] = xgb_model
    
    print("\n✓ All models trained successfully")
    
    return models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models on test set.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary with evaluation results for each model
    """
    print("\n" + "="*60)
    print("EVALUATING MODELS")
    print("="*60)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Accuracy
        accuracy = np.mean(y_pred == y_test)
        print(f"  Test Accuracy: {accuracy:.4f}")
        
        # Feature importance
        feature_importance = model.get_feature_importance()
        
        results[model_name] = {
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_true': y_test,
            'accuracy': accuracy,
            'feature_importance': feature_importance
        }
    
    return results


def run_trading_strategy(model_name, model, X_test, y_test, df_test, config):
    """
    Run trading strategy for a model.
    
    Args:
        model_name: Name of model
        model: Trained model
        X_test: Test features
        y_test: Test labels
        df_test: Test DataFrame with tick data
        config: Config object
        
    Returns:
        Dictionary with strategy results
    """
    print(f"\n{model_name} Strategy:")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Initialize strategy
    strategy = TradingStrategy(config.strategy)
    
    # Generate signals
    signals = strategy.generate_signals(y_pred, y_proba)
    
    # Apply latency
    signals = strategy.apply_latency(signals, df_test)
    
    # Execute trades
    trades = strategy.execute_trades(signals)
    
    # Strategy summary
    summary = strategy.get_strategy_summary(trades)
    
    print(f"  Total trades: {summary['total_trades']}")
    print(f"  Win rate:     {summary['win_rate']*100:.2f}%")
    print(f"  Total PnL:    ${summary['total_pnl']:.2f}")
    
    # Backtest
    backtester = Backtester(config.backtest)
    backtest_results = backtester.run_backtest(trades, df_test)
    
    # Risk metrics
    risk_calc = RiskMetrics(config.backtest)
    risk_metrics = risk_calc.compute_all_metrics(
        backtest_results['equity_curve'],
        trades
    )
    
    # Combine results
    results = {
        **summary,
        **backtest_results,
        **risk_metrics,
        'trades': trades,
        'signals': signals
    }
    
    return results


def run_baseline_strategy(df_test, config):
    """
    Run baseline (random) strategy.
    
    Args:
        df_test: Test DataFrame
        config: Config object
        
    Returns:
        Dictionary with baseline results
    """
    print("\nBaseline (Random) Strategy:")
    
    n_test = len(df_test)
    rng = np.random.RandomState(42)
    
    # Random predictions
    y_pred = rng.choice([0, 1, 2], size=n_test, p=[0.33, 0.34, 0.33])
    
    # Random probabilities
    y_proba = np.zeros((n_test, 3))
    for i in range(n_test):
        pred = y_pred[i]
        probs = rng.dirichlet([1, 1, 1])
        probs[pred] += 0.3  # Boost predicted class slightly
        probs = probs / probs.sum()
        y_proba[i] = probs
    
    # Run strategy
    strategy = TradingStrategy(config.strategy)
    signals = strategy.generate_signals(y_pred, y_proba)
    signals = strategy.apply_latency(signals, df_test)
    trades = strategy.execute_trades(signals)
    
    summary = strategy.get_strategy_summary(trades)
    print(f"  Total trades: {summary['total_trades']}")
    print(f"  Win rate:     {summary['win_rate']*100:.2f}%")
    print(f"  Total PnL:    ${summary['total_pnl']:.2f}")
    
    # Backtest
    backtester = Backtester(config.backtest)
    backtest_results = backtester.run_backtest(trades, df_test)
    
    # Risk metrics
    risk_calc = RiskMetrics(config.backtest)
    risk_metrics = risk_calc.compute_all_metrics(
        backtest_results['equity_curve'],
        trades
    )
    
    results = {
        **summary,
        **backtest_results,
        **risk_metrics,
        'trades': trades,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    return results


def main():
    """Main execution pipeline."""
    
    print("="*60)
    print("ORDER FLOW IMBALANCE & SHORT-TERM PRICE PREDICTION")
    print("="*60)
    
    overall_start = time.time()
    
    # 1. Generate HFT data
    print("\n[1/7] Generating HFT tick data...")
    df_raw = generate_hft_data(config.simulation)
    
    # 2. Engineer features
    print("\n[2/7] Engineering microstructure features...")
    df_features, feature_names = engineer_features(df_raw, config.features)
    
    # 3. Split data
    print("\n[3/7] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx = \
        split_data(df_features, feature_names, config.models)
    
    # 4. Train models
    print("\n[4/7] Training models...")
    models = train_all_models(X_train, y_train, X_val, y_val, feature_names, config.models)
    
    # 5. Evaluate models
    print("\n[5/7] Evaluating models...")
    eval_results = evaluate_models(models, X_test, y_test)
    
    # 6. Run trading strategies
    print("\n[6/7] Running trading strategies...")
    print("="*60)
    
    # Get test data
    df_test = df_features.iloc[test_idx].reset_index(drop=True)
    
    # Run strategy for each model
    strategy_results = {}
    for model_name, model in models.items():
        strategy_results[model_name] = run_trading_strategy(
            model_name, model, X_test, y_test, df_test, config
        )
    
    # Run baseline
    baseline_results = run_baseline_strategy(df_test, config)
    strategy_results['Baseline'] = baseline_results
    
    # 7. Create visualizations
    print("\n[7/7] Creating visualizations and reports...")
    print("="*60)
    
    # Combine evaluation and strategy results
    all_results = {}
    for model_name in eval_results.keys():
        all_results[model_name] = {
            **eval_results[model_name],
            **strategy_results[model_name]
        }
    
    # Add baseline
    all_results['Baseline'] = baseline_results
    
    # Create visualizations
    create_all_visualizations(all_results, config.evaluation)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    # Create comparison table
    comparison_data = []
    for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost', 'Baseline']:
        results = all_results[model_name]
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{results.get('accuracy', 0):.4f}" if 'accuracy' in results else 'N/A',
            'Total Return (%)': f"{results.get('total_return_pct', 0):.2f}",
            'Sharpe Ratio': f"{results.get('sharpe_ratio', 0):.2f}",
            'Max DD (%)': f"{results.get('max_drawdown_pct', 0):.2f}",
            'Win Rate (%)': f"{results.get('win_rate', 0)*100:.2f}",
            'Total Trades': results.get('total_trades', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n", comparison_df.to_string(index=False))
    
    # Save comparison
    viz = Visualizer(config.evaluation)
    viz.create_comparison_table(comparison_df)
    
    # Print best model
    best_model = comparison_df.iloc[:-1].loc[
        comparison_df.iloc[:-1]['Sharpe Ratio'].astype(float).idxmax()
    ]['Model']
    print(f"\n🏆 Best Model (by Sharpe Ratio): {best_model}")
    
    # Print detailed metrics for best model
    risk_calc = RiskMetrics(config.backtest)
    risk_calc.print_metrics_summary(all_results[best_model])
    
    # Save all metrics
    viz.save_metrics_json(all_results, 'all_results.json')
    
    # Total time
    total_time = time.time() - overall_start
    print(f"\n✓ Pipeline complete in {total_time:.2f}s")
    print(f"✓ Results saved to '{config.evaluation.output_dir}/' directory")
    
    print("\n" + "="*60)
    print("Thank you for using the Order Flow Imbalance Prediction System!")
    print("="*60)


if __name__ == "__main__":
    main()
