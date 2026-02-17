"""
Evaluation and visualization module.

This module creates comprehensive visualizations and analysis:
1. Confusion matrices
2. ROC curves
3. Feature importance plots
4. Equity curves
5. Drawdown plots
6. Performance comparison tables
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from typing import Dict, Any, List
import os


class Visualizer:
    """
    Create visualizations and evaluation plots.
    
    Attributes:
        config: EvaluationConfig object
        output_dir: Directory to save plots
    """
    
    def __init__(self, config):
        """
        Initialize visualizer.
        
        Args:
            config: EvaluationConfig object
        """
        self.config = config
        self.output_dir = config.output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = config.figure_dpi
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str, class_names: List[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of model
            class_names: Names of classes
        """
        if class_names is None:
            class_names = ['DOWN', 'NEUTRAL', 'UP']
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if configured
        if self.config.normalize_confusion:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
        print(f"  Saved: {filename}")
    
    def plot_roc_curves(self, models_data: Dict[str, Dict[str, Any]],
                       class_names: List[str] = None):
        """
        Plot ROC curves for all models.
        
        Args:
            models_data: Dictionary with model_name -> {y_true, y_proba}
            class_names: Names of classes
        """
        if class_names is None:
            class_names = ['DOWN', 'NEUTRAL', 'UP']
        
        n_classes = len(class_names)
        
        # Create subplots for each class
        fig, axes = plt.subplots(1, n_classes, figsize=(15, 5))
        
        for class_idx, class_name in enumerate(class_names):
            ax = axes[class_idx]
            
            for model_name, data in models_data.items():
                y_true = data['y_true']
                y_proba = data['y_proba']
                
                # Binary classification for this class
                y_true_binary = (y_true == class_idx).astype(int)
                y_score = y_proba[:, class_idx]
                
                # Compute ROC curve
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)
                
                # Plot
                ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
            
            # Diagonal line
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            
            ax.set_xlabel('False Positive Rate', fontsize=11)
            ax.set_ylabel('True Positive Rate', fontsize=11)
            ax.set_title(f'ROC Curve - {class_name}', fontsize=12, fontweight='bold')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_curves.png'))
        plt.close()
        
        print("  Saved: roc_curves.png")
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, model_name: str):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            model_name: Name of model
        """
        # Take top N features
        top_features = importance_df.head(self.config.top_n_features)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {self.config.top_n_features} Features - {model_name}',
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
        print(f"  Saved: {filename}")
    
    def plot_equity_curve(self, equity_df: pd.DataFrame, model_name: str,
                         baseline_equity: pd.DataFrame = None):
        """
        Plot equity curve.
        
        Args:
            equity_df: DataFrame with timestamp and equity
            model_name: Name of model/strategy
            baseline_equity: Optional baseline equity curve
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # Equity curve
        ax1.plot(equity_df['timestamp'], equity_df['equity'], 
                label=model_name, linewidth=2, color='#2E86AB')
        
        if baseline_equity is not None:
            ax1.plot(baseline_equity['timestamp'], baseline_equity['equity'],
                    label='Baseline', linewidth=2, color='#A23B72', alpha=0.7)
        
        ax1.axhline(y=equity_df['equity'].iloc[0], color='gray', 
                   linestyle='--', linewidth=1, label='Initial Capital')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.set_title(f'Equity Curve - {model_name}', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        running_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - running_max) / running_max * 100
        
        ax2.fill_between(equity_df['timestamp'], drawdown, 0, 
                        color='#C73E1D', alpha=0.5)
        ax2.set_xlabel('Tick', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f'equity_curve_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
        print(f"  Saved: {filename}")
    
    def plot_returns_distribution(self, equity_df: pd.DataFrame, model_name: str):
        """
        Plot distribution of returns.
        
        Args:
            equity_df: DataFrame with returns
            model_name: Name of model
        """
        returns = equity_df['returns'].values
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(returns, bins=50, edgecolor='black', alpha=0.7, color='#2E86AB')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Return')
        ax1.set_xlabel('Returns', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Returns Distribution - {model_name}', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f'returns_distribution_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
        print(f"  Saved: {filename}")
    
    def create_comparison_table(self, comparison_df: pd.DataFrame):
        """
        Create and save comparison table.
        
        Args:
            comparison_df: DataFrame with model comparisons
        """
        # Save as CSV
        comparison_df.to_csv(os.path.join(self.output_dir, 'model_comparison.csv'), 
                           index=False)
        
        # Create visual table
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=comparison_df.values,
                        colLabels=comparison_df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(comparison_df.columns)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(comparison_df) + 1):
            for j in range(len(comparison_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E8F4F8')
        
        plt.title('Model Comparison', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'), 
                   bbox_inches='tight')
        plt.close()
        
        print("  Saved: model_comparison.csv")
        print("  Saved: model_comparison.png")
    
    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str, class_names: List[str] = None):
        """
        Print classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of model
            class_names: Names of classes
        """
        if class_names is None:
            class_names = ['DOWN', 'NEUTRAL', 'UP']
        
        print(f"\nClassification Report - {model_name}")
        print("=" * 60)
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    def save_metrics_json(self, metrics: Dict[str, Any], filename: str = 'metrics.json'):
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Dictionary with metrics
            filename: Output filename
        """
        import json
        
        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, pd.DataFrame):
                return "DataFrame (not serialized)"
            elif isinstance(obj, pd.Series):
                return "Series (not serialized)"
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        metrics_clean = convert_types(metrics)
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics_clean, f, indent=2)
        
        print(f"  Saved: {filename}")


def create_all_visualizations(models_results: Dict[str, Any], 
                             config) -> None:
    """
    Create all visualizations for the project.
    
    Args:
        models_results: Dictionary with all model results
        config: EvaluationConfig object
    """
    print("\nCreating visualizations...")
    
    viz = Visualizer(config)
    
    # For each model
    for model_name, results in models_results.items():
        if 'y_true' in results and 'y_pred' in results:
            # Confusion matrix
            viz.plot_confusion_matrix(
                results['y_true'],
                results['y_pred'],
                model_name
            )
            
            # Classification report
            viz.print_classification_report(
                results['y_true'],
                results['y_pred'],
                model_name
            )
        
        # Feature importance
        if 'feature_importance' in results and results['feature_importance'] is not None:
            viz.plot_feature_importance(
                results['feature_importance'],
                model_name
            )
        
        # Equity curve
        if 'equity_curve' in results:
            baseline_equity = models_results.get('Baseline', {}).get('equity_curve')
            viz.plot_equity_curve(
                results['equity_curve'],
                model_name,
                baseline_equity
            )
            
            # Returns distribution
            viz.plot_returns_distribution(
                results['equity_curve'],
                model_name
            )
    
    # ROC curves (all models together)
    roc_data = {}
    for model_name, results in models_results.items():
        if 'y_true' in results and 'y_proba' in results:
            roc_data[model_name] = {
                'y_true': results['y_true'],
                'y_proba': results['y_proba']
            }
    
    if len(roc_data) > 0:
        viz.plot_roc_curves(roc_data)
    
    print("✓ All visualizations created")
