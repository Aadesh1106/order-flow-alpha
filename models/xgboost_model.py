"""
XGBoost model for price movement prediction.

XGBoost (Extreme Gradient Boosting) is a state-of-the-art gradient boosting
algorithm that builds trees sequentially, where each tree corrects errors
from previous trees.

Mathematical Model:
------------------
Objective function:
    L = Σ_i l(y_i, ŷ_i) + Σ_k Ω(f_k)

where:
    - l(): loss function (e.g., log loss for classification)
    - Ω(f): regularization term = γT + (λ/2)||w||²
    - T: number of leaves
    - w: leaf weights

Prediction:
    ŷ_i = Σ_k f_k(x_i)  where f_k is tree k

Advantages:
- State-of-the-art performance
- Built-in regularization (L1, L2)
- Handles missing values
- Feature importance
- Fast training with histogram-based algorithm
- Supports early stopping

Limitations:
- More hyperparameters to tune
- Can overfit if not regularized
- Less interpretable
"""

import numpy as np
import xgboost as xgb
from typing import Dict, Any
import time

from models.base import BasePredictionModel


class XGBoostModel(BasePredictionModel):
    """
    XGBoost model for multiclass classification.
    
    Inherits from BasePredictionModel and implements training logic.
    """
    
    def __init__(self, params: Dict[str, Any] = None, scale_features: bool = False):
        """
        Initialize XGBoost model.
        
        Args:
            params: Model hyperparameters (XGBoost params)
            scale_features: Whether to scale features (not needed for XGB)
        """
        super().__init__(name="XGBoost", scale_features=scale_features)
        self.params = params or {}
        self._create_model()
    
    def _create_model(self, **kwargs):
        """Create XGBoost classifier instance."""
        self.model = xgb.XGBClassifier(**self.params)
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (used for early stopping if provided)
            y_val: Validation labels
            
        Returns:
            Dictionary with training metrics
        """
        start_time = time.time()
        
        # Use early stopping if validation set provided
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            # Get best iteration
            best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') else None
        else:
            self.model.fit(X_train, y_train, verbose=False)
            best_iteration = None
        
        # Training metrics
        train_score = self.model.score(X_train, y_train)
        
        metrics = {
            'train_accuracy': train_score,
            'training_time': time.time() - start_time,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'best_iteration': best_iteration
        }
        
        # Validation score if provided
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            metrics['val_accuracy'] = val_score
            print(f"  Train accuracy: {train_score:.4f}")
            print(f"  Val accuracy:   {val_score:.4f}")
            if best_iteration:
                print(f"  Best iteration: {best_iteration}")
        else:
            print(f"  Train accuracy: {train_score:.4f}")
        
        print(f"  Training time:  {metrics['training_time']:.2f}s")
        print(f"  Trees:          {metrics['n_estimators']}")
        
        return metrics
