"""
Logistic Regression model for price movement prediction.

Logistic Regression is a linear model that estimates class probabilities
using the logistic (sigmoid) function. For multiclass problems, it uses
multinomial logistic regression (softmax).

Mathematical Model:
------------------
For class k, the probability is:
    P(y = k | x) = exp(w_k^T x + b_k) / Σ_j exp(w_j^T x + b_j)

where:
    - w_k: weight vector for class k
    - b_k: bias for class k
    - x: feature vector

Advantages:
- Fast training and prediction
- Probabilistic output
- Interpretable (linear decision boundary)
- Good baseline model

Limitations:
- Assumes linear separability
- May underfit complex patterns
- Sensitive to feature scaling
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any
import time

from models.base import BasePredictionModel


class LogisticRegressionModel(BasePredictionModel):
    """
    Logistic Regression model for multiclass classification.
    
    Inherits from BasePredictionModel and implements training logic.
    """
    
    def __init__(self, params: Dict[str, Any] = None, scale_features: bool = True):
        """
        Initialize Logistic Regression model.
        
        Args:
            params: Model hyperparameters (sklearn LogisticRegression params)
            scale_features: Whether to scale features (recommended for LR)
        """
        super().__init__(name="Logistic Regression", scale_features=scale_features)
        self.params = params or {}
        self._create_model()
    
    def _create_model(self, **kwargs):
        """Create sklearn LogisticRegression instance."""
        self.model = LogisticRegression(**self.params)
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the logistic regression model.
        
        Args:
            X_train: Training features (scaled)
            y_train: Training labels
            X_val: Validation features (scaled, unused for LR)
            y_val: Validation labels (unused for LR)
            
        Returns:
            Dictionary with training metrics
        """
        start_time = time.time()
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Training metrics
        train_score = self.model.score(X_train, y_train)
        
        metrics = {
            'train_accuracy': train_score,
            'training_time': time.time() - start_time,
            'n_iterations': self.model.n_iter_[0] if hasattr(self.model, 'n_iter_') else None
        }
        
        # Validation score if provided
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            metrics['val_accuracy'] = val_score
            print(f"  Train accuracy: {train_score:.4f}")
            print(f"  Val accuracy:   {val_score:.4f}")
        else:
            print(f"  Train accuracy: {train_score:.4f}")
        
        print(f"  Training time:  {metrics['training_time']:.2f}s")
        
        return metrics
