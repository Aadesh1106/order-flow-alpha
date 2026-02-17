"""
Random Forest model for price movement prediction.

Random Forest is an ensemble of decision trees that combines multiple
weak learners to create a strong predictor. Each tree is trained on
a bootstrap sample with random feature subsets.

Mathematical Model:
------------------
For classification:
    P(y = k | x) = (1/T) Σ_t I(tree_t(x) = k)

where:
    - T: number of trees
    - tree_t(x): prediction of tree t
    - I(): indicator function

Advantages:
- Handles non-linear relationships
- Robust to outliers
- Built-in feature importance
- Reduces overfitting via averaging
- Handles feature interactions

Limitations:
- Can be slow for large datasets
- Less interpretable than linear models
- May overfit if trees are too deep
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any
import time

from models.base import BasePredictionModel


class RandomForestModel(BasePredictionModel):
    """
    Random Forest model for multiclass classification.
    
    Inherits from BasePredictionModel and implements training logic.
    """
    
    def __init__(self, params: Dict[str, Any] = None, scale_features: bool = False):
        """
        Initialize Random Forest model.
        
        Args:
            params: Model hyperparameters (sklearn RandomForestClassifier params)
            scale_features: Whether to scale features (not needed for RF)
        """
        super().__init__(name="Random Forest", scale_features=scale_features)
        self.params = params or {}
        self._create_model()
    
    def _create_model(self, **kwargs):
        """Create sklearn RandomForestClassifier instance."""
        self.model = RandomForestClassifier(**self.params)
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
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
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth
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
        print(f"  Trees:          {metrics['n_estimators']}")
        
        return metrics
