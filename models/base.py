"""
Base model interface for prediction models.

This module defines the abstract base class that all prediction models
must implement, ensuring consistent interface across different algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler


class BasePredictionModel(ABC):
    """
    Abstract base class for all prediction models.
    
    All models must implement:
    - train(): Train the model
    - predict(): Make predictions
    - predict_proba(): Get probability estimates
    - get_feature_importance(): Get feature importance (if applicable)
    
    Attributes:
        name: Model name
        model: Underlying sklearn/xgboost model
        scaler: Feature scaler (optional)
        feature_names: List of feature names
        is_trained: Whether model has been trained
    """
    
    def __init__(self, name: str, scale_features: bool = True):
        """
        Initialize base model.
        
        Args:
            name: Model name
            scale_features: Whether to scale features
        """
        self.name = name
        self.model = None
        self.scaler = StandardScaler() if scale_features else None
        self.feature_names = None
        self.is_trained = False
        self.scale_features = scale_features
        
    @abstractmethod
    def _create_model(self, **kwargs):
        """
        Create the underlying model instance.
        
        Must be implemented by subclasses.
        """
        pass
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              feature_names: list = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Feature names (optional)
            
        Returns:
            Dictionary with training metrics
        """
        print(f"\nTraining {self.name}...")
        
        self.feature_names = feature_names
        
        # Scale features if needed
        if self.scale_features and self.scaler is not None:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Train model (implemented by subclass)
        metrics = self._train_model(X_train_scaled, y_train, X_val_scaled, y_val)
        
        self.is_trained = True
        print(f"✓ {self.name} training complete")
        
        return metrics
    
    @abstractmethod
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Internal training method to be implemented by subclasses.
        
        Args:
            X_train: Scaled training features
            y_train: Training labels
            X_val: Scaled validation features
            y_val: Validation labels
            
        Returns:
            Dictionary with training metrics
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make class predictions.
        
        Args:
            X: Features
            
        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} has not been trained yet")
        
        # Scale if needed
        if self.scale_features and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates for each class.
        
        Args:
            X: Features
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} has not been trained yet")
        
        # Scale if needed
        if self.scale_features and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with features and importance scores
            Returns None if model doesn't support feature importance
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} has not been trained yet")
        
        # Check if model has feature_importances_ attribute
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            if self.feature_names is not None:
                df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                })
            else:
                df = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(importances))],
                    'importance': importances
                })
            
            df = df.sort_values('importance', ascending=False)
            return df
        
        # For linear models, use coefficient magnitudes
        elif hasattr(self.model, 'coef_'):
            # For multiclass, average absolute coefficients across classes
            coefs = np.abs(self.model.coef_).mean(axis=0)
            
            if self.feature_names is not None:
                df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': coefs
                })
            else:
                df = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(coefs))],
                    'importance': coefs
                })
            
            df = df.sort_values('importance', ascending=False)
            return df
        
        else:
            print(f"Warning: {self.name} does not support feature importance")
            return None
    
    def __repr__(self) -> str:
        """String representation."""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.name} ({status})"
