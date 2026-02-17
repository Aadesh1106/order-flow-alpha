"""
Microstructure feature engineering module.

This module computes market microstructure features from tick data:
1. Order Flow Imbalance (OFI) - primary signal
2. Bid-Ask Spread - liquidity measure
3. Mid-Price Return - momentum
4. Volume Imbalance - buy/sell pressure
5. Rolling Volatility - risk measure
6. Market Pressure - aggressive order flow
7. Depth Imbalance - order book pressure
8. Lagged features - temporal dependencies

Mathematical Formulas:
---------------------
1. OFI (Order Flow Imbalance):
   OFI_t = (BidDepth_t × ΔBid_t) - (AskDepth_t × ΔAsk_t)
   
   Interpretation: Positive OFI → buying pressure, Negative OFI → selling pressure

2. Mid-Price Return:
   r_t = (MidPrice_t - MidPrice_{t-1}) / MidPrice_{t-1}

3. Volume Imbalance:
   VI_t = (BidDepth_t - AskDepth_t) / (BidDepth_t + AskDepth_t)
   
   Range: [-1, 1], where 1 = all bids, -1 = all asks

4. Rolling Volatility:
   σ_t = std(r_{t-w:t})  where w = window size

5. Market Pressure:
   MP_t = sign(ΔMidPrice_t) × Volume_t
   
   Captures aggressive buying/selling

6. Depth Imbalance:
   DI_t = (BidDepth_t - AskDepth_t) / (BidDepth_t + AskDepth_t)
"""

import numpy as np
import pandas as pd
from typing import Tuple
import time


class MicrostructureFeatures:
    """
    Computes market microstructure features from tick data.
    
    Attributes:
        config: FeatureConfig object with parameters
    """
    
    def __init__(self, config):
        """
        Initialize feature engineering module.
        
        Args:
            config: FeatureConfig object
        """
        self.config = config
        
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all microstructure features.
        
        Args:
            df: DataFrame with tick data (from simulation)
            
        Returns:
            DataFrame with original data + computed features
        """
        start_time = time.time()
        print("Computing microstructure features...")
        
        # Make a copy to avoid modifying original
        data = df.copy()
        
        # 1. Order Flow Imbalance (already in simulated data, but recalculate properly)
        data['ofi_computed'] = self._compute_ofi(data)
        
        # 2. Bid-Ask Spread (already in data)
        data['spread_bps'] = (data['spread'] / data['mid_price']) * 10000  # Basis points
        
        # 3. Mid-Price Return
        data['mid_return'] = data['mid_price'].pct_change()
        
        # 4. Volume Imbalance
        data['volume_imbalance'] = self._compute_volume_imbalance(data)
        
        # 5. Rolling Volatility
        data['rolling_volatility'] = self._compute_rolling_volatility(data)
        
        # 6. Market Pressure
        data['market_pressure'] = self._compute_market_pressure(data)
        
        # 7. Depth Imbalance
        data['depth_imbalance'] = self._compute_depth_imbalance(data)
        
        # 8. Price momentum
        data['price_momentum'] = data['mid_price'].diff()
        
        # 9. Spread changes
        data['spread_change'] = data['spread'].diff()
        
        # 10. Volume changes
        data['volume_change'] = data['trade_volume'].pct_change()
        
        # 11. Lagged features
        data = self._add_lagged_features(data)
        
        # 12. Rolling aggregates
        data = self._add_rolling_features(data)
        
        # Drop NaN rows created by rolling windows and lags
        initial_rows = len(data)
        data.dropna(inplace=True)
        dropped_rows = initial_rows - len(data)
        
        elapsed = time.time() - start_time
        print(f"✓ Feature computation complete in {elapsed:.2f}s")
        print(f"  Features created: {len([c for c in data.columns if c not in df.columns])}")
        print(f"  Rows dropped (NaN): {dropped_rows}")
        print(f"  Final dataset size: {len(data)} rows")
        
        return data
    
    def _compute_ofi(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Order Flow Imbalance.
        
        OFI_t = (BidDepth_t × ΔBid_t) - (AskDepth_t × ΔAsk_t)
        
        Args:
            df: DataFrame with bid/ask prices and depths
            
        Returns:
            Series with OFI values
        """
        bid_change = df['bid_price'].diff()
        ask_change = df['ask_price'].diff()
        
        ofi = (df['bid_depth'] * bid_change) - (df['ask_depth'] * ask_change)
        
        return ofi
    
    def _compute_volume_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute volume imbalance between bid and ask.
        
        VI_t = (BidDepth_t - AskDepth_t) / (BidDepth_t + AskDepth_t)
        
        Args:
            df: DataFrame with depths
            
        Returns:
            Series with volume imbalance in [-1, 1]
        """
        total_depth = df['bid_depth'] + df['ask_depth']
        # Avoid division by zero
        total_depth = total_depth.replace(0, np.nan)
        
        vi = (df['bid_depth'] - df['ask_depth']) / total_depth
        
        return vi
    
    def _compute_rolling_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute rolling volatility of mid-price returns.
        
        σ_t = std(r_{t-w:t})
        
        Args:
            df: DataFrame with mid_return
            
        Returns:
            Series with rolling volatility
        """
        vol = df['mid_return'].rolling(
            window=self.config.volatility_window,
            min_periods=self.config.volatility_window // 2
        ).std()
        
        return vol
    
    def _compute_market_pressure(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute market pressure indicator.
        
        MP_t = sign(ΔMidPrice_t) × Volume_t
        
        Positive pressure = aggressive buying
        Negative pressure = aggressive selling
        
        Args:
            df: DataFrame with price and volume
            
        Returns:
            Series with market pressure
        """
        price_change = df['mid_price'].diff()
        sign = np.sign(price_change)
        
        pressure = sign * df['trade_volume']
        
        return pressure
    
    def _compute_depth_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute depth imbalance (same as volume imbalance in this context).
        
        DI_t = (BidDepth_t - AskDepth_t) / (BidDepth_t + AskDepth_t)
        
        Args:
            df: DataFrame with depths
            
        Returns:
            Series with depth imbalance
        """
        return self._compute_volume_imbalance(df)
    
    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lagged versions of key features.
        
        This captures temporal dependencies and autocorrelation.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with lagged features added
        """
        # Features to lag
        lag_features = [
            'ofi_computed',
            'mid_return',
            'volume_imbalance',
            'market_pressure',
            'rolling_volatility'
        ]
        
        for feature in lag_features:
            if feature in df.columns:
                for lag in range(1, self.config.n_lags + 1):
                    df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling aggregate features.
        
        These capture recent trends and patterns.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with rolling features added
        """
        # Rolling mean of OFI
        df['ofi_rolling_mean'] = df['ofi_computed'].rolling(
            window=self.config.volume_window
        ).mean()
        
        # Rolling std of OFI
        df['ofi_rolling_std'] = df['ofi_computed'].rolling(
            window=self.config.volume_window
        ).std()
        
        # Rolling mean of volume
        df['volume_rolling_mean'] = df['trade_volume'].rolling(
            window=self.config.volume_window
        ).mean()
        
        # Rolling mean of spread
        df['spread_rolling_mean'] = df['spread'].rolling(
            window=self.config.pressure_window
        ).mean()
        
        # Rolling mean of market pressure
        df['pressure_rolling_mean'] = df['market_pressure'].rolling(
            window=self.config.pressure_window
        ).mean()
        
        return df
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create prediction target: next price movement.
        
        Target classes:
        - 0: DOWN (price decrease > threshold)
        - 1: NEUTRAL (small price change)
        - 2: UP (price increase > threshold)
        
        Args:
            df: DataFrame with mid_price
            
        Returns:
            DataFrame with 'target' column added
        """
        print("Creating prediction target...")
        
        # Future price change (next tick)
        df['future_return'] = df['mid_price'].pct_change().shift(-1)
        
        # Classify into UP/DOWN/NEUTRAL
        threshold = self.config.price_change_threshold
        
        conditions = [
            df['future_return'] < -threshold,  # DOWN
            (df['future_return'] >= -threshold) & (df['future_return'] <= threshold),  # NEUTRAL
            df['future_return'] > threshold  # UP
        ]
        
        choices = [0, 1, 2]  # DOWN, NEUTRAL, UP
        
        df['target'] = np.select(conditions, choices, default=1)
        
        # Drop the last row (no future return)
        df = df[:-1].copy()
        
        # Print class distribution
        class_counts = df['target'].value_counts().sort_index()
        print(f"  Target distribution:")
        print(f"    DOWN (0):    {class_counts.get(0, 0):6d} ({class_counts.get(0, 0)/len(df)*100:5.1f}%)")
        print(f"    NEUTRAL (1): {class_counts.get(1, 0):6d} ({class_counts.get(1, 0)/len(df)*100:5.1f}%)")
        print(f"    UP (2):      {class_counts.get(2, 0):6d} ({class_counts.get(2, 0)/len(df)*100:5.1f}%)")
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> list:
        """
        Get list of feature column names (excluding metadata and target).
        
        Args:
            df: DataFrame with all columns
            
        Returns:
            List of feature column names
        """
        # Columns to exclude
        exclude = [
            'timestamp', 'bid_price', 'ask_price', 'mid_price',
            'spread', 'trade_volume', 'bid_depth', 'ask_depth',
            'ofi', 'target', 'future_return'
        ]
        
        features = [col for col in df.columns if col not in exclude]
        
        return features


def engineer_features(df: pd.DataFrame, config) -> Tuple[pd.DataFrame, list]:
    """
    Convenience function to engineer all features and create target.
    
    Args:
        df: Raw tick data DataFrame
        config: FeatureConfig object
        
    Returns:
        Tuple of (DataFrame with features and target, list of feature names)
    """
    feature_eng = MicrostructureFeatures(config)
    
    # Compute features
    df_features = feature_eng.compute_all_features(df)
    
    # Create target
    df_features = feature_eng.create_target(df_features)
    
    # Get feature names
    feature_names = feature_eng.get_feature_names(df_features)
    
    print(f"✓ Feature engineering complete: {len(feature_names)} features")
    
    return df_features, feature_names
