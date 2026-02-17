"""
High-Frequency Trading (HFT) tick data simulation module.

This module generates realistic market microstructure data including:
- Bid/Ask prices with realistic spreads
- Trade volumes with autocorrelation
- Order book depth
- Order flow imbalance patterns

Mathematical Model:
------------------
Mid-price follows a mean-reverting process with microstructure noise:
    mid_t = mid_{t-1} + drift + σ * ε_t + α * OFI_{t-1}
    
where:
    - σ: volatility parameter
    - ε_t: standard normal shock
    - α: OFI impact coefficient
    - OFI: Order Flow Imbalance (autocorrelated)

Spread dynamics:
    spread_t ~ TruncatedNormal(μ_spread, σ_spread, min, max)

Volume and depth:
    volume_t ~ LogNormal(μ_vol, σ_vol)
    depth_t ~ LogNormal(μ_depth, σ_depth)
"""

import numpy as np
import pandas as pd
from typing import Tuple
import time


class HFTDataSimulator:
    """
    Simulates high-frequency trading tick data with realistic microstructure.
    
    Attributes:
        config: SimulationConfig object with parameters
        rng: NumPy random number generator for reproducibility
    """
    
    def __init__(self, config):
        """
        Initialize the HFT data simulator.
        
        Args:
            config: SimulationConfig object containing simulation parameters
        """
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
        
    def simulate(self) -> pd.DataFrame:
        """
        Generate simulated HFT tick data.
        
        Returns:
            DataFrame with columns:
                - timestamp: Sequential tick timestamps
                - bid_price: Bid price
                - ask_price: Ask price
                - mid_price: Mid price (bid + ask) / 2
                - spread: Bid-ask spread
                - trade_volume: Trade volume
                - bid_depth: Depth at bid
                - ask_depth: Depth at ask
                - ofi: Order Flow Imbalance
        """
        start_time = time.time()
        print(f"Simulating {self.config.n_ticks} ticks of HFT data...")
        
        n = self.config.n_ticks
        
        # Initialize arrays
        mid_prices = np.zeros(n)
        spreads = np.zeros(n)
        volumes = np.zeros(n)
        bid_depths = np.zeros(n)
        ask_depths = np.zeros(n)
        ofis = np.zeros(n)
        
        # Initial values
        mid_prices[0] = self.config.initial_price
        spreads[0] = self.config.spread_mean
        volumes[0] = self.config.volume_mean
        bid_depths[0] = self.config.depth_mean
        ask_depths[0] = self.config.depth_mean
        ofis[0] = 0.0
        
        # Generate correlated shocks
        price_shocks = self.rng.normal(0, self.config.volatility, n)
        spread_shocks = self.rng.normal(0, self.config.spread_std, n)
        volume_shocks = self.rng.lognormal(0, 0.3, n)
        depth_shocks = self.rng.lognormal(0, 0.3, n)
        ofi_shocks = self.rng.normal(0, 1, n)
        
        # Simulate tick-by-tick
        for t in range(1, n):
            # Order Flow Imbalance with autocorrelation
            # OFI_t = ρ * OFI_{t-1} + ε_t
            ofis[t] = (self.config.ofi_autocorr * ofis[t-1] + 
                      ofi_shocks[t] * self.config.volume_mean * 0.1)
            
            # Mid-price with mean reversion and OFI impact
            # Price impact: positive OFI → price increase
            ofi_impact = 0.00001 * ofis[t]  # Small impact coefficient
            mean_reversion = -self.config.price_autocorr * (mid_prices[t-1] - self.config.initial_price)
            
            mid_prices[t] = (mid_prices[t-1] + 
                           self.config.drift +
                           price_shocks[t] +
                           ofi_impact +
                           mean_reversion)
            
            # Ensure price stays positive
            mid_prices[t] = max(mid_prices[t], self.config.initial_price * 0.5)
            
            # Spread dynamics (truncated normal)
            spreads[t] = (self.config.spread_mean + spread_shocks[t])
            spreads[t] = np.clip(spreads[t], self.config.min_spread, self.config.max_spread)
            
            # Round to tick size
            spreads[t] = np.round(spreads[t] / self.config.tick_size) * self.config.tick_size
            
            # Volume (log-normal with mean reversion)
            volumes[t] = self.config.volume_mean * volume_shocks[t]
            volumes[t] = max(volumes[t], self.config.min_volume)
            
            # Depths (log-normal)
            bid_depths[t] = self.config.depth_mean * depth_shocks[t]
            ask_depths[t] = self.config.depth_mean * depth_shocks[t] * self.rng.uniform(0.8, 1.2)
            
            bid_depths[t] = max(bid_depths[t], self.config.min_depth)
            ask_depths[t] = max(ask_depths[t], self.config.min_depth)
        
        # Round prices to tick size
        mid_prices = np.round(mid_prices / self.config.tick_size) * self.config.tick_size
        
        # Ensure spreads are at least one tick to avoid bid >= ask after rounding
        spreads = np.maximum(spreads, self.config.tick_size)
        
        # Calculate bid and ask from mid and spread
        bid_prices = mid_prices - spreads / 2
        ask_prices = mid_prices + spreads / 2
        
        # Round to tick size
        bid_prices = np.round(bid_prices / self.config.tick_size) * self.config.tick_size
        ask_prices = np.round(ask_prices / self.config.tick_size) * self.config.tick_size
        
        # Final check: ensure bid < ask (fix any rounding issues)
        for i in range(len(bid_prices)):
            if bid_prices[i] >= ask_prices[i]:
                # Force ask to be at least one tick above bid
                ask_prices[i] = bid_prices[i] + self.config.tick_size
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': np.arange(n),
            'bid_price': bid_prices,
            'ask_price': ask_prices,
            'mid_price': mid_prices,
            'spread': spreads,
            'trade_volume': volumes,
            'bid_depth': bid_depths,
            'ask_depth': ask_depths,
            'ofi': ofis
        })
        
        elapsed = time.time() - start_time
        print(f"✓ Simulation complete in {elapsed:.2f}s")
        print(f"  Price range: ${df['mid_price'].min():.2f} - ${df['mid_price'].max():.2f}")
        print(f"  Mean spread: ${df['spread'].mean():.4f}")
        print(f"  Mean volume: {df['trade_volume'].mean():.0f}")
        
        return df
    
    def add_realistic_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add realistic market microstructure patterns to simulated data.
        
        This includes:
        - Spread widening during high volatility
        - Volume clustering
        - Depth changes around price movements
        
        Args:
            df: DataFrame with basic simulated data
            
        Returns:
            Enhanced DataFrame with realistic patterns
        """
        # Calculate rolling volatility
        df['rolling_vol'] = df['mid_price'].pct_change().rolling(20).std()
        
        # Widen spreads during high volatility
        high_vol_mask = df['rolling_vol'] > df['rolling_vol'].quantile(0.75)
        df.loc[high_vol_mask, 'spread'] *= 1.5
        
        # Ensure spread constraints
        df['spread'] = np.clip(df['spread'], self.config.min_spread, self.config.max_spread)
        
        # Ensure spreads are at least one tick
        df['spread'] = np.maximum(df['spread'], self.config.tick_size)
        
        # Recalculate bid/ask
        df['bid_price'] = df['mid_price'] - df['spread'] / 2
        df['ask_price'] = df['mid_price'] + df['spread'] / 2
        
        # Round to tick size
        df['bid_price'] = np.round(df['bid_price'] / self.config.tick_size) * self.config.tick_size
        df['ask_price'] = np.round(df['ask_price'] / self.config.tick_size) * self.config.tick_size
        
        # Ensure bid < ask after rounding
        mask = df['bid_price'] >= df['ask_price']
        df.loc[mask, 'ask_price'] = df.loc[mask, 'bid_price'] + self.config.tick_size
        
        # Drop temporary column
        df.drop('rolling_vol', axis=1, inplace=True)
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate simulated data for consistency.
        
        Checks:
        - No missing values
        - Bid < Ask always
        - Positive volumes and depths
        - Reasonable price ranges
        
        Args:
            df: Simulated data DataFrame
            
        Returns:
            True if validation passes, False otherwise
        """
        checks = []
        
        # No missing values
        if df.isnull().any().any():
            print("✗ Validation failed: Missing values detected")
            checks.append(False)
        else:
            checks.append(True)
        
        # Bid < Ask
        if not (df['bid_price'] < df['ask_price']).all():
            print("✗ Validation failed: Bid >= Ask detected")
            checks.append(False)
        else:
            checks.append(True)
        
        # Positive volumes
        if not (df['trade_volume'] > 0).all():
            print("✗ Validation failed: Non-positive volumes detected")
            checks.append(False)
        else:
            checks.append(True)
        
        # Positive depths
        if not ((df['bid_depth'] > 0).all() and (df['ask_depth'] > 0).all()):
            print("✗ Validation failed: Non-positive depths detected")
            checks.append(False)
        else:
            checks.append(True)
        
        # Reasonable prices
        if df['mid_price'].min() < 0 or df['mid_price'].max() > self.config.initial_price * 2:
            print("✗ Validation failed: Unreasonable price range")
            checks.append(False)
        else:
            checks.append(True)
        
        if all(checks):
            print("✓ Data validation passed")
            return True
        else:
            return False


def generate_hft_data(config) -> pd.DataFrame:
    """
    Convenience function to generate and validate HFT data.
    
    Args:
        config: SimulationConfig object
        
    Returns:
        Validated DataFrame with HFT tick data
    """
    simulator = HFTDataSimulator(config)
    df = simulator.simulate()
    df = simulator.add_realistic_patterns(df)
    
    if simulator.validate_data(df):
        return df
    else:
        raise ValueError("Data validation failed")
