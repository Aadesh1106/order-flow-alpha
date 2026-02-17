# Quick Start Guide
## Order Flow Imbalance & Short-Term Price Prediction

### Installation

1. **Install Python dependencies**:
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn scipy
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Running the System

**Single command to run everything**:
```bash
python main.py
```

This will execute the complete pipeline:
1. ✅ Simulate 10,000 ticks of HFT data (~2 seconds)
2. ✅ Engineer 30+ microstructure features (~1 second)
3. ✅ Train 3 ML models (Logistic, RF, XGBoost) (~5-10 seconds)
4. ✅ Backtest trading strategies (~2 seconds)
5. ✅ Generate visualizations and reports (~5 seconds)

**Total execution time**: ~30-60 seconds

### Output

All results are saved in the `results/` directory:

**Visualizations** (18 PNG files):
- `confusion_matrix_*.png` - Classification performance
- `roc_curves.png` - ROC curves for all models
- `feature_importance_*.png` - Top features by model
- `equity_curve_*.png` - Portfolio value over time
- `returns_distribution_*.png` - Return distributions
- `model_comparison.png` - Side-by-side comparison

**Data Files**:
- `model_comparison.csv` - Performance metrics table
- `all_results.json` - Complete results (927 KB)

### Key Results

**Best Model**: XGBoost
- **Accuracy**: 95.79%
- **Total Return**: -0.01%
- **Sharpe Ratio**: -109.87
- **Max Drawdown**: -0.01%
- **Total Trades**: 3

**Why negative returns despite high accuracy?**
- Transaction costs (spread + commission)
- Execution latency (2 ticks)
- Conservative entry threshold (55% confidence)
- Very few trades (high selectivity)

### Configuration

Edit `utils/config.py` to customize:

**Data Simulation**:
```python
n_ticks = 10000          # Number of ticks to simulate
initial_price = 100.0    # Starting price
volatility = 0.002       # Per-tick volatility
```

**Feature Engineering**:
```python
volatility_window = 20   # Rolling volatility window
n_lags = 3              # Number of lagged features
```

**Models**:
```python
# Random Forest
rf_params = {
    'n_estimators': 100,
    'max_depth': 10,
    ...
}

# XGBoost
xgb_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    ...
}
```

**Trading Strategy**:
```python
entry_threshold = 0.55      # Minimum probability to trade
execution_latency = 2       # Delay in ticks
commission_per_share = 0.001  # Transaction cost
```

**Backtesting**:
```python
initial_capital = 100000.0  # Starting capital
risk_free_rate = 0.02       # Annual risk-free rate
```

### Understanding the Output

**Console Output**:
```
============================================================
ORDER FLOW IMBALANCE & SHORT-TERM PRICE PREDICTION
============================================================

[1/7] Generating HFT tick data...
✓ Simulation complete in 1.23s
  Price range: $99.50 - $100.45
  Mean spread: $0.0200
  Mean volume: 1000

[2/7] Engineering microstructure features...
✓ Feature computation complete in 0.89s
  Features created: 32
  Final dataset size: 9950 rows

[3/7] Splitting data...
  Train: 6965 samples (70%)
  Val:   1492 samples (15%)
  Test:  1493 samples (15%)

[4/7] Training models...
Training Logistic Regression...
  Train accuracy: 0.5421
  Val accuracy:   0.5434
  Training time:  0.45s

Training Random Forest...
  Train accuracy: 0.9998
  Val accuracy:   0.8865
  Training time:  2.34s

Training XGBoost...
  Train accuracy: 0.9876
  Val accuracy:   0.9579
  Training time:  1.23s

[5/7] Evaluating models...
[6/7] Running trading strategies...
[7/7] Creating visualizations and reports...

FINAL SUMMARY
============================================================
Best Model (by Sharpe Ratio): XGBoost

✓ Pipeline complete in 28.45s
✓ Results saved to 'results/' directory
```

### Project Structure

```
ORDER FLOW IMBALANCE & SHORT-TERM PRICE PREDICTION/
│
├── README.md                  # Theory, assumptions, limitations
├── PROJECT_SUMMARY.md         # Execution summary and results
├── QUICKSTART.md             # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── main.py                  # Entry point
│
├── data/
│   ├── __init__.py
│   └── simulation.py        # HFT tick data simulator
│
├── features/
│   ├── __init__.py
│   └── microstructure.py    # Feature engineering
│
├── models/
│   ├── __init__.py
│   ├── base.py             # Base model interface
│   ├── logistic.py         # Logistic Regression
│   ├── random_forest.py    # Random Forest
│   └── xgboost_model.py    # XGBoost
│
├── strategy/
│   ├── __init__.py
│   └── trading_strategy.py # Trading logic
│
├── backtesting/
│   ├── __init__.py
│   └── backtester.py       # Backtesting engine
│
├── risk/
│   ├── __init__.py
│   └── metrics.py          # Risk & performance metrics
│
├── evaluation/
│   ├── __init__.py
│   └── visualizer.py       # Visualization & analysis
│
├── utils/
│   ├── __init__.py
│   └── config.py           # Configuration
│
└── results/                # Generated outputs
    ├── *.png              # Visualizations
    ├── *.csv              # Data tables
    └── *.json             # Complete results
```

### Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'xgboost'`
**Solution**: 
```bash
pip install xgboost
```

**Issue**: Plots not displaying
**Solution**: Plots are saved to `results/` directory. Open PNG files directly.

**Issue**: Out of memory
**Solution**: Reduce `n_ticks` in `utils/config.py`:
```python
n_ticks = 5000  # Instead of 10000
```

**Issue**: Slow execution
**Solution**: Reduce model complexity in `utils/config.py`:
```python
rf_params = {'n_estimators': 50, ...}  # Instead of 100
xgb_params = {'n_estimators': 50, ...}  # Instead of 100
```

### Next Steps

1. **Explore Results**:
   - Open `results/` folder
   - View confusion matrices, ROC curves, equity curves
   - Analyze feature importance

2. **Experiment with Parameters**:
   - Adjust entry threshold (higher = fewer trades)
   - Change latency (lower = better execution)
   - Modify feature windows

3. **Try Different Strategies**:
   - Edit `strategy/trading_strategy.py`
   - Implement stop-loss logic
   - Add position sizing rules

4. **Use Real Data**:
   - Replace `data/simulation.py` with real data loader
   - Ensure same column format (bid, ask, volume, depth)

5. **Add More Models**:
   - Create new model class inheriting from `models/base.py`
   - Implement `_train_model()` method
   - Add to `main.py`

### Support

For questions or issues:
1. Check `README.md` for theory and assumptions
2. Review `PROJECT_SUMMARY.md` for results analysis
3. Examine code comments for implementation details

### License

This is a research/educational project demonstrating production-level quantitative trading system design.

---

**Happy Trading! 📈**
