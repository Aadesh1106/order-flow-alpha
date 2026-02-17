# Order Flow Imbalance & Short-Term Price Prediction

## Overview

This project implements a market microstructure-based short-term price prediction system using Order Flow Imbalance (OFI) and other high-frequency trading features. The system predicts next price movements and backtests a trading strategy with realistic market conditions.

## Theory

### Market Microstructure
Market microstructure studies how trading mechanisms affect price formation. Key concepts:

1. **Order Flow Imbalance (OFI)**: Measures the imbalance between buy and sell orders
   - Formula: `OFI = (BidVolume × BidPriceChange) - (AskVolume × AskPriceChange)`
   - Positive OFI suggests buying pressure → price increase
   - Negative OFI suggests selling pressure → price decrease

2. **Bid-Ask Spread**: Liquidity indicator
   - Wider spreads indicate lower liquidity and higher transaction costs
   - Spread = Ask - Bid

3. **Mid-Price**: Fair value estimate
   - Mid = (Bid + Ask) / 2

4. **Depth Imbalance**: Order book pressure
   - Imbalance = (BidDepth - AskDepth) / (BidDepth + AskDepth)

5. **Market Pressure**: Combined volume and price momentum
   - Captures aggressive buying/selling behavior

### Prediction Approach

We frame this as a **classification problem**:
- **Target**: Next price movement (UP, DOWN, NEUTRAL)
- **Features**: Microstructure indicators computed from tick data
- **Models**: Logistic Regression, Random Forest, XGBoost
- **Horizon**: Short-term (next few ticks)

### Statistical Considerations

1. **Non-stationarity**: Financial time series are non-stationary
   - Use returns instead of prices
   - Rolling windows for feature computation

2. **Autocorrelation**: Microstructure features exhibit autocorrelation
   - Use lagged features
   - Be aware of look-ahead bias

3. **Overfitting**: High-frequency data has noise
   - Cross-validation with time-based splits
   - Regularization in models
   - Feature importance analysis

## Architecture

```
order_flow_prediction/
├── data/
│   └── simulation.py          # HFT tick data simulator
├── features/
│   └── microstructure.py      # Feature engineering
├── models/
│   ├── base.py                # Base model interface
│   ├── logistic.py            # Logistic Regression
│   ├── random_forest.py       # Random Forest
│   └── xgboost_model.py       # XGBoost
├── strategy/
│   └── trading_strategy.py    # Trading logic
├── backtesting/
│   └── backtester.py          # Backtesting engine
├── risk/
│   └── metrics.py             # Risk & performance metrics
├── evaluation/
│   └── visualizer.py          # Plots & analysis
├── utils/
│   └── config.py              # Configuration
└── main.py                    # Entry point
```

## Features Computed

1. **Order Flow Imbalance (OFI)**: Primary signal
2. **Bid-Ask Spread**: Liquidity measure
3. **Mid-Price Return**: Price momentum
4. **Volume Imbalance**: Buy vs sell volume
5. **Rolling Volatility**: Risk measure
6. **Market Pressure**: Aggressive order flow
7. **Depth Imbalance**: Order book pressure
8. **Lagged Features**: Temporal dependencies

## Models

### 1. Logistic Regression
- **Pros**: Interpretable, fast, probabilistic output
- **Cons**: Linear decision boundary, limited complexity
- **Use**: Baseline model

### 2. Random Forest
- **Pros**: Non-linear, handles interactions, robust
- **Cons**: Can overfit, less interpretable
- **Use**: Strong ensemble baseline

### 3. XGBoost
- **Pros**: State-of-the-art gradient boosting, regularization
- **Cons**: Hyperparameter tuning required
- **Use**: Best performance expected

## Trading Strategy

**Logic**:
1. Predict next price movement with probability
2. Trade when `P(UP) > threshold` or `P(DOWN) > threshold`
3. Position sizing based on confidence
4. Include realistic latency (1-5 ticks)
5. Include transaction costs (spread + commission)

**Risk Management**:
- Maximum position size
- Stop-loss mechanisms
- Exposure limits

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision/Recall**: Class-specific performance
- **ROC-AUC**: Discrimination ability
- **Confusion Matrix**: Error analysis

### Trading Metrics
- **PnL**: Profit and Loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst loss period
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

## Assumptions

1. **Market Simulation**:
   - Tick data follows realistic microstructure patterns
   - Order flow has autocorrelation
   - Spreads and depths are realistic

2. **Trading**:
   - Can execute at quoted prices (within latency)
   - Slippage modeled via latency
   - No market impact (small size assumption)

3. **Features**:
   - Features are computed without look-ahead bias
   - Rolling windows capture temporal dependencies

## Limitations

1. **Simulated Data**: Real market data has more complexity
   - Regime changes
   - News events
   - Microstructure noise

2. **Transaction Costs**: Simplified model
   - Real costs vary by venue
   - Market impact for larger sizes

3. **Latency**: Fixed latency assumption
   - Real latency is stochastic
   - Queue position matters

4. **No Adverse Selection**: 
   - Informed traders may pick off stale quotes
   - Real HFT faces adverse selection risk

5. **Single Asset**: 
   - No cross-asset effects
   - No portfolio considerations

## Future Improvements

1. **Data**:
   - Use real tick data (e.g., LOBSTER, NASDAQ ITCH)
   - Multi-asset simulation
   - Incorporate news/events

2. **Features**:
   - Deep order book features (beyond top-of-book)
   - Trade flow toxicity measures
   - Cross-asset correlations

3. **Models**:
   - Deep learning (LSTM, Transformers)
   - Online learning for adaptation
   - Ensemble methods

4. **Strategy**:
   - Dynamic position sizing
   - Multi-timeframe signals
   - Market-making strategies

5. **Risk**:
   - Real-time risk monitoring
   - Stress testing
   - Regime detection

## Installation

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

## Usage

```bash
python main.py
```

This will:
1. Simulate HFT tick data
2. Compute microstructure features
3. Train all models
4. Evaluate predictions
5. Backtest trading strategy
6. Generate visualizations

## Results

Results are saved in `results/`:
- `confusion_matrix.png`: Classification performance
- `roc_curve.png`: ROC curves for all models
- `equity_curve.png`: Strategy PnL over time
- `feature_importance.png`: Top features
- `metrics.json`: All performance metrics

## Mathematical Documentation

See inline comments in code for detailed formulas and statistical reasoning.

## Performance Profiling

Key operations are timed and logged. Check console output for:
- Data simulation time
- Feature computation time
- Model training time
- Backtesting time

## Author

Built with production-level standards for quantitative trading research.
