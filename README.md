A stock analysis platform powered by machine learning and technical indicators. Features real-time stock data analysis, AI-driven trend insights, and comprehensive technical analysis dashboards.

## Features

- **AI Analysis**: Advanced transformer models for market trend analysis
- **Technical Indicators**: 20+ professional technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Interactive Charts**: Real-time price charts with volume analysis
- **Risk Assessment**: Volatility analysis and risk metrics
- **Multi-Stock Support**: Analyze multiple stocks simultaneously

## Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ManuelP06/stock_analyzer.git
cd stock_analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## Usage Guide

### Basic Analysis

2. **Select Stocks**: Choose from popular stocks or enter custom symbols (e.g., TSLA,AAPL,MSFT)
3. **Configure Parameters**:
   - Training Period: 6 months to 10 years
   - Data Interval: 1 minute to 1 month
   - Epochs: 10-100 (higher = more training)
4. **Start Analysis**: Click "START ANALYSIS"


#### Command Line Training
```bash
python stock_trader.py train --symbols AAPL MSFT GOOGL --epochs 30 --period 2y
```

#### Technical Indicators Only
```bash
python src/features/technical_indicators.py -i data.csv --indicators rsi macd bollinger
```

#### Custom Analysis
```bash
python stock_trader.py both --symbols TSLA --seq_len 120 --pred_len 10
```

## Technical Indicators Included

- **Trend**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, Stochastic, Williams %R, CCI
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, MFI, VWAP
- **Support/Resistance**: Donchian Channels

## Configuration Options

### Training Parameters
- `--epochs`: Number of training epochs (default: 25)
- `--seq_len`: Sequence length for ML model (default: 60)
- `--pred_len`: Prediction horizon days (default: 7)
- `--period`: Training data period (6mo, 1y, 2y, 3y, 5y, 10y)
- `--interval`: Data granularity (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)


## Project Structure

```
stock_analyzer/
├── app.py                          # Main Streamlit dashboard
├── stock_trader.py                 # Core ML training and analysis
├── src/
│   ├── data/
│   │   ├── downloader.py          # Stock data fetching
│   │   └── preprocessor.py        # Data preprocessing
│   ├── features/
│   │   └── technical_indicators.py # Technical analysis
│   └── models/
│       ├── trainer.py             # ML model training
│       └── patchtst.py           # Transformer architecture
└── requirements.txt               # Dependencies
```

## Dependencies

```txt
streamlit>=1.28.0
polars>=0.19.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.15.0
yfinance>=0.2.18
scikit-learn>=1.3.0
```

## Important Disclaimers

**Educational Purpose Only**

This platform is designed for:
- Educational demonstration of ML in financial analysis
- Stock research and technical analysis learning
- Understanding market patterns and indicators

**Limitations:**
- ML predictions have inherent uncertainty
- Past performance doesn't guarantee future results
- Markets are influenced by unpredictable events
- Not suitable for automated trading

**Not Investment Advice:**
- Always consult qualified financial professionals
- Conduct thorough due diligence before investing
- Never invest more than you can afford to lose
- Use appropriate risk management

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the (LICENSE) file for details.

