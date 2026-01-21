# Crypto-Trading-AI 🤖

**Autonomous AI-Powered Cryptocurrency Trading Agent** using Machine Learning ensemble models, multi-timeframe technical analysis, and intelligent risk management.

---

## 🌟 Overview

This project implements a **fully autonomous trading agent** that:
- 📊 Collects real-time market data from cryptocurrency exchanges
- 🧠 Trains ensemble ML models (CatBoost + XGBoost) with intelligent feature selection
- 🎯 Generates trading predictions with quality scoring and confidence levels
- 🔍 Analyzes signals across multiple timeframes (5m, 15m, 1h, 4h, 1d)
- ✅ Filters low-quality trades automatically (quality threshold: 50/100)
- 📈 Provides detailed risk management (position sizing, stop loss, take profit)

**Key Innovation:** The agent doesn't just predict prices—it **evaluates the quality of its own predictions** and only recommends trades when confidence is high.

---

## ✨ Key Features

### 🤖 Autonomous Agent
- **Smart Consensus Analysis**: Combines ML predictions + technical indicators
- **Quality Filtering**: Rejects trades below quality threshold (prevents over-trading)
- **Market Regime Detection**: Adapts strategy to trending vs ranging markets
- **Risk Management**: Automatic position sizing and stop-loss calculation

### 🧠 Machine Learning
- **Ensemble Models**: CatBoost + XGBoost combination
- **Intelligent Feature Selection**: Top 50 features via Mutual Information (32% faster training)
- **GPU-Ready**: Infrastructure supports GPU acceleration
- **Multi-Symbol Support**: BTC, ETH, BNB, SOL, ADA across 5 timeframes

### 📊 Technical Analysis
- **90+ Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R, OBV, Ichimoku, ADX
- **Multi-Timeframe Consensus**: Analyzes 5m, 15m, 1h, 4h, 1d simultaneously
- **Custom Indicators**: Market regime, volatility clustering, momentum strength

### 🔄 Automation
- **24/7 Operation**: Scheduled data collection and analysis
- **Alert System**: Email, desktop, and log-based notifications
- **Performance Tracking**: Backtesting and live performance metrics

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (optional - works without API keys for public data)
nano .env
```

### 3. Run the Agent

**Option A: Use Control Center (Recommended)**
```bash
python control_center.py
```

**Option B: Manual 4-Step Workflow**
```bash
# Step 1: Collect market data
python collect_data.py

# Step 2: Train ML models
python train_models.py

# Step 3: Generate predictions
python generate_predictions.py

# Step 4: Run the agent
python run_agent.py
```

---

## 📁 Project Structure

```
Crypto-Trading-AI/
├── 🎛️  Core Scripts (6 files)
│   ├── collect_data.py          # Data collection from exchanges
│   ├── train_models.py          # ML model training (CatBoost + XGBoost)
│   ├── generate_predictions.py  # Generate predictions from trained models
│   ├── analyze_signals.py       # Technical analysis + signal generation
│   ├── run_agent.py            # Run the autonomous trading agent
│   └── control_center.py       # Unified control interface (CLI)
│
├── 🤖 Agent Logic
│   └── crypto_agent/
│       ├── agent.py            # Main agent orchestration
│       ├── tools.py            # Agent tools (analysis, predictions, etc.)
│       ├── database.py         # Database operations
│       ├── config.py           # Agent configuration
│       └── prompts.py          # LLM prompts (if using AI reasoning)
│
├── 🧠 ML & Analysis
│   └── crypto_ai/
│       ├── features/           # Feature engineering
│       ├── database/           # Database schemas
│       ├── automation/         # Scheduling & automation
│       └── gpu_utils.py        # GPU acceleration utilities
│
├── 📊 Data & Models
│   ├── data/
│   │   ├── ml_crypto_data.db           # Main database (market data, predictions)
│   │   └── backtest_baseline_final.db  # Backtest results
│   ├── ml_models/              # Trained model files (.pkl)
│   └── ml_predictions/         # Prediction outputs
│
├── 📝 Reports & Logs
│   ├── ml_reports/             # Validation reports, benchmarks
│   └── logs/                   # Application logs
│
├── 🧪 Testing & Validation
│   ├── tests/                  # Unit tests
│   ├── presentation_validation_tests.py  # Comprehensive validation
│   ├── profitability_analysis.py        # Profitability analysis
│   └── agent_vs_random_backtest.py      # Agent vs random baseline
│
├── ⚙️  Configuration
│   ├── .env                    # Environment variables (API keys, etc.)
│   ├── automation_config.json  # Main configuration
│   └── automation_config_ml.json  # ML-specific settings
│
└── 📚 Documentation
    ├── README.md               # This file
    ├── FILE_RENAMING.md        # File renaming history
    ├── project_report.md       # Project status report
    └── ml_reports/final_presentation_summary.md  # Latest validation results
```

---

## 🎯 Current Performance

### ✅ Validation Results (Jan 20, 2026)

**Data Collection:**
- ✅ 139,049 records across 5 symbols × 5 timeframes
- ✅ Real-time data from Binance API

**Model Training:**
- ✅ 4 ensemble models trained (BTC/ETH on 1h/4h)
- ✅ Direction accuracy: 50-54% (baseline: 50%)
- ✅ Feature selection: 32% faster training

**Agent Performance:**
- ✅ Analyzed 25 symbol/timeframe combinations
- ✅ Identified 1 high-quality signal: **BNB/USDT 4h STRONG_BUY**
  - Quality Score: 74/100 (Grade: B)
  - Confidence: 82.4%
  - Consensus: 100% weighted agreement

**Key Insight:** The agent successfully **filters out low-quality trades**, analyzing 25 combinations but only recommending 1 high-confidence opportunity. This prevents over-trading and protects capital.

### ⚠️ Known Issues

**Overfitting Challenge:**
- Train Accuracy: 100%
- Test Accuracy: 52%
- **Gap: 48%** (target: <10%)

**Mitigation in Progress:**
- ✅ Feature selection (implemented)
- ✅ Ensemble methods (implemented)
- 🔄 Regularization (in progress)

---

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Optional - system works with public data
EXCHANGE_API_KEY=your_api_key_here
EXCHANGE_API_SECRET=your_api_secret_here

# Email alerts (optional)
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# Database (optional - defaults to data/ml_crypto_data.db)
DATABASE_PATH=data/ml_crypto_data.db
```

### Configuration Files

**automation_config.json** - Main settings
```json
{
  "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
  "timeframes": ["5m", "15m", "1h", "4h", "1d"],
  "quality_threshold": 50,
  "confidence_threshold": 0.50
}
```

**automation_config_ml.json** - ML settings
```json
{
  "models": ["catboost", "xgboost"],
  "feature_selection": true,
  "top_k_features": 50,
  "use_gpu": false
}
```

---

## 📊 Usage Examples

### 🤖 Run the Agent (Recommended)
```python
from crypto_agent import CryptoTradingAgent

# Initialize agent
agent = CryptoTradingAgent()

# Get market overview
overview = agent.get_market_overview()
print(f"Market Regime: {overview['market_regime']}")
print(f"Top Opportunities: {len(overview['top_opportunities'])}")

# Analyze specific opportunity
analysis = agent.analyze_trading_opportunity('BTC/USDT', '1h')
print(agent.format_recommendation(analysis))
```

### 📊 Collect Data
```python
from crypto_ai.data import ComprehensiveMLCollector

collector = ComprehensiveMLCollector()
collector.collect_all_data()  # Collects all symbols × timeframes
```

### 🧠 Train Models
```python
from crypto_ai.ml import OptimizedMLSystem

ml_system = OptimizedMLSystem()
ml_system.train_models()  # Trains ensemble models with feature selection
```

### 🎯 Generate Predictions
```python
from crypto_ai.predictions import PredictionGenerator

generator = PredictionGenerator()
predictions = generator.generate_all_predictions()
```

### 🔍 Analyze Signals
```python
from crypto_ai.analysis import SmartConsensusAnalyzer

analyzer = SmartConsensusAnalyzer()
signals = analyzer.analyze_all_symbols()

for signal in signals:
    if signal['quality_score'] >= 70:
        print(f"{signal['symbol']}: {signal['recommendation']} (Quality: {signal['quality_score']}/100)")
```

---

## 🧪 Testing & Validation

### Run Comprehensive Validation
```bash
python presentation_validation_tests.py
```

This runs:
- ✅ Feature selection benchmark
- ✅ Model training validation
- ✅ Agent execution test
- ✅ Quality filtering verification

### Run Profitability Analysis
```bash
python profitability_analysis.py
```

### Run Agent vs Random Baseline
```bash
python agent_vs_random_backtest.py
```

---

## 🛠️ Development

### File Naming Convention (Updated Jan 21, 2026)

Old files were renamed for clarity:

| Old Name | New Name | Purpose |
|----------|----------|---------|
| `comprehensive_ml_collector_v2.py` | `collect_data.py` | Data collection |
| `optimized_ml_system_v2.py` | `train_models.py` | Model training |
| `generate_and_save_predictions.py` | `generate_predictions.py` | Prediction generation |
| `unified_crypto_analyzer.py` | `analyze_signals.py` | Signal analysis |
| `run_agent_FINAL.py` | `run_agent.py` | Agent execution |
| `crypto_control_center.py` | `control_center.py` | Control center |

See [FILE_RENAMING.md](FILE_RENAMING.md) for details.

### Database Schema

**ml_crypto_data.db** contains:
- `market_data` - OHLCV data
- `technical_indicators` - 90+ indicators
- `ml_predictions` - Model predictions
- `model_performance` - Model metrics
- `agent_recommendations` - Agent decisions
- `market_regime` - Market state detection

---

## 📚 Documentation

- **[FILE_RENAMING.md](FILE_RENAMING.md)** - File renaming history
- **[project_report.md](project_report.md)** - Project status report
- **[ml_reports/final_presentation_summary.md](ml_reports/final_presentation_summary.md)** - Latest validation results
- **Code docstrings** - Inline documentation

---

## 🔐 Security Best Practices

- ✅ Environment variables for sensitive data
- ✅ `.env` file in `.gitignore`
- ✅ No hardcoded API keys
- ✅ Config sanitization before saving
- ⚠️ **Never commit API keys or passwords!**

---

## 🚀 Roadmap

### Immediate (In Progress)
- [ ] Add L1/L2 regularization to reduce overfitting
- [ ] Implement cross-validation for model selection
- [ ] Add more symbols (MATIC, AVAX, DOT)

### Short-term
- [ ] Live trading integration (paper trading first)
- [ ] Web dashboard for monitoring
- [ ] Telegram bot for alerts
- [ ] Backtesting framework improvements

### Long-term
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Sentiment analysis integration
- [ ] Multi-exchange support
- [ ] Portfolio optimization

---

## 🤝 Contributing

1. Review open issues and roadmap
2. Follow code style guidelines (PEP 8)
3. Add tests for new features
4. Update documentation
5. Submit pull request

---

## 📝 License

DS_Course_Final_Project

---

## 🆘 Support & Troubleshooting

### Common Issues

**"No module named 'crypto_agent'"**
```bash
# Ensure you're in the project root directory
cd /path/to/Crypto-Trading-AI
python run_agent.py
```

**"Database not found"**
```bash
# Run data collection first
python collect_data.py
```

**"No trained models found"**
```bash
# Train models first
python train_models.py
```

### Logs
- Application logs: `logs/`
- Data collection logs: `comprehensive_collector.log`
- Agent logs: Check terminal output

---

## ⚠️ Disclaimer

**This is an experimental trading system.**

- ✅ Always test with **paper trading** first
- ✅ Never risk more than you can afford to lose
- ✅ Cryptocurrency trading involves significant risk
- ✅ Past performance does not guarantee future results
- ✅ This is a research/educational project

**The developers are not responsible for any financial losses incurred using this system.**

---

## 📊 Project Stats

- **Lines of Code:** ~15,000+
- **Models Trained:** 4 ensemble models
- **Features Engineered:** 90+ technical indicators
- **Data Points:** 139,049+ market records
- **Symbols Supported:** 5 (BTC, ETH, BNB, SOL, ADA)
- **Timeframes:** 5 (5m, 15m, 1h, 4h, 1d)

---

**Built with ❤️ for the crypto trading community**

*Last Updated: January 21, 2026*
