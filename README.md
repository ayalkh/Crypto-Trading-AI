# Crypto-Trading-AI 🚀

AI-powered cryptocurrency trading assistant using Machine Learning and Deep Learning for signal generation and trading automation.

## ✨ Features

- **Multi-Timeframe Analysis**: Analyze signals across 5m, 15m, 1h, 4h, and 1d timeframes
- **ML/DL Predictions**: LightGBM, XGBoost, CatBoost ensemble for price and direction prediction
- **Technical Indicators**: RSI, MACD, Bollinger Bands, and custom indicators
- **Automated Trading Signals**: Combined signal analysis with confidence scoring
- **Performance Tracking**: Backtesting and performance metrics
- **24/7 Automation**: Scheduled data collection and analysis
- **Alert System**: Email, desktop, and log-based alerts

## 🎯 ML Performance

**Current Configuration:** Enhanced (Advanced Features + Tuning)

| Metric | Value |
|--------|-------|
| Configuration | 100+ features, feature selection, hyperparameter tuning |
| Models | LightGBM, XGBoost, CatBoost, GRU, LSTM ensemble |
| Advanced Indicators | Stochastic, Williams %R, OBV, Ichimoku, ADX, Market Regime |

> **Note:** The system uses **enhanced configuration** with advanced technical indicators, feature selection, and hyperparameter optimization for maximum predictive power.

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install python-dotenv  # For environment variable support
   ```

2. **Configure Environment**
   - Copy `.env.example` to `.env` and fill in your values
   - Edit `automation_config.json` with your preferences

3. **Run the System**
   ```bash
   python crypto_control_center.py
   ```

For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)

## 📁 Project Structure

```
Crypto-Trading-AI/
├── data/                    # Database and data files
├── ml_models/              # Trained ML models
├── ml_predictions/         # ML prediction outputs
├── utils/                  # Utility modules (NEW!)
│   ├── config_loader.py    # Enhanced config with env vars
│   └── retry_handler.py    # Retry logic for API calls
├── multi_timeframe_collector.py    # Data collection
├── multi_timeframe_analyzer.py     # Signal analysis
├── ml_integration_system.py        # ML predictions
├── crypto_control_center.py        # Main control interface
├── performance_tracker.py           # Performance metrics
└── automation_config.json           # Configuration file
```

## 🔧 Configuration

### Environment Variables (.env)
- `EXCHANGE_API_KEY` - Exchange API key (if needed)
- `EMAIL_USERNAME` - Email for alerts
- `EMAIL_PASSWORD` - Email password/app password
- `DATABASE_PATH` - Database file path

### Configuration Files
- `automation_config.json` - Main configuration
- `automation_config_ml.json` - ML-specific settings

## 📊 Usage Examples

### Data Collection
```python
from multi_timeframe_collector import EnhancedMultiTimeframeCollector

collector = EnhancedMultiTimeframeCollector()
collector.collect_all_data()
```

### Signal Analysis
```python
from multi_timeframe_analyzer import IntegratedUltimateAnalyzer

analyzer = IntegratedUltimateAnalyzer()
results = analyzer.analyze_all_symbols()
```

### ML Predictions
```python
from ml_integration_system import CryptoMLSystem

ml_system = CryptoMLSystem()
predictions = ml_system.make_predictions('BTC/USDT', '1h')
```

## 🛠️ Improvements

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for:
- Prioritized improvement recommendations
- Security enhancements
- Code quality improvements
- Performance optimizations
- Testing strategies

## 📚 Documentation

- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Detailed setup instructions
- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Improvement roadmap
- Code docstrings - Inline documentation

## 🔐 Security

- ✅ Environment variables for sensitive data
- ✅ `.env` file in `.gitignore`
- ✅ Config sanitization before saving
- ⚠️ Never commit API keys or passwords!

## 🧪 Testing

```bash
# Run tests (if available)
pytest tests/

# Check database
python check_database.py
```

## 📈 Performance Tracking

```bash
python performance_tracker.py
```

## 🤝 Contributing

1. Review [IMPROVEMENTS.md](IMPROVEMENTS.md) for areas to improve
2. Follow code style guidelines
3. Add tests for new features
4. Update documentation

## 📝 License

DS_Course_Final_Project

## 🆘 Support

- Check logs in project root
- Review [SETUP_GUIDE.md](SETUP_GUIDE.md) for troubleshooting
- Ensure all dependencies are installed

---

**Note**: This is a trading system. Always test thoroughly with paper trading before using real funds. Trading cryptocurrencies involves risk.
