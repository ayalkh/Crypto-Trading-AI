"""
Unified Crypto Trading Analyzer
Combines Technical Analysis + ML Predictions in one comprehensive system

Features:
- Technical Analysis (RSI, MACD, Bollinger Bands, Volume)
- Multi-timeframe signal combination
- ML prediction integration (FULLY FUNCTIONAL)
- Unified database access (ml_crypto_data.db)
- Works with trained ensemble models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
import argparse
import json
from typing import Dict, Optional
from crypto_ai.features import FeatureEngineer

# Fix Windows encoding
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_analyzer.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)


class DatabaseManager:
    """Manages database connections and data loading"""
    
    def __init__(self, db_path='data/ml_crypto_data.db'):
        """Initialize database manager"""
        self.db_path = db_path
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        logging.info(f"✅ Database manager initialized: {db_path}")
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_available_symbols(self):
        """Get list of available trading symbols"""
        try:
            with self.get_connection() as conn:
                query = "SELECT DISTINCT symbol FROM price_data ORDER BY symbol"
                symbols = pd.read_sql_query(query, conn)['symbol'].tolist()
                return symbols
        except Exception as e:
            logging.error(f"❌ Error getting symbols: {e}")
            return []
    
    def get_available_timeframes(self, symbol=None):
        """Get available timeframes"""
        try:
            with self.get_connection() as conn:
                if symbol:
                    query = "SELECT DISTINCT timeframe FROM price_data WHERE symbol = ? ORDER BY timeframe"
                    params = (symbol,)
                else:
                    query = "SELECT DISTINCT timeframe FROM price_data ORDER BY timeframe"
                    params = ()
                
                timeframes = pd.read_sql_query(query, conn, params=params)['timeframe'].tolist()
                return timeframes
        except Exception as e:
            logging.error(f"❌ Error getting timeframes: {e}")
            return []
    
    def load_price_data(self, symbol, timeframe='1h', limit_hours=168):
        """Load OHLCV price data from database"""
        try:
            with self.get_connection() as conn:
                hours_ago = datetime.now() - timedelta(hours=limit_hours)
                
                query = """
                SELECT timestamp, open, high, low, close, volume
                FROM price_data 
                WHERE symbol = ? AND timeframe = ? AND timestamp >= ?
                ORDER BY timestamp ASC
                """
                
                df = pd.read_sql_query(
                    query, 
                    conn, 
                    params=(symbol, timeframe, hours_ago.strftime('%Y-%m-%d %H:%M:%S'))
                )
                
                if df.empty:
                    logging.warning(f"⚠️ No data found for {symbol} {timeframe}")
                    return None
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Ensure numeric types
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove any NaN rows
                df = df.dropna()
                
                logging.info(f"📊 Loaded {len(df)} candles for {symbol} {timeframe}")
                return df
                
        except Exception as e:
            logging.error(f"❌ Error loading data for {symbol} {timeframe}: {e}")
            return None
    
    def get_data_summary(self):
        """Get summary of available data"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT 
                    symbol,
                    timeframe,
                    COUNT(*) as record_count,
                    MIN(timestamp) as earliest,
                    MAX(timestamp) as latest
                FROM price_data
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
                """
                
                df = pd.read_sql_query(query, conn)
                return df
        except Exception as e:
            logging.error(f"❌ Error getting data summary: {e}")
            return pd.DataFrame()


class TechnicalAnalyzer:
    """Handles all technical analysis calculations"""
    
    def __init__(self):
        """Initialize technical analyzer"""
        # Signal weights (total = 100%)
        self.weights = {
            'rsi': 25,
            'macd': 25,
            'bollinger': 20,
            'volume': 15,
            'momentum': 15
        }
        
        # Thresholds
        self.strong_buy_threshold = 80
        self.buy_threshold = 65
        self.sell_threshold = 35
        self.strong_sell_threshold = 20
        
        logging.info("📊 Technical analyzer initialized")
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd_line - macd_signal
        
        return macd_line, macd_signal, macd_histogram
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def calculate_rsi_score(self, rsi):
        """Convert RSI to 0-100 score (100=strong buy, 0=strong sell)"""
        # RSI oversold (low) = bullish = high score
        # RSI overbought (high) = bearish = low score
        if rsi <= 20:
            return 100
        elif rsi <= 30:
            return 80
        elif rsi <= 40:
            return 65
        elif rsi <= 60:
            return 50
        elif rsi <= 70:
            return 35
        elif rsi <= 80:
            return 20
        else:
            return 0
    
    def calculate_macd_score(self, macd_line, macd_signal, macd_histogram, 
                            prev_macd_line, prev_macd_signal, prev_histogram):
        """Convert MACD signals to 0-100 score"""
        score = 50  # Neutral default
        
        # Check for bullish/bearish crossover
        bullish_crossover = (macd_line > macd_signal) and (prev_macd_line <= prev_macd_signal)
        bearish_crossover = (macd_line < macd_signal) and (prev_macd_line >= prev_macd_signal)
        
        # Check if MACD is above/below zero
        above_zero = macd_line > 0
        
        # Calculate score
        if bullish_crossover:
            score = 90 if above_zero else 75
        elif bearish_crossover:
            score = 10 if not above_zero else 25
        elif macd_line > macd_signal:
            # Bullish
            if macd_histogram > prev_histogram:  # Increasing histogram
                score = 70 if above_zero else 60
            else:
                score = 60 if above_zero else 55
        else:
            # Bearish
            if macd_histogram < prev_histogram:  # Decreasing histogram
                score = 30 if not above_zero else 40
            else:
                score = 40 if not above_zero else 45
        
        return score
    
    def calculate_bollinger_score(self, close, bb_upper, bb_lower):
        """Convert Bollinger Band position to 0-100 score"""
        # Calculate position within bands (0-100)
        bb_range = bb_upper - bb_lower
        if bb_range == 0:
            return 50
        
        position = ((close - bb_lower) / bb_range) * 100
        
        # Near lower band = oversold = bullish = high score
        # Near upper band = overbought = bearish = low score
        if position <= 5:
            return 95
        elif position <= 20:
            return 80
        elif position <= 40:
            return 65
        elif position <= 60:
            return 50
        elif position <= 80:
            return 35
        elif position <= 95:
            return 20
        else:
            return 5
    
    def calculate_volume_score(self, volume, volume_ma, price_change):
        """Calculate volume confirmation score"""
        if volume_ma == 0:
            return 50
        
        volume_ratio = volume / volume_ma
        
        # High volume on up move = bullish
        # High volume on down move = bearish
        if price_change > 0:
            if volume_ratio > 2.0:
                return 90
            elif volume_ratio > 1.5:
                return 75
            elif volume_ratio > 0.7:
                return 60
            else:
                return 45
        elif price_change < 0:
            if volume_ratio > 2.0:
                return 10
            elif volume_ratio > 1.5:
                return 25
            elif volume_ratio > 0.7:
                return 40
            else:
                return 55
        else:
            return 50
    
    def calculate_momentum_score(self, momentum_5, momentum_10, momentum_20):
        """Calculate momentum score based on multiple periods"""
        # Positive momentum = bullish
        score = 50
        
        if momentum_5 > 0.02:  # >2% gain
            score += 15
        elif momentum_5 > 0:
            score += 5
        elif momentum_5 < -0.02:  # >2% loss
            score -= 15
        else:
            score -= 5
        
        if momentum_10 > 0.03:
            score += 10
        elif momentum_10 > 0:
            score += 5
        elif momentum_10 < -0.03:
            score -= 10
        else:
            score -= 5
        
        if momentum_20 > 0.05:
            score += 5
        elif momentum_20 < -0.05:
            score -= 5
        
        return max(0, min(100, score))
    
    def analyze(self, df):
        """
        Perform complete technical analysis on price data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            dict with analysis results
        """
        if df is None or len(df) < 50:
            if df is not None:
                logging.warning(f"⚠️ Insufficient data for analysis: {len(df)} candles (need 50+)")
            else:
                logging.warning("⚠️ Insufficient data for analysis: No data available")
            return None
        
        try:
            # Calculate indicators
            df['rsi'] = self.calculate_rsi(df['close'])
            
            macd_line, macd_signal, macd_histogram = self.calculate_macd(df['close'])
            df['macd_line'] = macd_line
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_histogram
            
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'])
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            
            # Calculate volume indicators
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['price_change'] = df['close'].pct_change()
            
            # Calculate momentum
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_10'] = df['close'].pct_change(10)
            df['momentum_20'] = df['close'].pct_change(20)
            
            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Calculate individual scores
            rsi_score = self.calculate_rsi_score(latest['rsi'])
            
            macd_score = self.calculate_macd_score(
                latest['macd_line'], latest['macd_signal'], latest['macd_histogram'],
                prev['macd_line'], prev['macd_signal'], prev['macd_histogram']
            )
            
            bollinger_score = self.calculate_bollinger_score(
                latest['close'], latest['bb_upper'], latest['bb_lower']
            )
            
            volume_score = self.calculate_volume_score(
                latest['volume'], latest['volume_ma'], latest['price_change']
            )
            
            momentum_score = self.calculate_momentum_score(
                latest['momentum_5'], latest['momentum_10'], latest['momentum_20']
            )
            
            # Calculate weighted combined score
            combined_score = (
                rsi_score * self.weights['rsi'] / 100 +
                macd_score * self.weights['macd'] / 100 +
                bollinger_score * self.weights['bollinger'] / 100 +
                volume_score * self.weights['volume'] / 100 +
                momentum_score * self.weights['momentum'] / 100
            )
            
            # Determine signal
            if combined_score >= self.strong_buy_threshold:
                signal = 'STRONG_BUY'
            elif combined_score >= self.buy_threshold:
                signal = 'BUY'
            elif combined_score <= self.strong_sell_threshold:
                signal = 'STRONG_SELL'
            elif combined_score <= self.sell_threshold:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            # Calculate confidence (distance from neutral)
            confidence = abs(combined_score - 50) * 2  # 0-100 scale
            
            return {
                'timestamp': latest['timestamp'],
                'price': latest['close'],
                'signal': signal,
                'combined_score': combined_score,
                'confidence': confidence,
                'individual_scores': {
                    'rsi': rsi_score,
                    'macd': macd_score,
                    'bollinger': bollinger_score,
                    'volume': volume_score,
                    'momentum': momentum_score
                },
                'indicators': {
                    'rsi': latest['rsi'],
                    'macd_line': latest['macd_line'],
                    'macd_signal': latest['macd_signal'],
                    'macd_histogram': latest['macd_histogram'],
                    'bb_upper': latest['bb_upper'],
                    'bb_middle': latest['bb_middle'],
                    'bb_lower': latest['bb_lower'],
                    'volume_ratio': latest['volume'] / latest['volume_ma'] if latest['volume_ma'] > 0 else 1
                }
            }
            
        except Exception as e:
            logging.error(f"❌ Analysis error: {e}")
            return None


class MLPredictor:
    """
    ML Prediction system - Loads and uses trained ensemble models
    """
    
    def __init__(self, models_dir='ml_models'):
        """Initialize ML predictor"""
        self.models_dir = models_dir
        self.models_cache = {}  # Cache loaded models
        self.scalers_cache = {}  # Cache loaded scalers
        self.features_cache = {}  # Cache feature lists
        self.models_available = False
        
        # Check if joblib is available
        try:
            import joblib
            self.joblib = joblib
            self.joblib_available = True
        except ImportError:
            self.joblib_available = False
            logging.error("❌ joblib not available. Install: pip install joblib")
            return
        
        # Check if models exist
        if os.path.exists(models_dir):
            self._scan_available_models()
        else:
            logging.warning(f"⚠️ Models directory not found: {models_dir}")
        
        logging.info(f"🤖 ML Predictor initialized (Models available: {self.models_available})")
        self.feature_engineer = FeatureEngineer()
    
    def _scan_available_models(self):
        """Scan and count available models"""
        try:
            model_files = [f for f in os.listdir(self.models_dir) 
                          if f.endswith('.joblib') and not f.endswith('_scaler.joblib') 
                          and not f.endswith('_features.joblib')]
            
            if not model_files:
                logging.warning("⚠️ No model files found in ml_models/")
                return
            
            # Count models by type
            price_models = [f for f in model_files if '_price_' in f]
            direction_models = [f for f in model_files if '_direction_' in f]
            
            logging.info(f"✅ Found trained models:")
            logging.info(f"   Price models: {len(price_models)}")
            logging.info(f"   Direction models: {len(direction_models)}")
            
            self.models_available = len(price_models) > 0 or len(direction_models) > 0
            
        except Exception as e:
            logging.error(f"❌ Error scanning models: {e}")
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features EXACTLY as done during training
        Delegated to FeatureEngineer class for consistency
        """
        if df.empty or len(df) < 60:
            return pd.DataFrame()
        
        logging.info("🧠 Generating features using centralized FeatureEngineer...")
        df_features = self.feature_engineer.create_features(df)
        
        # Drop NaN values
        df_features.dropna(inplace=True)
        
        return df_features
    
    def _load_model_components(self, symbol: str, timeframe: str, model_type: str):
        """Load model, scaler, and feature list for a specific configuration"""
        safe_symbol = symbol.replace('/', '_')
        cache_key = f"{safe_symbol}_{timeframe}_{model_type}"
        
        # Check cache first
        if cache_key in self.models_cache:
            return (self.models_cache[cache_key], 
                   self.scalers_cache.get(cache_key),
                   self.features_cache.get(cache_key))
        
        try:
            # Load model (try ensemble models: lightgbm, xgboost, catboost)
            models = {}
            for model_name in ['lightgbm', 'xgboost', 'catboost']:
                model_path = f"{self.models_dir}/{safe_symbol}_{timeframe}_{model_type}_{model_name}.joblib"
                if os.path.exists(model_path):
                    models[model_name] = self.joblib.load(model_path)
            
            if not models:
                return None, None, None
            
            # Load scaler
            scaler_path = f"{self.models_dir}/{safe_symbol}_{timeframe}_{model_type}_scaler.joblib"
            scaler = self.joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            
            # Load feature list
            features_path = f"{self.models_dir}/{safe_symbol}_{timeframe}_{model_type}_features.joblib"
            features = self.joblib.load(features_path) if os.path.exists(features_path) else None
            
            # Cache the loaded components
            self.models_cache[cache_key] = models
            if scaler:
                self.scalers_cache[cache_key] = scaler
            if features:
                self.features_cache[cache_key] = features
            
            return models, scaler, features
            
        except Exception as e:
            logging.warning(f"⚠️ Could not load model components for {cache_key}: {e}")
            return None, None, None
    
    def predict(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Dict:
        """
        Make ML predictions using trained ensemble models
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h')
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with ML predictions
        """
        if not self.joblib_available or not self.models_available:
            return {
                'ml_available': False,
                'price_change_prediction': 0.0,
                'direction_prediction': 'NEUTRAL',
                'direction_probability': 0.5,
                'confidence': 0.0,
                'note': 'ML models not available'
            }
        
        if df is None or df.empty or len(df) < 60:
            return {
                'ml_available': False,
                'price_change_prediction': 0.0,
                'direction_prediction': 'NEUTRAL',
                'direction_probability': 0.5,
                'confidence': 0.0,
                'note': 'Insufficient data for ML prediction'
            }
        
        try:
            # Create features
            df_features = self._create_features(df)
            if df_features.empty:
                raise ValueError("Failed to create features")
            
            # Get predictions from both price and direction models
            price_pred = self._predict_price(symbol, timeframe, df_features)
            direction_pred = self._predict_direction(symbol, timeframe, df_features)
            
            # Combine predictions
            if price_pred or direction_pred:
                # Use price prediction as primary
                if price_pred:
                    price_change = price_pred['price_change']
                    confidence = price_pred['confidence']
                else:
                    price_change = 0.0
                    confidence = 0.0
                
                # Use direction prediction
                if direction_pred:
                    direction = direction_pred['direction']
                    direction_prob = direction_pred['probability']
                    if confidence == 0:
                        confidence = direction_prob
                else:
                    direction = 'UP' if price_change > 0 else 'DOWN' if price_change < 0 else 'NEUTRAL'
                    direction_prob = abs(price_change) if price_change != 0 else 0.5
                
                return {
                    'ml_available': True,
                    'price_change_prediction': float(price_change),
                    'direction_prediction': direction,
                    'direction_probability': float(direction_prob),
                    'confidence': float(confidence),
                    'note': f'Ensemble prediction from {len(price_pred.get("models_used", []))} models'
                }
            else:
                # No models available for this symbol/timeframe
                return {
                    'ml_available': False,
                    'price_change_prediction': 0.0,
                    'direction_prediction': 'NEUTRAL',
                    'direction_probability': 0.5,
                    'confidence': 0.0,
                    'note': f'No trained models for {symbol} {timeframe}'
                }
                
        except Exception as e:
            logging.error(f"❌ ML prediction error for {symbol} {timeframe}: {e}")
            return {
                'ml_available': False,
                'price_change_prediction': 0.0,
                'direction_prediction': 'NEUTRAL',
                'direction_probability': 0.5,
                'confidence': 0.0,
                'note': f'Prediction error: {str(e)}'
            }
    
    def _predict_price(self, symbol: str, timeframe: str, df_features: pd.DataFrame) -> Optional[Dict]:
        """Predict price change using ensemble of regression models"""
        models, scaler, feature_cols = self._load_model_components(symbol, timeframe, 'price')
        
        if not models or scaler is None or feature_cols is None:
            return None
        
        try:
            # Prepare features
            available_features = [col for col in feature_cols if col in df_features.columns]
            if not available_features:
                return None
            
            X_latest = df_features[available_features].iloc[-1:].values
            X_scaled = scaler.transform(X_latest)
            
            # Get predictions from all models
            predictions = []
            weights = {
                'lightgbm': 0.50,
                'xgboost': 0.30,
                'catboost': 0.20
            }
            
            models_used = []
            for model_name, model in models.items():
                pred = model.predict(X_scaled)[0]
                predictions.append(pred * weights.get(model_name, 0.33))
                models_used.append(model_name)
            
            # Ensemble prediction (weighted average)
            ensemble_pred = sum(predictions)
            
            # Calculate confidence (inverse of prediction variance)
            if len(predictions) > 1:
                pred_std = np.std(predictions)
                confidence = max(0.0, min(1.0, 1.0 - pred_std * 10))  # Scale std to 0-1
            else:
                confidence = 0.7  # Default confidence for single model
            
            return {
                'price_change': ensemble_pred,
                'confidence': confidence,
                'models_used': models_used
            }
            
        except Exception as e:
            logging.warning(f"⚠️ Price prediction error: {e}")
            return None
    
    def _predict_direction(self, symbol: str, timeframe: str, df_features: pd.DataFrame) -> Optional[Dict]:
        """Predict direction using ensemble of classification models"""
        models, scaler, feature_cols = self._load_model_components(symbol, timeframe, 'direction')
        
        if not models or scaler is None or feature_cols is None:
            return None
        
        try:
            # Prepare features
            available_features = [col for col in feature_cols if col in df_features.columns]
            if not available_features:
                return None
            
            X_latest = df_features[available_features].iloc[-1:].values
            X_scaled = scaler.transform(X_latest)
            
            # Get predictions from all models
            up_votes = 0
            down_votes = 0
            
            for model_name, model in models.items():
                pred = model.predict(X_scaled)[0]
                if pred == 1:
                    up_votes += 1
                else:
                    down_votes += 1
            
            # Determine direction and probability
            total_votes = up_votes + down_votes
            if up_votes > down_votes:
                direction = 'UP'
                probability = up_votes / total_votes
            elif down_votes > up_votes:
                direction = 'DOWN'
                probability = down_votes / total_votes
            else:
                direction = 'NEUTRAL'
                probability = 0.5
            
            return {
                'direction': direction,
                'probability': probability
            }
            
        except Exception as e:
            logging.warning(f"⚠️ Direction prediction error: {e}")
            return None


class UnifiedAnalyzer:
    """
    Main unified analyzer combining TA and ML
    """
    
    def __init__(self, db_path='data/ml_crypto_data.db'):
        """Initialize unified analyzer"""
        self.db_manager = DatabaseManager(db_path)
        self.technical_analyzer = TechnicalAnalyzer()
        self.ml_predictor = MLPredictor()
        
        # Multi-timeframe weights
        self.timeframe_weights = {
            '5m': 10,
            '15m': 20,
            '1h': 30,
            '4h': 25,
            '1d': 15
        }
        
        # ML vs TA weighting by timeframe
        self.ml_ta_weights = {
            '5m': {'ml': 0.7, 'ta': 0.3},   # More ML for short-term
            '15m': {'ml': 0.6, 'ta': 0.4},
            '1h': {'ml': 0.5, 'ta': 0.5},   # Balanced
            '4h': {'ml': 0.4, 'ta': 0.6},
            '1d': {'ml': 0.3, 'ta': 0.7}    # More TA for long-term
        }
        
        logging.info("🚀 Unified Analyzer initialized")
    
    def analyze_single_timeframe(self, symbol, timeframe):
        """Analyze a single symbol and timeframe"""
        logging.info(f"🔍 Analyzing {symbol} {timeframe}")
        
        # Use appropriate lookback period for each timeframe
        # We need at least 50 candles, but more is better for indicators
        lookback_hours = {
            '5m': 168,      # 7 days = 2,016 candles
            '15m': 336,     # 14 days = 1,344 candles
            '1h': 720,      # 30 days = 720 candles
            '4h': 1440,     # 60 days = 360 candles
            '1d': 4320      # 180 days = 180 candles
        }
        
        limit_hours = lookback_hours.get(timeframe, 720)  # Default 30 days
        
        # Load data
        df = self.db_manager.load_price_data(symbol, timeframe, limit_hours)
        if df is None:
            return None
        
        # Technical analysis
        ta_result = self.technical_analyzer.analyze(df)
        if ta_result is None:
            return None
        
        # ML predictions
        ml_result = self.ml_predictor.predict(symbol, timeframe, df)
        
        # Combine signals
        combined = self._combine_ta_ml_signals(ta_result, ml_result, timeframe)
        
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': ta_result['timestamp'],
            'price': ta_result['price'],
            'technical_analysis': ta_result,
            'ml_predictions': ml_result,
            'combined': combined
        }
        
        logging.info(f"✅ {symbol} {timeframe}: {combined['signal']} "
                    f"(Score: {combined['combined_score']:.1f}, "
                    f"Confidence: {combined['confidence']:.1f}%)")
        
        return result
    
    def _combine_ta_ml_signals(self, ta_result, ml_result, timeframe):
        """Combine TA and ML signals with intelligent weighting"""
        
        # Get weights for this timeframe
        weights = self.ml_ta_weights.get(timeframe, {'ml': 0.5, 'ta': 0.5})
        
        # TA score (0-100)
        ta_score = ta_result['combined_score']
        ta_confidence = ta_result['confidence']
        
        # ML score (convert to 0-100 scale)
        if ml_result['ml_available'] and ml_result['confidence'] > 0:
            # Convert ML prediction to score
            ml_direction = ml_result['direction_prediction']
            ml_prob = ml_result['direction_probability']
            
            if ml_direction == 'UP':
                ml_score = 50 + (ml_prob * 50)  # 50-100
            elif ml_direction == 'DOWN':
                ml_score = 50 - (ml_prob * 50)  # 0-50
            else:
                ml_score = 50  # Neutral
            
            ml_confidence = ml_result['confidence'] * 100
        else:
            # No ML available, use neutral
            ml_score = 50
            ml_confidence = 0
        
        # Combine scores with weights
        if ml_confidence > 0:
            combined_score = (ta_score * weights['ta'] + ml_score * weights['ml'])
            combined_confidence = (ta_confidence * weights['ta'] + ml_confidence * weights['ml'])
        else:
            # Only TA available
            combined_score = ta_score
            combined_confidence = ta_confidence
        
        # Determine final signal
        if combined_score >= 80:
            signal = 'STRONG_BUY'
        elif combined_score >= 65:
            signal = 'BUY'
        elif combined_score <= 20:
            signal = 'STRONG_SELL'
        elif combined_score <= 35:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'combined_score': combined_score,
            'confidence': combined_confidence,
            'ta_score': ta_score,
            'ml_score': ml_score,
            'ta_weight': weights['ta'],
            'ml_weight': weights['ml']
        }
    
    def analyze_multi_timeframe(self, symbol):
        """Analyze symbol across multiple timeframes"""
        logging.info(f"📊 Multi-timeframe analysis for {symbol}")
        
        available_timeframes = self.db_manager.get_available_timeframes(symbol)
        timeframe_results = []
        
        for timeframe in available_timeframes:
            if timeframe in self.timeframe_weights:
                result = self.analyze_single_timeframe(symbol, timeframe)
                if result:
                    timeframe_results.append(result)
        
        if not timeframe_results:
            logging.warning(f"⚠️ No valid results for {symbol}")
            return None
        
        # Combine multi-timeframe signals
        mtf_result = self._combine_multi_timeframe(timeframe_results)
        mtf_result['symbol'] = symbol
        mtf_result['timeframe_results'] = timeframe_results
        
        logging.info(f"🎯 {symbol} Multi-Timeframe: {mtf_result['multi_timeframe_signal']} "
                    f"(Confidence: {mtf_result['mtf_confidence']:.1f}%)")
        
        return mtf_result
    
    def _combine_multi_timeframe(self, timeframe_results):
        """Combine signals from multiple timeframes"""
        total_weight = 0
        weighted_score = 0
        
        for result in timeframe_results:
            timeframe = result['timeframe']
            weight = self.timeframe_weights.get(timeframe, 20)
            score = result['combined']['combined_score']
            
            weighted_score += score * weight
            total_weight += weight
        
        mtf_score = weighted_score / total_weight if total_weight > 0 else 50
        
        # Determine signal
        if mtf_score >= 80:
            mtf_signal = 'STRONG_BUY'
        elif mtf_score >= 65:
            mtf_signal = 'BUY'
        elif mtf_score <= 20:
            mtf_signal = 'STRONG_SELL'
        elif mtf_score <= 35:
            mtf_signal = 'SELL'
        else:
            mtf_signal = 'HOLD'
        
        # Calculate confidence
        mtf_confidence = abs(mtf_score - 50) * 2
        
        # Get consensus
        signals = [r['combined']['signal'] for r in timeframe_results]
        from collections import Counter
        signal_counts = Counter(signals)
        dominant_signal = signal_counts.most_common(1)[0][0]
        consensus = (signal_counts[dominant_signal] / len(signals)) * 100
        
        # Get latest price
        latest_price = timeframe_results[0]['price']
        
        return {
            'multi_timeframe_signal': mtf_signal,
            'mtf_combined_score': mtf_score,
            'mtf_confidence': mtf_confidence,
            'price': latest_price,
            'dominant_signal': dominant_signal,
            'signal_consensus': consensus,
            'timestamp': datetime.now()
        }
    
    def analyze_all_symbols(self):
        """Analyze all available symbols"""
        logging.info("🚀 Starting comprehensive analysis")
        
        symbols = self.db_manager.get_available_symbols()
        if not symbols:
            logging.error("❌ No symbols found in database")
            return {}
        
        results = {}
        for symbol in symbols:
            try:
                result = self.analyze_multi_timeframe(symbol)
                if result:
                    results[symbol] = result
            except Exception as e:
                logging.error(f"❌ Error analyzing {symbol}: {e}")
        
        logging.info(f"✅ Analysis complete: {len(results)} symbols")
        return results
    
    def display_results(self, results):
        """Display analysis results"""
        print(f"\n{'='*80}")
        print("🚀 UNIFIED CRYPTO TRADING ANALYSIS")
        print(f"{'='*80}")
        print(f"⏰ Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Technical Analysis + ML Predictions")
        print(f"{'='*80}")
        
        if not results:
            print("❌ No analysis results available")
            return
        
        for symbol, result in results.items():
            signal = result['multi_timeframe_signal']
            score = result['mtf_combined_score']
            confidence = result['mtf_confidence']
            price = result['price']
            consensus = result['signal_consensus']
            
            # Signal emoji
            emoji = {
                'STRONG_BUY': '🟢🟢',
                'BUY': '🟢',
                'HOLD': '🟡',
                'SELL': '🔴',
                'STRONG_SELL': '🔴🔴'
            }.get(signal, '⚪')
            
            print(f"\n📊 {symbol}")
            print(f"   {emoji} Signal: {signal}")
            print(f"   💰 Price: ${price:,.2f}")
            print(f"   📈 Combined Score: {score:.1f}/100")
            print(f"   🎯 Confidence: {confidence:.1f}%")
            print(f"   🤝 Consensus: {consensus:.1f}%")
            
            # Timeframe breakdown
            print(f"   🔬 Timeframe Breakdown:")
            for tf_result in result['timeframe_results']:
                tf = tf_result['timeframe']
                tf_signal = tf_result['combined']['signal']
                tf_score = tf_result['combined']['combined_score']
                tf_conf = tf_result['combined']['confidence']
                weight = self.timeframe_weights.get(tf, 0)
                
                # Show ML contribution
                ml_available = tf_result['ml_predictions']['ml_available']
                ml_note = "🤖 ML" if ml_available else "📊 TA"
                
                tf_emoji = {
                    'STRONG_BUY': '🟢🟢',
                    'BUY': '🟢',
                    'HOLD': '🟡',
                    'SELL': '🔴',
                    'STRONG_SELL': '🔴🔴'
                }.get(tf_signal, '⚪')
                
                print(f"      {tf:>3}: {tf_emoji} {tf_signal:12} "
                      f"Score: {tf_score:5.1f} "
                      f"Conf: {tf_conf:5.1f}% "
                      f"(Weight: {weight:2d}%) {ml_note}")
        
        # Market summary
        self._display_market_summary(results)
    
    def _display_market_summary(self, results):
        """Display market summary statistics"""
        signals = [r['multi_timeframe_signal'] for r in results.values()]
        
        buy_count = sum(1 for s in signals if s in ['BUY', 'STRONG_BUY'])
        sell_count = sum(1 for s in signals if s in ['SELL', 'STRONG_SELL'])
        hold_count = sum(1 for s in signals if s == 'HOLD')
        
        print(f"\n{'='*80}")
        print("🌍 MARKET SUMMARY")
        print(f"{'='*80}")
        print(f"   🟢 Buy Signals: {buy_count}")
        print(f"   🔴 Sell Signals: {sell_count}")
        print(f"   🟡 Hold Signals: {hold_count}")
        
        if buy_count > sell_count:
            sentiment = "📈 BULLISH"
        elif sell_count > buy_count:
            sentiment = "📉 BEARISH"
        else:
            sentiment = "⚖️  NEUTRAL"
        
        print(f"   🌍 Overall Sentiment: {sentiment}")
        
        # Best opportunities
        buy_opps = [(s, r) for s, r in results.items() 
                   if r['multi_timeframe_signal'] in ['BUY', 'STRONG_BUY']]
        sell_opps = [(s, r) for s, r in results.items() 
                    if r['multi_timeframe_signal'] in ['SELL', 'STRONG_SELL']]
        
        print(f"\n🎯 TOP OPPORTUNITIES:")
        
        if buy_opps:
            best_buy = max(buy_opps, key=lambda x: x[1]['mtf_confidence'])
            print(f"   🟢 Best BUY: {best_buy[0]} - {best_buy[1]['multi_timeframe_signal']} "
                  f"({best_buy[1]['mtf_confidence']:.1f}% confidence)")
        
        if sell_opps:
            best_sell = max(sell_opps, key=lambda x: x[1]['mtf_confidence'])
            print(f"   🔴 Best SELL: {best_sell[0]} - {best_sell[1]['multi_timeframe_signal']} "
                  f"({best_sell[1]['mtf_confidence']:.1f}% confidence)")
        
        if not buy_opps and not sell_opps:
            print(f"   ⏳ All markets in HOLD - Wait for clearer signals")
        
        print(f"{'='*80}\n")
    
    def save_results(self, results, output_dir='analysis_results'):
        """Save analysis results to JSON"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{output_dir}/analysis_{timestamp}.json"
            
            # Convert to serializable format
            serializable_results = {}
            for symbol, result in results.items():
                serializable_results[symbol] = {
                    'multi_timeframe_signal': result['multi_timeframe_signal'],
                    'mtf_combined_score': float(result['mtf_combined_score']),
                    'mtf_confidence': float(result['mtf_confidence']),
                    'price': float(result['price']),
                    'timestamp': result['timestamp'].isoformat()
                }
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            # Also save as latest
            with open(f"{output_dir}/latest_analysis.json", 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logging.info(f"💾 Results saved: {filename}")
            
        except Exception as e:
            logging.error(f"❌ Error saving results: {e}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Unified Crypto Trading Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all symbols
  python unified_crypto_analyzer.py
  
  # Analyze specific symbols
  python unified_crypto_analyzer.py --symbols BTC/USDT ETH/USDT
  
  # Show database summary
  python unified_crypto_analyzer.py --summary
  
  # Save results to file
  python unified_crypto_analyzer.py --save
        """
    )
    
    parser.add_argument('--symbols', nargs='+',
                       help='Specific symbols to analyze')
    parser.add_argument('--summary', action='store_true',
                       help='Show database summary')
    parser.add_argument('--save', action='store_true',
                       help='Save results to JSON file')
    parser.add_argument('--db-path', default='data/ml_crypto_data.db',
                       help='Database path')
    
    args = parser.parse_args()
    
    print("🚀 UNIFIED CRYPTO TRADING ANALYZER")
    print("=" * 80)
    
    try:
        # Initialize analyzer
        analyzer = UnifiedAnalyzer(db_path=args.db_path)
        
        # Show database summary if requested
        if args.summary:
            summary = analyzer.db_manager.get_data_summary()
            print("\n📊 DATABASE SUMMARY")
            print("=" * 80)
            print(summary.to_string(index=False))
            print("=" * 80)
            return
        
        # Get symbols to analyze
        if args.symbols:
            # Analyze specific symbols
            results = {}
            for symbol in args.symbols:
                result = analyzer.analyze_multi_timeframe(symbol)
                if result:
                    results[symbol] = result
        else:
            # Analyze all symbols
            results = analyzer.analyze_all_symbols()
        
        # Display results
        if results:
            analyzer.display_results(results)
            
            # Save if requested
            if args.save:
                analyzer.save_results(results)
        else:
            print("❌ No analysis results available")
            print("💡 Make sure your database has sufficient data!")
        
        print("\n✅ Analysis complete!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("💡 Run the unified_ml_collector.py first to collect data!")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()