"""
ML-Enhanced Multi-Timeframe Analyzer
Integrates machine learning predictions with your existing technical analysis
"""
import os
import sys

if sys.platform.startswith('win'):
    try:
        # Try to set UTF-8 encoding for stdout
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        # If reconfigure doesn't work, try alternative
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
import numpy as np
import pandas as pd
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List
import logging

# Import your existing analyzer (assuming it exists)
try:
    from multi_timeframe_analyzer import MultiTimeframeAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    logging.warning("⚠️ Original analyzer not found, creating standalone version")

# Import our ML system
from ml_integration_system import CryptoMLSystem

class MLEnhancedAnalyzer:
    def __init__(self, db_path='data/multi_timeframe_data.db'):
        """Initialize the ML-enhanced analyzer"""
        self.db_path = db_path
        self.ml_system = CryptoMLSystem(db_path)
        
        # Initialize original analyzer if available
        if ANALYZER_AVAILABLE:
            self.original_analyzer = MultiTimeframeAnalyzer(db_path)
        
        self.results = {}
        
        logging.info("🧠 ML-Enhanced Analyzer initialized")
    
    def analyze_symbol(self, symbol: str, timeframes: List[str] = None) -> Dict:
        """Analyze a symbol using both technical analysis and ML predictions"""
        if timeframes is None:
            timeframes = ['15m', '1h', '4h', '1d']
        
        logging.info(f"🔍 Analyzing {symbol} with ML enhancement")
        
        symbol_results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'timeframes': {},
            'ml_predictions': {},
            'combined_signals': {}
        }
        
        # Run original technical analysis if available
        if ANALYZER_AVAILABLE:
            try:
                ta_results = self.original_analyzer.analyze_symbol(symbol, timeframes)
                symbol_results['technical_analysis'] = ta_results
            except Exception as e:
                logging.error(f"❌ Technical analysis failed: {e}")
                symbol_results['technical_analysis'] = {}
        
        # Run ML predictions for each timeframe
        for timeframe in timeframes:
            logging.info(f"🔮 Getting ML predictions for {symbol} {timeframe}")
            
            # Load models if not already loaded
            self.ml_system.load_models(symbol, timeframe)
            
            # Get ML predictions
            ml_predictions = self.ml_system.make_predictions(symbol, timeframe)
            
            if ml_predictions:
                symbol_results['ml_predictions'][timeframe] = ml_predictions
                
                # Create combined signals
                combined_signal = self.create_combined_signal(
                    symbol, timeframe, ml_predictions, 
                    symbol_results.get('technical_analysis', {})
                )
                symbol_results['combined_signals'][timeframe] = combined_signal
        
        # Generate overall recommendation
        symbol_results['recommendation'] = self.generate_overall_recommendation(symbol_results)
        
        return symbol_results
    
    def create_combined_signal(self, symbol: str, timeframe: str, 
                              ml_predictions: Dict, ta_results: Dict) -> Dict:
        """Combine ML predictions with technical analysis"""
        signal = {
            'timestamp': datetime.now().isoformat(),
            'timeframe': timeframe,
            'ml_signals': {},
            'technical_signals': {},
            'combined_score': 0,
            'confidence': 0,
            'action': 'HOLD'
        }
        
        # Process ML signals
        ml_score = 0
        ml_confidence = 0
        
        if 'direction' in ml_predictions:
            direction = ml_predictions['direction']
            prob = ml_predictions.get('direction_probability', 0.5)
            
            signal['ml_signals']['direction'] = direction
            signal['ml_signals']['direction_confidence'] = prob
            
            if direction == 'UP' and prob > 0.6:
                ml_score += 1
                ml_confidence += prob
            elif direction == 'DOWN' and prob > 0.6:
                ml_score -= 1
                ml_confidence += prob
        
        if 'price_change' in ml_predictions:
            price_change = ml_predictions['price_change']
            signal['ml_signals']['predicted_change'] = price_change
            
            if abs(price_change) > 0.01:  # > 1% change
                if price_change > 0:
                    ml_score += 1
                else:
                    ml_score -= 1
                ml_confidence += min(abs(price_change) * 100, 1.0)  # Cap at 1.0
        
        if 'lstm_price_change' in ml_predictions:
            lstm_change = ml_predictions['lstm_price_change']
            signal['ml_signals']['lstm_predicted_change'] = lstm_change
            
            if abs(lstm_change) > 0.005:  # > 0.5% change
                if lstm_change > 0:
                    ml_score += 0.5
                else:
                    ml_score -= 0.5
                ml_confidence += min(abs(lstm_change) * 200, 0.5)
        
        # Process technical analysis signals (if available)
        ta_score = 0
        ta_confidence = 0
        
        if ta_results and timeframe in ta_results.get('timeframes', {}):
            tf_data = ta_results['timeframes'][timeframe]
            
            # Extract technical signals (adapt to your analyzer's output format)
            if 'signals' in tf_data:
                for signal_name, signal_data in tf_data['signals'].items():
                    if isinstance(signal_data, dict) and 'action' in signal_data:
                        action = signal_data['action']
                        confidence = signal_data.get('confidence', 0.5)
                        
                        if action == 'BUY':
                            ta_score += 1
                            ta_confidence += confidence
                        elif action == 'SELL':
                            ta_score -= 1
                            ta_confidence += confidence
        
        # Combine scores
        total_signals = 0
        if ml_score != 0:
            total_signals += 1
        if ta_score != 0:
            total_signals += 1
        
        if total_signals > 0:
            # Weight ML predictions higher for shorter timeframes, TA for longer
            if timeframe in ['5m', '15m']:
                ml_weight, ta_weight = 0.7, 0.3
            elif timeframe in ['1h', '4h']:
                ml_weight, ta_weight = 0.6, 0.4
            else:  # 1d and above
                ml_weight, ta_weight = 0.4, 0.6
            
            combined_score = (ml_score * ml_weight + ta_score * ta_weight)
            combined_confidence = (ml_confidence * ml_weight + ta_confidence * ta_weight)
            
            signal['combined_score'] = combined_score
            signal['confidence'] = min(combined_confidence, 1.0)
            
            # Generate action based on combined score and confidence
            if combined_confidence > 0.6:  # High confidence threshold
                if combined_score > 0.5:
                    signal['action'] = 'BUY'
                elif combined_score < -0.5:
                    signal['action'] = 'SELL'
                else:
                    signal['action'] = 'HOLD'
            else:
                signal['action'] = 'HOLD'  # Low confidence = hold
        
        return signal
    
    def generate_overall_recommendation(self, symbol_results: Dict) -> Dict:
        """Generate overall recommendation based on all timeframes"""
        recommendation = {
            'action': 'HOLD',
            'confidence': 0,
            'reasoning': [],
            'risk_level': 'MEDIUM',
            'timeframe_consensus': {}
        }
        
        combined_signals = symbol_results.get('combined_signals', {})
        
        if not combined_signals:
            recommendation['reasoning'].append("No signals available")
            return recommendation
        
        # Analyze consensus across timeframes
        buy_signals = 0
        sell_signals = 0
        total_confidence = 0
        
        for timeframe, signal in combined_signals.items():
            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 0)
            
            recommendation['timeframe_consensus'][timeframe] = {
                'action': action,
                'confidence': confidence
            }
            
            if action == 'BUY':
                buy_signals += 1
                total_confidence += confidence
            elif action == 'SELL':
                sell_signals += 1
                total_confidence += confidence
        
        total_signals = buy_signals + sell_signals
        
        if total_signals > 0:
            avg_confidence = total_confidence / total_signals
            
            # Determine overall action
            if buy_signals > sell_signals and avg_confidence > 0.6:
                recommendation['action'] = 'BUY'
                recommendation['reasoning'].append(f"Bullish consensus: {buy_signals}/{len(combined_signals)} timeframes")
            elif sell_signals > buy_signals and avg_confidence > 0.6:
                recommendation['action'] = 'SELL'
                recommendation['reasoning'].append(f"Bearish consensus: {sell_signals}/{len(combined_signals)} timeframes")
            else:
                recommendation['action'] = 'HOLD'
                recommendation['reasoning'].append("Mixed or weak signals")
            
            recommendation['confidence'] = avg_confidence
            
            # Determine risk level
            if avg_confidence > 0.8:
                recommendation['risk_level'] = 'LOW'
            elif avg_confidence > 0.6:
                recommendation['risk_level'] = 'MEDIUM'
            else:
                recommendation['risk_level'] = 'HIGH'
        
        return recommendation
    
    def run_full_analysis(self, symbols: List[str] = None, 
                         timeframes: List[str] = None) -> Dict:
        """Run full analysis on multiple symbols"""
        if symbols is None:
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        if timeframes is None:
            timeframes = ['15m', '1h', '4h', '1d']
        
        logging.info(f"🚀 Running full ML-enhanced analysis")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'symbols': {},
            'summary': {
                'total_symbols': len(symbols),
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0
            }
        }
        
        for symbol in symbols:
            try:
                symbol_result = self.analyze_symbol(symbol, timeframes)
                results['symbols'][symbol] = symbol_result
                
                # Update summary
                action = symbol_result['recommendation']['action']
                if action == 'BUY':
                    results['summary']['buy_signals'] += 1
                elif action == 'SELL':
                    results['summary']['sell_signals'] += 1
                else:
                    results['summary']['hold_signals'] += 1
                    
            except Exception as e:
                logging.error(f"❌ Analysis failed for {symbol}: {e}")
                results['symbols'][symbol] = {'error': str(e)}
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict):
        """Save analysis results to file"""
        try:
            os.makedirs('ml_predictions', exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ml_predictions/ml_analysis_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Also save as latest
            with open('ml_predictions/latest_ml_analysis.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            logging.info(f"💾 ML analysis results saved: {filename}")
            
        except Exception as e:
            logging.error(f"❌ Failed to save results: {e}")
    
    def train_models_for_symbols(self, symbols: List[str], timeframes: List[str]):
        """Train ML models for multiple symbols and timeframes"""
        logging.info("🎓 Training ML models for all symbols...")
        
        for symbol in symbols:
            for timeframe in timeframes:
                logging.info(f"Training models for {symbol} {timeframe}")
                
                try:
                    # Train price prediction models
                    self.ml_system.train_price_prediction_models(symbol, timeframe)
                    
                    # Train direction prediction models
                    self.ml_system.train_direction_prediction_models(symbol, timeframe)
                    
                    # Train LSTM model
                    self.ml_system.train_lstm_model(symbol, timeframe)
                    
                except Exception as e:
                    logging.error(f"❌ Model training failed for {symbol} {timeframe}: {e}")
        
        logging.info("✅ Model training completed")

def main():
    """Main function for ML-enhanced analysis"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize analyzer
    analyzer = MLEnhancedAnalyzer()
    
    # Define symbols and timeframes
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    timeframes = ['15m', '1h', '4h']
    
    print("🧠 ML-ENHANCED CRYPTO ANALYZER")
    print("=" * 50)
    
    # Check if we should train models
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        print("🎓 Training ML models...")
        analyzer.train_models_for_symbols(symbols, timeframes)
        print("✅ Model training completed")
        return
    
    # Run analysis
    print("🔍 Running ML-enhanced analysis...")
    results = analyzer.run_full_analysis(symbols, timeframes)
    
    # Display summary
    print("\n📊 ANALYSIS SUMMARY")
    print("=" * 30)
    summary = results['summary']
    print(f"Total Symbols: {summary['total_symbols']}")
    print(f"🟢 Buy Signals: {summary['buy_signals']}")
    print(f"🔴 Sell Signals: {summary['sell_signals']}")
    print(f"🟡 Hold Signals: {summary['hold_signals']}")
    
    # Display individual recommendations
    print("\n🎯 RECOMMENDATIONS")
    print("=" * 30)
    for symbol, data in results['symbols'].items():
        if 'error' in data:
            print(f"{symbol}: ❌ Error - {data['error']}")
            continue
            
        rec = data['recommendation']
        action = rec['action']
        confidence = rec['confidence']
        risk = rec['risk_level']
        
        emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
        print(f"{symbol}: {emoji} {action} (Confidence: {confidence:.2f}, Risk: {risk})")
        
        if rec['reasoning']:
            print(f"   Reasoning: {', '.join(rec['reasoning'])}")

if __name__ == "__main__":
    main()