"""
Main Agent Orchestration
Coordinates all tools and manages conversation flow
"""
import logging
import os
import sys
from typing import Dict, List, Optional
from datetime import datetime

from .database import AgentDatabase
from .tools import (
    SmartConsensusAnalyzer,
    TradeQualityScorer,
    MarketContextAnalyzer,
    PredictionOutcomeTracker
)
from .prompts import SYSTEM_PROMPT
from .config import AGENT_CONFIG, SYMBOLS, TIMEFRAMES


class CryptoTradingAgent:
    """
    Main Crypto Trading Agent
    
    Orchestrates all tools to provide intelligent trading recommendations
    """
    
    def __init__(self, log_to_file: bool = True):
        """Initialize the trading agent"""
        
        # Setup logging FIRST
        self._setup_logging(log_to_file)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Initializing Crypto Trading Agent...")
        
        # Initialize database
        self.db = AgentDatabase()
        
        # Initialize tools
        self.consensus_analyzer = SmartConsensusAnalyzer(self.db)
        self.quality_scorer = TradeQualityScorer(self.db)
        self.market_analyzer = MarketContextAnalyzer(self.db)
        self.outcome_tracker = PredictionOutcomeTracker(self.db)
        
        # Agent state
        self.conversation_history = []
        self.last_recommendations = {}  # Track recent recommendations
        
        self.logger.info(f"✅ Agent '{AGENT_CONFIG['name']}' v{AGENT_CONFIG['version']} ready!")
        self.logger.info(f"📊 Monitoring {len(SYMBOLS)} symbols across {len(TIMEFRAMES)} timeframes")
    
    def _setup_logging(self, log_to_file: bool):
        """Setup logging configuration"""
        
        # Create logs directory
        log_dir = 'logs/agent'
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers = []
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler (always enabled)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if log_to_file:
            # Create timestamped log file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'agent_{timestamp}.log')
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # More detailed in file
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Also create a "latest.log" that always points to most recent
            latest_log = os.path.join(log_dir, 'latest.log')
            latest_handler = logging.FileHandler(latest_log, mode='w', encoding='utf-8')
            latest_handler.setLevel(logging.DEBUG)
            latest_handler.setFormatter(formatter)
            logger.addHandler(latest_handler)
            
            print(f"📝 Logging to: {log_file}")
            print(f"📝 Latest log: {latest_log}")
    
    def analyze_trading_opportunity(self, symbol: str, timeframe: str,
                                   save_recommendation: bool = True) -> Dict:
        """
        Comprehensive analysis of a trading opportunity
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h')
            save_recommendation: Whether to save recommendation to database
            
        Returns:
            Dict with complete analysis and recommendation
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🎯 Analyzing Trading Opportunity: {symbol} {timeframe}")
        self.logger.info(f"{'='*70}")
        
        try:
            # Step 1: Get Smart Consensus
            self.logger.info(f"🔄 Step 1/4: Running Smart Consensus Analyzer...")
            consensus = self.consensus_analyzer.analyze(symbol, timeframe)
            self.logger.info(f"✅ Step 1/4: Consensus Analysis Complete")
            
            # Step 2: Score Trade Quality
            self.logger.info(f"🔄 Step 2/4: Running Trade Quality Scorer...")
            quality = self.quality_scorer.score(symbol, timeframe, consensus)
            self.logger.info(f"✅ Step 2/4: Quality Scoring Complete")
            
            # Step 3: Get Market Context
            self.logger.info(f"🔄 Step 3/4: Running Market Context Analyzer...")
            market_context = self.market_analyzer.analyze()  # Uses BTC as market leader
            self.logger.info(f"✅ Step 3/4: Market Context Analysis Complete")
            
            # Step 4: Adjust recommendation based on context
            self.logger.info(f"🔄 Step 4/4: Synthesizing Final Recommendation...")
            final_recommendation = self._synthesize_recommendation(
                symbol, timeframe, consensus, quality, market_context
            )
            self.logger.info(f"✅ Step 4/4: Final Recommendation Generated")
            
            # Generate comprehensive result
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'recommendation': final_recommendation['action'],
                'confidence': final_recommendation['confidence'],
                'quality_score': quality['quality_score'],
                'quality_grade': quality['grade'],
                'consensus': consensus,
                'quality_details': quality,
                'market_context': market_context,
                'position_sizing': final_recommendation['position_sizing'],
                'stop_loss': final_recommendation.get('stop_loss'),
                'take_profit': final_recommendation.get('take_profit'),
                'reasoning': final_recommendation['reasoning'],
                'risk_factors': final_recommendation['risk_factors'],
                'should_trade': final_recommendation['should_trade']
            }
            
            # Save recommendation if requested
            if save_recommendation and final_recommendation['should_trade']:
                self.outcome_tracker.log_recommendation(
                    symbol=symbol,
                    timeframe=timeframe,
                    recommendation=final_recommendation['action'],
                    confidence=final_recommendation['confidence'],
                    quality_score=quality['quality_score'],
                    consensus=consensus,
                    market_regime=market_context['regime'],
                    reasoning=final_recommendation['reasoning']
                )
            
            # Track in memory
            self.last_recommendations[f"{symbol}_{timeframe}"] = {
                'timestamp': datetime.now(),
                'recommendation': final_recommendation['action'],
                'quality_score': quality['quality_score']
            }
            
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"🎯 FINAL: {final_recommendation['action']} {symbol} {timeframe}")
            self.logger.info(f"   Quality: {quality['quality_score']}/100 ({quality['grade']})")
            self.logger.info(f"   Confidence: {final_recommendation['confidence']:.0%}")
            self.logger.info(f"   Should Trade: {final_recommendation['should_trade']}")
            self.logger.info(f"{'='*70}\n")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing {symbol} {timeframe}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _synthesize_recommendation(self, symbol: str, timeframe: str,
                                   consensus: Dict, quality: Dict,
                                   market_context: Dict) -> Dict:
        """
        Synthesize final recommendation from all analyses
        
        Applies business logic to determine final action
        """
        
        # Get base recommendation from consensus
        base_action = consensus['recommendation']
        base_confidence = consensus['confidence']
        quality_score = quality['quality_score']
        
        # Get current price
        current_price = self.db.get_latest_price(symbol, timeframe)
        
        # Apply market context adjustments
        regime = market_context['regime']
        position_adjustment = market_context['recommendations']['position_adjustment']
        
        # Decision logic
        should_trade = True
        adjusted_action = base_action
        adjusted_confidence = base_confidence
        risk_factors = []
        
        # Rule 1: Quality threshold
        if quality_score < 60:
            should_trade = False
            adjusted_action = 'HOLD'
            risk_factors.append(f"Quality score too low ({quality_score}/100, need 60+)")
        
        # Rule 2: Confidence threshold
        if base_confidence < 0.50:
            should_trade = False
            adjusted_action = 'HOLD'
            risk_factors.append(f"Low confidence ({base_confidence:.0%})")
        
        # Rule 3: High volatility regime
        if regime == 'High Volatility':
            position_adjustment *= 0.5
            risk_factors.append("High volatility - reduced position sizing by 50%")
            if quality_score < 75:  # Higher bar in volatile markets
                should_trade = False
                adjusted_action = 'HOLD'
                risk_factors.append("Volatility too high for this quality level")
        
        # Rule 4: Regime-specific adjustments
        if regime == 'Trending Bull' and base_action in ['SELL', 'STRONG_SELL']:
            adjusted_confidence *= 0.8
            risk_factors.append("Selling in bull trend - increased risk of false reversal")
            if quality_score < 80:
                should_trade = False
                adjusted_action = 'HOLD'
        
        elif regime == 'Trending Bear' and base_action in ['BUY', 'STRONG_BUY']:
            adjusted_confidence *= 0.8
            risk_factors.append("Buying in bear trend - increased risk")
            if quality_score < 80:
                should_trade = False
                adjusted_action = 'HOLD'
        
        # Rule 5: Overtrading check
        recent_trades = self._check_recent_trades(symbol, timeframe)
        if recent_trades >= 3:
            should_trade = False
            adjusted_action = 'HOLD'
            risk_factors.append(f"Too many recent trades ({recent_trades} in last 24h)")
        
        # Rule 6: Model disagreement
        models_used = consensus.get('models_used', [])
        if len(models_used) >= 3:
            directions = [m['direction'] for m in models_used]
            agreement_rate = directions.count(directions[0]) / len(directions) if directions else 0
            if agreement_rate < 0.6:
                risk_factors.append(f"Model disagreement ({agreement_rate:.0%} consensus)")
                adjusted_confidence *= 0.9
        
        # Calculate position sizing
        position_min, position_max = quality['position_size_pct']
        position_min *= position_adjustment
        position_max *= position_adjustment
        
        # Calculate stop loss and take profit
        stop_loss_pct = self._calculate_stop_loss(timeframe, quality_score)
        take_profit_pct = self._calculate_take_profit(timeframe, quality_score, consensus)
        
        stop_loss_price = None
        take_profit_price = None
        
        if current_price and should_trade:
            if adjusted_action in ['BUY', 'STRONG_BUY']:
                stop_loss_price = current_price * (1 - stop_loss_pct)
                take_profit_price = current_price * (1 + take_profit_pct)
            elif adjusted_action in ['SELL', 'STRONG_SELL']:
                stop_loss_price = current_price * (1 + stop_loss_pct)
                take_profit_price = current_price * (1 - take_profit_pct)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            consensus, quality, market_context, 
            adjusted_action, should_trade
        )
        
        # Add quality strengths/weaknesses to risk factors
        if quality.get('weaknesses'):
            risk_factors.extend([f"Weak: {w}" for w in quality['weaknesses'][:2]])
        
        return {
            'action': adjusted_action,
            'confidence': adjusted_confidence,
            'should_trade': should_trade,
            'position_sizing': (round(position_min, 1), round(position_max, 1)),
            'stop_loss': stop_loss_price,
            'stop_loss_pct': stop_loss_pct,
            'take_profit': take_profit_price,
            'take_profit_pct': take_profit_pct,
            'reasoning': reasoning,
            'risk_factors': risk_factors,
            'regime_adjustment': position_adjustment
        }
    
    def _check_recent_trades(self, symbol: str, timeframe: str) -> int:
        """Check how many trades were made recently"""
        key = f"{symbol}_{timeframe}"
        
        if key not in self.last_recommendations:
            return 0
        
        last_rec = self.last_recommendations[key]
        hours_ago = (datetime.now() - last_rec['timestamp']).total_seconds() / 3600
        
        if hours_ago < 24:
            # Query database for recent trades
            historical = self.db.get_historical_signals(symbol, timeframe, days_back=1)
            return len(historical) if not historical.empty else 0
        
        return 0
    
    def _calculate_stop_loss(self, timeframe: str, quality_score: int) -> float:
        """Calculate stop loss percentage based on timeframe and quality"""
        
        # Base stop loss by timeframe
        base_stops = {
            '5m': 0.015,   # 1.5%
            '15m': 0.020,  # 2.0%
            '1h': 0.025,   # 2.5%
            '4h': 0.035,   # 3.5%
            '1d': 0.050    # 5.0%
        }
        
        base = base_stops.get(timeframe, 0.03)
        
        # Tighter stops for higher quality
        if quality_score >= 85:
            return base * 0.9
        elif quality_score >= 75:
            return base
        else:
            return base * 1.1
    
    def _calculate_take_profit(self, timeframe: str, quality_score: int,
                               consensus: Dict) -> float:
        """Calculate take profit percentage"""
        
        # Use predicted price change if available
        predicted_change = abs(consensus.get('weighted_price_change', 0))
        
        if predicted_change > 0.01:  # Use prediction if > 1%
            # Target 80% of predicted move (conservative)
            return predicted_change * 0.8
        
        # Otherwise use quality-based targets
        if quality_score >= 85:
            multiplier = 2.5  # 2.5x risk
        elif quality_score >= 75:
            multiplier = 2.0
        else:
            multiplier = 1.5
        
        stop_loss = self._calculate_stop_loss(timeframe, quality_score)
        return stop_loss * multiplier
    
    def _generate_reasoning(self, consensus: Dict, quality: Dict,
                           market_context: Dict, action: str,
                           should_trade: bool) -> str:
        """Generate human-readable reasoning"""
        
        if not should_trade:
            return f"Do not trade: {consensus.get('reasoning', 'Insufficient quality or confidence')}"
        
        parts = []
        
        # Consensus reasoning
        parts.append(consensus.get('reasoning', 'Model consensus'))
        
        # Quality highlights
        if quality.get('strengths'):
            parts.append(f"Strengths: {', '.join(quality['strengths'][:2])}")
        
        # Market context
        regime = market_context['regime']
        parts.append(f"Market: {regime}")
        
        return '. '.join(parts) + '.'
    
    def get_market_overview(self) -> Dict:
        """Get comprehensive market overview"""
        self.logger.info("🌍 Generating market overview...")
        
        market_context = self.market_analyzer.analyze()
        
        # Get top opportunities
        opportunities = self._scan_all_opportunities()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'market_regime': market_context['regime'],
            'market_characteristics': market_context['characteristics'],
            'recommendations': market_context['recommendations'],
            'risk_level': market_context['risk_level'],
            'top_opportunities': opportunities[:5],  # Top 5
            'symbols_analyzed': len(SYMBOLS)
        }
    
    def _scan_all_opportunities(self) -> List[Dict]:
        """Scan all symbols for trading opportunities"""
        
        opportunities = []
        
        # Focus on priority timeframes
        priority_timeframes = ['4h', '1h']
        
        for symbol in SYMBOLS:
            for timeframe in priority_timeframes:
                try:
                    # Quick analysis
                    consensus = self.consensus_analyzer.analyze(symbol, timeframe)
                    
                    if consensus['confidence'] > 0.6:
                        quality = self.quality_scorer.score(symbol, timeframe, consensus)
                        
                        if quality['quality_score'] >= 70:
                            opportunities.append({
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'recommendation': consensus['recommendation'],
                                'quality_score': quality['quality_score'],
                                'confidence': consensus['confidence']
                            })
                
                except Exception as e:
                    self.logger.warning(f"⚠️ Error analyzing {symbol} {timeframe}: {e}")
        
        # Sort by quality score
        opportunities.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return opportunities
    
    def get_performance_report(self, symbol: Optional[str] = None,
                              timeframe: Optional[str] = None,
                              days_back: int = 30) -> Dict:
        """Get performance report"""
        self.logger.info(f"📊 Generating performance report ({days_back} days)...")
        
        if symbol and timeframe:
            stats = self.outcome_tracker.get_performance_stats(
                symbol, timeframe, days_back
            )
            insights = self.outcome_tracker.get_insights(
                symbol, timeframe, days_back
            )
        else:
            # Overall performance (would need implementation)
            stats = {'note': 'Overall performance tracking not yet implemented'}
            insights = []
        
        return {
            'timestamp': datetime.now().isoformat(),
            'period_days': days_back,
            'symbol': symbol,
            'timeframe': timeframe,
            'statistics': stats,
            'insights': insights
        }
    
    def format_recommendation(self, analysis: Dict) -> str:
        """Format analysis as human-readable recommendation"""
        
        symbol = analysis['symbol']
        timeframe = analysis['timeframe']
        action = analysis['recommendation']
        confidence = analysis['confidence']
        quality = analysis['quality_score']
        grade = analysis['quality_grade']
        should_trade = analysis['should_trade']
        
        # Build output
        lines = []
        lines.append("=" * 70)
        lines.append(f"🎯 TRADING RECOMMENDATION: {symbol} {timeframe}")
        lines.append("=" * 70)
        
        # Main recommendation
        emoji = {
            'STRONG_BUY': '🟢🟢',
            'BUY': '🟢',
            'HOLD': '🟡',
            'SELL': '🔴',
            'STRONG_SELL': '🔴🔴'
        }.get(action, '⚪')
        
        lines.append(f"\n{emoji} Recommendation: {action}")
        lines.append(f"   Confidence: {confidence:.0%} | Quality: {quality}/100 ({grade})")
        
        if not should_trade:
            lines.append(f"\n⚠️  DO NOT TRADE - Conditions not met")
        
        # Current price
        current_price = self.db.get_latest_price(symbol, timeframe)
        if current_price:
            lines.append(f"   Current Price: ${current_price:,.2f}")
        
        # Key reasons
        lines.append(f"\n📊 Analysis:")
        lines.append(f"   {analysis['reasoning']}")
        
        # Risk factors
        if analysis['risk_factors']:
            lines.append(f"\n⚠️  Risk Factors:")
            for risk in analysis['risk_factors'][:3]:
                lines.append(f"   - {risk}")
        
        # Trading details (if should trade)
        if should_trade and analysis['stop_loss']:
            lines.append(f"\n💼 Trading Plan:")
            pos_min, pos_max = analysis['position_sizing']
            lines.append(f"   Position Size: {pos_min:.1f}% - {pos_max:.1f}% of portfolio")
            lines.append(f"   Stop Loss: ${analysis['stop_loss']:,.2f} ({analysis['stop_loss_pct']:.1%})")
            lines.append(f"   Take Profit: ${analysis['take_profit']:,.2f} ({analysis['take_profit_pct']:.1%})")
        
        # Market context
        market = analysis['market_context']
        lines.append(f"\n🌍 Market Context:")
        lines.append(f"   Regime: {market['regime']} ({market['confidence']:.0%} confidence)")
        lines.append(f"   Risk Level: {market['risk_level']}")
        
        # Quality breakdown (top strengths)
        if analysis['quality_details'].get('strengths'):
            lines.append(f"\n✅ Strengths:")
            for strength in analysis['quality_details']['strengths'][:3]:
                lines.append(f"   - {strength}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


def main():
    """Main function for interactive mode"""
    
    print("\n" + "="*70)
    print("🚀 CRYPTO TRADING AGENT - INTERACTIVE MODE")
    print("="*70)
    
    # Initialize agent
    agent = CryptoTradingAgent()
    
    print("\nAgent ready! Commands:")
    print("  analyze <SYMBOL> <TIMEFRAME>  - Analyze trading opportunity")
    print("  market                         - Get market overview")
    print("  performance <SYMBOL> <TF>      - Get performance stats")
    print("  scan                          - Scan all symbols for opportunities")
    print("  quit                          - Exit")
    print()
    
    while True:
        try:
            user_input = input("📊 Command: ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split()
            command = parts[0].lower()
            
            if command == 'quit':
                print("👋 Goodbye!")
                break
            
            elif command == 'analyze':
                if len(parts) < 3:
                    print("Usage: analyze <SYMBOL> <TIMEFRAME>")
                    print("Example: analyze BTC/USDT 4h")
                    continue
                
                symbol = parts[1].upper()
                if '/' not in symbol:
                    symbol = f"{symbol}/USDT"
                
                timeframe = parts[2].lower()
                
                # Run analysis
                analysis = agent.analyze_trading_opportunity(symbol, timeframe)
                
                # Display formatted recommendation
                print("\n" + agent.format_recommendation(analysis))
            
            elif command == 'market':
                overview = agent.get_market_overview()
                
                print("\n" + "="*70)
                print("🌍 MARKET OVERVIEW")
                print("="*70)
                print(f"\nRegime: {overview['market_regime']}")
                print(f"Risk Level: {overview['risk_level']}")
                print(f"\nTop Opportunities:")
                
                for i, opp in enumerate(overview['top_opportunities'], 1):
                    emoji = '🟢' if 'BUY' in opp['recommendation'] else '🔴' if 'SELL' in opp['recommendation'] else '🟡'
                    print(f"  {i}. {emoji} {opp['symbol']} {opp['timeframe']} - "
                          f"{opp['recommendation']} (Quality: {opp['quality_score']}/100)")
                
                print("="*70)
            
            elif command == 'performance':
                if len(parts) < 3:
                    print("Usage: performance <SYMBOL> <TIMEFRAME>")
                    continue
                
                symbol = parts[1].upper()
                if '/' not in symbol:
                    symbol = f"{symbol}/USDT"
                
                timeframe = parts[2].lower()
                
                report = agent.get_performance_report(symbol, timeframe, days_back=30)
                
                print("\n" + "="*70)
                print(f"📊 PERFORMANCE REPORT: {symbol} {timeframe}")
                print("="*70)
                
                stats = report['statistics']
                if stats.get('total_recommendations', 0) > 0:
                    print(f"\nTotal Recommendations: {stats['total_recommendations']}")
                    print(f"Completed (4h): {stats.get('completed_4h', 0)}")
                    print(f"Win Rate: {stats.get('win_rate_4h', 0):.0%}")
                    print(f"Average Return: {stats.get('avg_return_4h', 0):+.2%}")
                    
                    if report['insights']:
                        print(f"\n💡 Insights:")
                        for insight in report['insights']:
                            print(f"   - {insight}")
                else:
                    print(f"\n{stats.get('note', 'No data available')}")
                
                print("="*70)
            
            elif command == 'scan':
                print("\n🔍 Scanning all symbols for opportunities...")
                overview = agent.get_market_overview()
                
                print(f"\nFound {len(overview['top_opportunities'])} quality opportunities:")
                for i, opp in enumerate(overview['top_opportunities'][:10], 1):
                    emoji = '🟢' if 'BUY' in opp['recommendation'] else '🔴' if 'SELL' in opp['recommendation'] else '🟡'
                    print(f"  {i}. {emoji} {opp['symbol']} {opp['timeframe']} - "
                          f"{opp['recommendation']} "
                          f"(Quality: {opp['quality_score']}/100, "
                          f"Confidence: {opp['confidence']:.0%})")
            
            else:
                print(f"Unknown command: {command}")
                print("Type 'quit' to exit or use: analyze, market, performance, scan")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()