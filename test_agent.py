"""
Test script for Crypto Trading Agent
Quick validation that everything works
Saves detailed logs to logs/agent_tests/ folder
"""
import sys
import os
import logging
from datetime import datetime
import traceback

# Add project root to path
sys.path.insert(0, '.')

def setup_test_logging():
    """Setup logging for tests"""
    
    # Create logs directory
    log_dir = 'logs/agent_tests'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'test_agent_{timestamp}.log')
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Latest log
    latest_log = os.path.join(log_dir, 'latest_test.log')
    latest_handler = logging.FileHandler(latest_log, mode='w', encoding='utf-8')
    latest_handler.setLevel(logging.DEBUG)
    latest_handler.setFormatter(formatter)
    logger.addHandler(latest_handler)
    
    print(f"📝 Test log: {log_file}")
    print(f"📝 Latest: {latest_log}")
    
    return logger

def test_agent():
    """Test basic agent functionality"""
    
    # Setup logging
    logger = setup_test_logging()
    
    logger.info("="*70)
    logger.info("🧪 TESTING CRYPTO TRADING AGENT")
    logger.info("="*70)
    
    test_results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    # Test 1: Import agent
    test_results['total'] += 1
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Import CryptoTradingAgent")
    logger.info("="*70)
    
    try:
        from crypto_agent import CryptoTradingAgent
        logger.info("✅ Import successful")
        test_results['passed'] += 1
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        logger.error(traceback.format_exc())
        test_results['failed'] += 1
        test_results['errors'].append({
            'test': 'Import',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        return test_results
    
    # Test 2: Initialize agent
    test_results['total'] += 1
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Initialize Agent")
    logger.info("="*70)
    
    try:
        agent = CryptoTradingAgent(log_to_file=True)
        logger.info("✅ Agent initialized successfully")
        test_results['passed'] += 1
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
        logger.error(traceback.format_exc())
        test_results['failed'] += 1
        test_results['errors'].append({
            'test': 'Initialize Agent',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        return test_results
    
    # Test 3: Database connection
    test_results['total'] += 1
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Database Connection")
    logger.info("="*70)
    
    try:
        conn = agent.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 5")
        tables = cursor.fetchall()
        conn.close()
        
        logger.info(f"✅ Database connected - Found {len(tables)} tables")
        for table in tables:
            logger.info(f"   - {table[0]}")
        test_results['passed'] += 1
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.error(traceback.format_exc())
        test_results['failed'] += 1
        test_results['errors'].append({
            'test': 'Database Connection',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
    
    # Test 4: Get latest price
    test_results['total'] += 1
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Get Latest Price")
    logger.info("="*70)
    
    try:
        price = agent.db.get_latest_price('BTC/USDT', '1h')
        if price:
            logger.info(f"✅ Latest BTC/USDT price: ${price:,.2f}")
            test_results['passed'] += 1
        else:
            logger.warning("⚠️ No price data found (database might be empty)")
            test_results['passed'] += 1  # Not a failure, just no data
    except Exception as e:
        logger.error(f"❌ Get price failed: {e}")
        logger.error(traceback.format_exc())
        test_results['failed'] += 1
        test_results['errors'].append({
            'test': 'Get Latest Price',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
    
    # Test 5: Market overview
    test_results['total'] += 1
    logger.info("\n" + "="*70)
    logger.info("TEST 5: Market Overview")
    logger.info("="*70)
    
    try:
        overview = agent.get_market_overview()
        logger.info(f"✅ Market regime: {overview['market_regime']}")
        logger.info(f"✅ Risk level: {overview['risk_level']}")
        logger.info(f"✅ Symbols analyzed: {overview['symbols_analyzed']}")
        logger.info(f"✅ Top opportunities: {len(overview['top_opportunities'])}")
        
        if overview['top_opportunities']:
            logger.info("\n   Top 3 opportunities:")
            for i, opp in enumerate(overview['top_opportunities'][:3], 1):
                logger.info(f"   {i}. {opp['symbol']} {opp['timeframe']} - "
                          f"{opp['recommendation']} (Quality: {opp['quality_score']}/100)")
        
        test_results['passed'] += 1
    except Exception as e:
        logger.error(f"❌ Market overview failed: {e}")
        logger.error(traceback.format_exc())
        test_results['failed'] += 1
        test_results['errors'].append({
            'test': 'Market Overview',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
    
    # Test 6: Single analysis (BTC/USDT 4h)
    test_results['total'] += 1
    logger.info("\n" + "="*70)
    logger.info("TEST 6: Analyze BTC/USDT 4h")
    logger.info("="*70)
    
    try:
        analysis = agent.analyze_trading_opportunity('BTC/USDT', '4h', save_recommendation=False)
        
        logger.info(f"✅ Recommendation: {analysis['recommendation']}")
        logger.info(f"✅ Quality score: {analysis['quality_score']}/100 ({analysis['quality_grade']})")
        logger.info(f"✅ Confidence: {analysis['confidence']:.0%}")
        logger.info(f"✅ Should trade: {analysis['should_trade']}")
        logger.info(f"✅ Position sizing: {analysis['position_sizing']}")
        
        # Display formatted output
        formatted = agent.format_recommendation(analysis)
        logger.info("\n" + "="*70)
        logger.info("FORMATTED RECOMMENDATION:")
        logger.info("="*70)
        logger.info("\n" + formatted)
        
        test_results['passed'] += 1
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        logger.error(traceback.format_exc())
        test_results['failed'] += 1
        test_results['errors'].append({
            'test': 'Analyze BTC/USDT 4h',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
    
    # Test 7: Test with different symbol
    test_results['total'] += 1
    logger.info("\n" + "="*70)
    logger.info("TEST 7: Analyze ETH/USDT 1h")
    logger.info("="*70)
    
    try:
        analysis = agent.analyze_trading_opportunity('ETH/USDT', '1h', save_recommendation=False)
        
        logger.info(f"✅ Recommendation: {analysis['recommendation']}")
        logger.info(f"✅ Quality score: {analysis['quality_score']}/100")
        logger.info(f"✅ Should trade: {analysis['should_trade']}")
        
        test_results['passed'] += 1
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        logger.error(traceback.format_exc())
        test_results['failed'] += 1
        test_results['errors'].append({
            'test': 'Analyze ETH/USDT 1h',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
    
    # Test 8: Performance report (may have no data)
    test_results['total'] += 1
    logger.info("\n" + "="*70)
    logger.info("TEST 8: Performance Report")
    logger.info("="*70)
    
    try:
        report = agent.get_performance_report('BTC/USDT', '4h', days_back=30)
        
        stats = report['statistics']
        logger.info(f"✅ Performance report generated")
        logger.info(f"   Total recommendations: {stats.get('total_recommendations', 0)}")
        
        if stats.get('total_recommendations', 0) > 0:
            logger.info(f"   Win rate: {stats.get('win_rate_4h', 0):.0%}")
            logger.info(f"   Avg return: {stats.get('avg_return_4h', 0):+.2%}")
        else:
            logger.info(f"   {stats.get('note', 'No historical data')}")
        
        test_results['passed'] += 1
    except Exception as e:
        logger.error(f"❌ Performance report failed: {e}")
        logger.error(traceback.format_exc())
        test_results['failed'] += 1
        test_results['errors'].append({
            'test': 'Performance Report',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
    
    # Final Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    logger.info(f"\nTotal Tests: {test_results['total']}")
    logger.info(f"✅ Passed: {test_results['passed']}")
    logger.info(f"❌ Failed: {test_results['failed']}")
    logger.info(f"Success Rate: {(test_results['passed']/test_results['total']*100):.1f}%")
    
    # Print errors if any
    if test_results['errors']:
        logger.info("\n" + "="*70)
        logger.info("ERRORS DETAILS")
        logger.info("="*70)
        for i, error in enumerate(test_results['errors'], 1):
            logger.error(f"\n{i}. {error['test']}")
            logger.error(f"   Error: {error['error']}")
            logger.error(f"   Traceback:\n{error['traceback']}")
    
    logger.info("\n" + "="*70)
    
    if test_results['failed'] == 0:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("="*70)
        return True
    else:
        logger.error("⚠️ SOME TESTS FAILED")
        logger.error("="*70)
        return False


def main():
    """Main entry point"""
    
    print("\n" + "="*70)
    print("🧪 CRYPTO TRADING AGENT - TEST SUITE")
    print("="*70)
    print()
    
    try:
        success = test_agent()
        
        print("\n" + "="*70)
        if success:
            print("✅ Test suite completed successfully!")
            print("="*70)
            print("\n💡 Next steps:")
            print("   1. Review logs in logs/agent_tests/latest_test.log")
            print("   2. Run interactive agent: python -m crypto_agent.agent")
            print("   3. Or use: from crypto_agent import CryptoTradingAgent")
            sys.exit(0)
        else:
            print("❌ Test suite completed with errors")
            print("="*70)
            print("\n📝 Check detailed logs in logs/agent_tests/latest_test.log")
            print("   Copy the log file content and share it for debugging")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print(traceback.format_exc())
        print("\n📝 Check logs in logs/agent_tests/ for details")
        sys.exit(1)


if __name__ == "__main__":
    main()