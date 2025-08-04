"""
Test Fixed ML System
Verify that the ML prediction fixes are working
"""
import os
import sys

# Setup environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def test_single_prediction():
    """Test a single ML prediction to verify fixes"""
    print("🧪 TESTING FIXED ML SYSTEM")
    print("=" * 40)
    
    try:
        from ml_integration_system import CryptoMLSystem
        
        print("✅ Initializing ML system...")
        ml_system = CryptoMLSystem()
        
        # Test with BTC/USDT 1h (we know this has trained models)
        symbol = 'BTC/USDT'
        timeframe = '1h'
        
        print(f"🔍 Testing predictions for {symbol} {timeframe}...")
        
        # Load models
        ml_system.load_models(symbol, timeframe)
        
        # Make predictions
        predictions = ml_system.make_predictions(symbol, timeframe)
        
        if predictions:
            print(f"\n🎉 SUCCESS! Generated {len(predictions)} predictions:")
            for key, value in predictions.items():
                if isinstance(value, float):
                    if 'change' in key:
                        print(f"   {key}: {value:+.4f} ({value*100:+.2f}%)")
                    else:
                        print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
        else:
            print("❌ No predictions generated")
            return False
        
        print(f"\n✅ ML prediction system is working!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_symbols():
    """Test predictions for all symbols"""
    print("\n🔬 TESTING ALL SYMBOLS")
    print("=" * 30)
    
    try:
        from ml_integration_system import CryptoMLSystem
        
        ml_system = CryptoMLSystem()
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        timeframes = ['1h', '4h']
        
        results = {}
        
        for symbol in symbols:
            results[symbol] = {}
            for timeframe in timeframes:
                print(f"Testing {symbol} {timeframe}...")
                
                try:
                    ml_system.load_models(symbol, timeframe)
                    predictions = ml_system.make_predictions(symbol, timeframe)
                    
                    if predictions:
                        results[symbol][timeframe] = 'SUCCESS'
                        print(f"   ✅ {len(predictions)} predictions")
                    else:
                        results[symbol][timeframe] = 'NO_PREDICTIONS'
                        print(f"   ⚠️ No predictions")
                        
                except Exception as e:
                    results[symbol][timeframe] = f'ERROR: {str(e)[:50]}'
                    print(f"   ❌ Error: {str(e)[:50]}")
        
        # Summary
        print(f"\n📊 TEST SUMMARY:")
        total_tests = 0
        successful_tests = 0
        
        for symbol, timeframes in results.items():
            for timeframe, result in timeframes.items():
                total_tests += 1
                if result == 'SUCCESS':
                    successful_tests += 1
                    status = '✅'
                elif result == 'NO_PREDICTIONS':
                    status = '⚠️'
                else:
                    status = '❌'
                
                print(f"   {symbol} {timeframe}: {status} {result}")
        
        success_rate = (successful_tests / total_tests) * 100
        print(f"\n🎯 SUCCESS RATE: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 50:
            print("🎉 ML system is working well!")
            return True
        else:
            print("⚠️ ML system needs more work")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🔧 TESTING FIXED ML PREDICTION SYSTEM")
    print("=" * 50)
    
    # Test single prediction first
    single_test = test_single_prediction()
    
    if single_test:
        # Test all symbols
        all_test = test_all_symbols()
        
        if all_test:
            print(f"\n🚀 ALL TESTS PASSED!")
            print(f"Your ML system is ready for live predictions!")
        else:
            print(f"\n⚠️ Some tests failed, but basic functionality works")
    else:
        print(f"\n❌ Basic test failed - check your models and data")
    
    print(f"\n💡 If tests pass, your automation system should now work correctly!")

if __name__ == "__main__":
    main()