
import logging
import sys
from optimized_ml_system import OptimizedCryptoMLSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def verify_ensemble_output():
    print("\n🔍 Verifying Ensemble Weight Logic...")
    
    ml_system = OptimizedCryptoMLSystem()
    symbol = 'BTC/USDT'
    timeframe = '15m'
    
    # Check if we have dynamic weights stored
    if f"{symbol}_{timeframe}_direction" in ml_system.model_performance:
        perf = ml_system.model_performance[f"{symbol}_{timeframe}_direction"]
        if 'dynamic_weights' in perf:
            print(f"✅ Found stored dynamic weights: {perf['dynamic_weights']}")
        else:
            print("⚠️ No dynamic weights key in performance data")
    else:
         print("⚠️ No performance data loaded for this symbol/timeframe")

    # Run prediction
    print("\n🔮 Running make_ensemble_prediction...")
    result = ml_system.make_ensemble_prediction(symbol, timeframe)
    
    if not result:
        print("❌ Prediction returned empty result")
        return

    # Check for breakdown
    if 'model_breakdown' in result:
        print("\n✅ 'model_breakdown' key found in result:")
        breakdown = result['model_breakdown']
        for model, data in breakdown.items():
            print(f"   - {model}: {data}")
            
            # Verify weight matches dynamic weight if available
            weight = data.get('weight')
            print(f"     Weight used: {weight:.4f}")
    else:
        print("❌ 'model_breakdown' key NOT found in result")

if __name__ == "__main__":
    verify_ensemble_output()
