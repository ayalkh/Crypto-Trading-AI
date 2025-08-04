"""
Test Data Loading for ML System
Verify that ML system can load data from your database
"""
import os
import sys

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def test_ml_data_loading():
    """Test if ML system can load data from your database"""
    print("🧪 TESTING ML DATA LOADING")
    print("=" * 40)
    
    try:
        # Import ML system
        from ml_integration_system import CryptoMLSystem
        
        # Initialize ML system
        print("🔧 Initializing ML system...")
        ml_system = CryptoMLSystem()
        print(f"✅ ML system initialized, using table: {ml_system.table_name}")
        
        # Test symbols and timeframes
        test_cases = [
            ('BTC/USDT', '5m'),
            ('BTC/USDT', '15m'), 
            ('BTC/USDT', '1h'),
            ('ETH/USDT', '5m'),
            ('BNB/USDT', '5m')
        ]
        
        print(f"\n📊 Testing data loading for {len(test_cases)} cases...")
        
        successful_loads = 0
        total_records = 0
        
        for symbol, timeframe in test_cases:
            print(f"\n🔍 Testing {symbol} {timeframe}...")
            
            # Try to load data
            df = ml_system.load_data(symbol, timeframe, days_back=7)
            
            if df.empty:
                print(f"   ⚠️ No data found for {symbol} {timeframe}")
            else:
                records = len(df)
                total_records += records
                successful_loads += 1
                print(f"   ✅ Loaded {records} records")
                
                # Show sample data
                if records > 0:
                    latest = df.iloc[-1]
                    print(f"   📈 Latest: {latest['close']:.2f} (Volume: {latest['volume']:.2f})")
        
        # Summary
        print(f"\n📊 LOADING TEST RESULTS")
        print(f"   Successful loads: {successful_loads}/{len(test_cases)}")
        print(f"   Total records: {total_records}")
        
        if successful_loads > 0:
            print(f"   🎉 SUCCESS! ML system can load your data")
            return True
        else:
            print(f"   ❌ FAILED! No data could be loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error testing data loading: {e}")
        return False

def test_feature_creation():
    """Test feature creation on loaded data"""
    print(f"\n🔧 TESTING FEATURE CREATION")
    print("=" * 40)
    
    try:
        from ml_integration_system import CryptoMLSystem
        
        ml_system = CryptoMLSystem()
        
        # Load some data
        print("📊 Loading BTC/USDT 1h data...")
        df = ml_system.load_data('BTC/USDT', '1h', days_back=7)
        
        if df.empty:
            print("❌ No data to test features")
            return False
        
        print(f"✅ Loaded {len(df)} records")
        
        # Create features
        print("🔧 Creating ML features...")
        df_features = ml_system.create_features(df)
        
        if df_features.empty:
            print("❌ Feature creation failed")
            return False
        
        feature_count = len(df_features.columns)
        sample_count = len(df_features)
        
        print(f"✅ Created {feature_count} features from {sample_count} samples")
        
        # Show some features
        print(f"📊 Sample features:")
        for i, col in enumerate(df_features.columns[:10]):
            print(f"   {i+1}. {col}")
        if feature_count > 10:
            print(f"   ... and {feature_count - 10} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing features: {e}")
        return False

def main():
    print("🧪 ML DATA LOADING TEST")
    print("=" * 50)
    
    # Test data loading
    loading_ok = test_ml_data_loading()
    
    if loading_ok:
        # Test feature creation
        features_ok = test_feature_creation()
        
        if features_ok:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"Your ML system is ready for training!")
            print(f"\nNext step: python train_initial_models.py")
        else:
            print(f"\n⚠️ Data loading works, but feature creation failed")
    else:
        print(f"\n❌ Data loading failed")
        print(f"Check your database and data collector")

if __name__ == "__main__":
    main()