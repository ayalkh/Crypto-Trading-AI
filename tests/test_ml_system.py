"""
Quick test script to verify ML system is working
"""
import os
import sys

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def test_imports():
    """Test all ML imports"""
    print("🧪 Testing ML imports...")
    
    try:
        import numpy as np
        print("   ✅ numpy")
        
        import pandas as pd
        print("   ✅ pandas")
        
        import sklearn
        print("   ✅ scikit-learn")
        
        import tensorflow as tf
        print("   ✅ tensorflow")
        
        import xgboost as xgb
        print("   ✅ xgboost")
        
        import joblib
        print("   ✅ joblib")
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_ml_system():
    """Test ML system initialization"""
    print("\n🔧 Testing ML system...")
    
    try:
        from ml_integration_system import CryptoMLSystem
        ml_system = CryptoMLSystem()
        print("   ✅ ML system initialized")
        return True
        
    except Exception as e:
        print(f"   ❌ ML system failed: {e}")
        return False

def test_ml_analyzer():
    """Test ML analyzer"""
    print("\n🔍 Testing ML analyzer...")
    
    try:
        from ml_enhanced_analyzer import MLEnhancedAnalyzer
        analyzer = MLEnhancedAnalyzer()
        print("   ✅ ML analyzer initialized")
        return True
        
    except Exception as e:
        print(f"   ❌ ML analyzer failed: {e}")
        return False

def main():
    print("🧠 ML SYSTEM TEST")
    print("=" * 30)
    
    # Test imports
    if not test_imports():
        return False
    
    # Test ML system
    if not test_ml_system():
        return False
    
    # Test ML analyzer
    if not test_ml_analyzer():
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("Your ML system is ready to go!")
    return True

if __name__ == "__main__":
    main()