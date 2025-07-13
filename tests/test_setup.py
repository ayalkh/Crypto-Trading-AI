"""
First test script for Crypto Trading AI
This will test if everything is working
"""

print("🚀 Crypto Trading AI - Setup Test")
print("=" * 40)

# Test 1: Basic Python
print("✅ Python is working!")

# Test 2: Try importing basic libraries
try:
    import datetime
    print("✅ Datetime module working")
    print(f"   Current time: {datetime.datetime.now()}")
except ImportError:
    print("❌ Datetime not working")

# Test 3: Try pandas (if installed)
try:
    import pandas as pd
    print("✅ Pandas is working")
    
    # Create a simple test dataframe
    test_data = {
        'crypto': ['BTC', 'ETH', 'BNB'],
        'price': [67000, 3500, 310]
    }
    df = pd.DataFrame(test_data)
    print("   Sample crypto data:")
    print(df)
except ImportError:
    print("❌ Pandas not installed yet - we'll install it later")

# Test 4: Try requests (for API calls)
try:
    import requests
    print("✅ Requests library working")
except ImportError:
    print("❌ Requests not installed yet - we'll install it later")

# Test 5: Simple crypto price test (without libraries)
print("\n💰 Sample crypto prices (hardcoded for now):")
crypto_prices = {
    'BTC': 67234.50,
    'ETH': 3456.78,
    'BNB': 310.25
}

for crypto, price in crypto_prices.items():
    print(f"   {crypto}: ${price:,.2f}")

print("\n🎯 Next steps:")
print("1. Install required libraries")
print("2. Set up data collection")
print("3. Connect to real crypto APIs")
print("\n🎉 Basic setup test complete!")

