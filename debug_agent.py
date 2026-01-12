"""
Quick debug script to find where agent initialization hangs
"""
import sys
sys.path.insert(0, '.')

print("="*70)
print("🔍 DEBUG: Agent Initialization")
print("="*70)

print("\n1. Testing imports...")
try:
    from crypto_agent.config import DB_PATH, SYMBOLS, TIMEFRAMES
    print(f"   ✅ Config imported")
    print(f"   - DB Path: {DB_PATH}")
    print(f"   - Symbols: {SYMBOLS}")
except Exception as e:
    print(f"   ❌ Config import failed: {e}")
    sys.exit(1)

print("\n2. Testing database connection...")
try:
    import sqlite3
    import os
    
    if os.path.exists(DB_PATH):
        print(f"   ✅ Database file exists: {DB_PATH}")
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 5")
        tables = cursor.fetchall()
        conn.close()
        
        print(f"   ✅ Connected to database")
        print(f"   - Found {len(tables)} tables")
        for table in tables:
            print(f"     • {table[0]}")
    else:
        print(f"   ❌ Database file not found: {DB_PATH}")
        sys.exit(1)
        
except Exception as e:
    print(f"   ❌ Database connection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Testing AgentDatabase class...")
try:
    from crypto_agent.database import AgentDatabase
    print(f"   ✅ AgentDatabase imported")
    
    db = AgentDatabase()
    print(f"   ✅ AgentDatabase initialized")
    
    # Test a simple query
    price = db.get_latest_price('BTC/USDT', '1h')
    if price:
        print(f"   ✅ Query works - BTC price: ${price:,.2f}")
    else:
        print(f"   ⚠️  No data returned (might be empty database)")
        
except Exception as e:
    print(f"   ❌ AgentDatabase failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Testing tools import...")
try:
    from crypto_agent.tools import (
        SmartConsensusAnalyzer,
        TradeQualityScorer,
        MarketContextAnalyzer,
        PredictionOutcomeTracker
    )
    print(f"   ✅ All tools imported")
except Exception as e:
    print(f"   ❌ Tools import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n5. Testing tool initialization...")
try:
    db = AgentDatabase()
    
    print(f"   🔄 Initializing SmartConsensusAnalyzer...")
    consensus = SmartConsensusAnalyzer(db)
    print(f"   ✅ SmartConsensusAnalyzer OK")
    
    print(f"   🔄 Initializing TradeQualityScorer...")
    quality = TradeQualityScorer(db)
    print(f"   ✅ TradeQualityScorer OK")
    
    print(f"   🔄 Initializing MarketContextAnalyzer...")
    market = MarketContextAnalyzer(db)
    print(f"   ✅ MarketContextAnalyzer OK")
    
    print(f"   🔄 Initializing PredictionOutcomeTracker...")
    tracker = PredictionOutcomeTracker(db)
    print(f"   ✅ PredictionOutcomeTracker OK")
    
except Exception as e:
    print(f"   ❌ Tool initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n6. Testing full agent initialization...")
try:
    from crypto_agent import CryptoTradingAgent
    
    print(f"   🔄 Initializing CryptoTradingAgent (with logging)...")
    agent = CryptoTradingAgent(log_to_file=False)  # Disable file logging for debug
    print(f"   ✅ Agent initialized successfully!")
    
except Exception as e:
    print(f"   ❌ Agent initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL DEBUG CHECKS PASSED!")
print("="*70)
print("\n💡 Agent should work now. Try running test_agent.py again")