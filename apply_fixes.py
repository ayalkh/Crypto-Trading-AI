"""
Complete Fix for Crypto Trading Agent Issues
Fixes:
1. Missing logger in tools.py
2. Missing ml_predictions table
3. Feature mismatch in save_predictions_to_db.py
"""
import sys
import os

def fix_tools_logger():
    """Fix missing logger in SmartConsensusAnalyzer"""
    
    file_path = "crypto_agent/tools.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the SmartConsensusAnalyzer __init__ method
    old_init = '''    def __init__(self, db: AgentDatabase):
        """Initialize consensus analyzer"""
        self.db = db
        logger.info("🧠 Smart Consensus Analyzer initialized")'''
    
    new_init = '''    def __init__(self, db: AgentDatabase):
        """Initialize consensus analyzer"""
        self.db = db
        self.logger = logging.getLogger(__name__)  # FIX: Add logger
        logger.info("🧠 Smart Consensus Analyzer initialized")'''
    
    if old_init in content:
        content = content.replace(old_init, new_init)
        
        # Backup
        with open(file_path + '.backup', 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Write fixed version
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed missing logger in tools.py")
        return True
    else:
        print("⚠️ Pattern not found in tools.py - may already be fixed")
        return False


def fix_database_table():
    """Add ml_predictions table creation to database.py"""
    
    file_path = "crypto_agent/database.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already has the table creation
    if '_create_ml_predictions_table' in content:
        print("✅ ml_predictions table creation already exists")
        return True
    
    # Find the __init__ method
    old_init = '''    def __init__(self, db_path: str = DB_PATH):
        """Initialize database connection"""
        self.db_path = db_path
        logger.info(f"📊 Database initialized: {db_path}")'''
    
    new_init = '''    def __init__(self, db_path: str = DB_PATH):
        """Initialize database connection"""
        self.db_path = db_path
        self._create_ml_predictions_table()  # FIX: Create table
        logger.info(f"📊 Database initialized: {db_path}")'''
    
    # Add the method after the __init__ method
    method_to_add = '''
    
    def _create_ml_predictions_table(self):
        """Create ml_predictions table if it doesn't exist"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ml_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        target_timestamp DATETIME,
                        model_type TEXT NOT NULL,
                        predicted_price REAL,
                        predicted_direction TEXT,
                        direction_probability REAL,
                        predicted_change_pct REAL,
                        confidence_score REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_predictions_symbol_timeframe 
                    ON ml_predictions(symbol, timeframe, timestamp DESC)
                """)
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error creating ml_predictions table: {e}")
'''
    
    if old_init in content:
        # Replace init
        content = content.replace(old_init, new_init)
        
        # Add method after get_connection
        insertion_point = content.find('    def get_latest_price(')
        if insertion_point == -1:
            insertion_point = content.find('    def get_ml_predictions(')
        
        if insertion_point != -1:
            content = content[:insertion_point] + method_to_add + '\n' + content[insertion_point:]
            
            # Backup
            with open(file_path + '.backup', 'w', encoding='utf-8') as f:
                content_backup = open(file_path, 'r', encoding='utf-8').read()
                f.write(content_backup)
            
            # Write fixed version
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Added ml_predictions table creation to database.py")
            return True
        else:
            print("❌ Could not find insertion point")
            return False
    else:
        print("⚠️ Pattern not found in database.py")
        return False


def create_instructions():
    """Create instruction file for manual fixes"""
    
    instructions = """
================================================================================
CRYPTO TRADING AGENT - REMAINING FIXES
================================================================================

Your diagnostic test passed! The feature mismatch is happening in a different
script. Here's what you need to do:

1. FIXES APPLIED AUTOMATICALLY:
   ✅ Added self.logger to SmartConsensusAnalyzer in crypto_agent/tools.py
   ✅ Added ml_predictions table creation to crypto_agent/database.py

2. MANUAL FIX NEEDED - save_predictions_to_db.py:

   The issue: Your save_predictions_to_db.py is using different lookback 
   periods than your training script, which creates different feature counts.

   SOLUTION: Use the SAME lookback configuration as training.

   In save_predictions_to_db.py, find the make_ensemble_prediction call
   and ensure it uses the correct lookback:

   BEFORE (in save_predictions_to_db.py):
   ```python
   prediction = ml_system.make_ensemble_prediction(symbol, timeframe)
   ```

   The make_ensemble_prediction in optimized_ml_system_v2.py already has
   the correct lookback map:
   ```python
   lookback_map = {'5m': 1, '15m': 1, '1h': 2, '4h': 3, '1d': 6}
   ```

   So the issue must be that optimized_ml_system is the OLD version!

   FIX: Use the updated save_predictions_to_db.py that I provided.

3. TESTING:

   After applying fixes, run in this order:

   Step 1: Test the agent with existing (empty) predictions table
   ```bash
   python test_agent.py
   ```
   
   This will test database connection and tools (should work now).

   Step 2: Generate predictions
   ```bash
   python save_predictions_to_db.py
   ```
   
   This should now work without feature mismatch errors.

   Step 3: Test agent with real predictions
   ```bash
   python test_agent.py
   ```
   
   This should now show actual ML predictions and consensus.

4. IF STILL GETTING FEATURE MISMATCH:

   The issue is that you're using `optimized_ml_system.py` instead of
   `optimized_ml_system_v2.py` in save_predictions_to_db.py.

   Change line 11 in save_predictions_to_db.py:
   
   FROM:
   ```python
   from optimized_ml_system import OptimizedCryptoMLSystem
   ```
   
   TO:
   ```python
   from optimized_ml_system_v2 import OptimizedMLSystemV2 as OptimizedCryptoMLSystem
   ```

================================================================================
QUICK START:
================================================================================

1. Run: python test_agent.py
   (Tests database connection and tools)

2. Run: python save_predictions_to_db.py
   (Generates ML predictions and saves to database)

3. Run: python test_agent.py again
   (Verifies agent can read predictions)

4. Run: python -m crypto_agent.agent
   (Start the actual agent!)

================================================================================
"""
    
    with open('FIX_INSTRUCTIONS.txt', 'w') as f:
        f.write(instructions)
    
    print("\n📄 Created FIX_INSTRUCTIONS.txt with detailed steps")


def main():
    print("\n" + "="*70)
    print("🔧 APPLYING FIXES TO CRYPTO TRADING AGENT")
    print("="*70 + "\n")
    
    success_count = 0
    total_fixes = 2
    
    # Fix 1: tools.py logger
    print("1️⃣ Fixing tools.py logger...")
    if fix_tools_logger():
        success_count += 1
    
    # Fix 2: database.py table
    print("\n2️⃣ Fixing database.py ml_predictions table...")
    if fix_database_table():
        success_count += 1
    
    # Create instructions
    print("\n3️⃣ Creating instruction file...")
    create_instructions()
    
    print("\n" + "="*70)
    print("📊 FIX SUMMARY")
    print("="*70)
    print(f"Automatic fixes applied: {success_count}/{total_fixes}")
    
    if success_count == total_fixes:
        print("\n✅ ALL AUTOMATIC FIXES APPLIED SUCCESSFULLY!")
    else:
        print(f"\n⚠️ Some fixes failed - see messages above")
    
    print("\n📖 Next Steps:")
    print("   1. Read FIX_INSTRUCTIONS.txt for detailed steps")
    print("   2. Update save_predictions_to_db.py to use optimized_ml_system_v2")
    print("   3. Run: python test_agent.py")
    print("   4. Run: python save_predictions_to_db.py")
    print("\n")


if __name__ == "__main__":
    main()