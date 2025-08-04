"""
Check Database Structure
Find out what tables and columns exist in your database
"""
import sqlite3
import pandas as pd

def check_database_structure(db_path='data/multi_timeframe_data.db'):
    """Check what tables and columns exist in the database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🔍 CHECKING DATABASE STRUCTURE")
        print("=" * 40)
        print(f"Database: {db_path}")
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("❌ No tables found in database!")
            print("You need to run your data collector first.")
            return None
        
        print(f"\n📊 Found {len(tables)} table(s):")
        
        table_info = {}
        
        for table in tables:
            table_name = table[0]
            print(f"\n📋 Table: {table_name}")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            print("   Columns:")
            column_names = []
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                column_names.append(col_name)
                print(f"     - {col_name} ({col_type})")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            print(f"   Rows: {row_count}")
            
            # Get sample data
            if row_count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample_data = cursor.fetchall()
                print("   Sample data:")
                for i, row in enumerate(sample_data):
                    print(f"     Row {i+1}: {row}")
            
            table_info[table_name] = {
                'columns': column_names,
                'row_count': row_count
            }
        
        conn.close()
        return table_info
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return None

def suggest_table_mapping(table_info):
    """Suggest which table to use for ML"""
    print("\n💡 SUGGESTIONS FOR ML INTEGRATION")
    print("=" * 40)
    
    if not table_info:
        print("❌ No tables to analyze")
        return None
    
    # Look for tables with OHLCV data
    for table_name, info in table_info.items():
        columns = [col.lower() for col in info['columns']]
        
        # Check if this looks like market data
        has_ohlcv = all(col in columns for col in ['open', 'high', 'low', 'close'])
        has_timestamp = any(col in columns for col in ['timestamp', 'time', 'date'])
        has_symbol = any(col in columns for col in ['symbol', 'pair'])
        
        score = 0
        if has_ohlcv: score += 3
        if has_timestamp: score += 2  
        if has_symbol: score += 2
        if 'volume' in columns: score += 1
        
        print(f"\n📊 Table: {table_name}")
        print(f"   Compatibility Score: {score}/8")
        print(f"   Has OHLCV: {'✅' if has_ohlcv else '❌'}")
        print(f"   Has Timestamp: {'✅' if has_timestamp else '❌'}")
        print(f"   Has Symbol: {'✅' if has_symbol else '❌'}")
        print(f"   Has Volume: {'✅' if 'volume' in columns else '❌'}")
        print(f"   Row Count: {info['row_count']}")
        
        if score >= 6:
            print(f"   🎯 RECOMMENDED: This table looks good for ML!")
            return table_name
        elif score >= 4:
            print(f"   ⚠️ POSSIBLE: This table might work with modifications")
        else:
            print(f"   ❌ NOT SUITABLE: This table doesn't look like market data")
    
    return None

def main():
    print("🔍 DATABASE STRUCTURE ANALYZER")
    print("=" * 50)
    
    # Check database structure
    table_info = check_database_structure()
    
    if table_info:
        # Suggest best table for ML
        recommended_table = suggest_table_mapping(table_info)
        
        if recommended_table:
            print(f"\n✅ RECOMMENDATION: Use table '{recommended_table}' for ML")
            print(f"   Update your ML configuration to use this table.")
        else:
            print(f"\n⚠️ No suitable table found for ML")
            print(f"   Make sure your data collector is creating proper OHLCV data")
    else:
        print("\n❌ No data found. Steps to fix:")
        print("1. Run your data collector: python multi_timeframe_collector.py")
        print("2. Let it collect data for at least 30 minutes")
        print("3. Run this script again to check the structure")

if __name__ == "__main__":
    main()