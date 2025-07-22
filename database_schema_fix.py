"""
Database Schema Fix Script
Adds missing columns to existing database for enhanced collector
"""

import sqlite3
import os

def fix_database_schema():
    """Fix database schema to work with enhanced collector"""
    db_path = 'data/multi_timeframe_data.db'
    
    if not os.path.exists(db_path):
        print("❌ Database file not found!")
        return False
    
    print("🔧 FIXING DATABASE SCHEMA")
    print("=" * 40)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if last_timestamp column exists
        cursor.execute("PRAGMA table_info(collection_status)")
        columns = [column[1] for column in cursor.fetchall()]
        
        print(f"📊 Current columns: {columns}")
        
        if 'last_timestamp' not in columns:
            print("🔄 Adding missing last_timestamp column...")
            cursor.execute("""
                ALTER TABLE collection_status 
                ADD COLUMN last_timestamp DATETIME
            """)
            print("✅ Added last_timestamp column")
        else:
            print("✅ last_timestamp column already exists")
        
        # Update existing records with last_timestamp
        print("🔄 Updating existing records...")
        cursor.execute("""
            UPDATE collection_status 
            SET last_timestamp = (
                SELECT MAX(timestamp) 
                FROM price_data 
                WHERE price_data.symbol = collection_status.symbol 
                AND price_data.timeframe = collection_status.timeframe
            )
            WHERE last_timestamp IS NULL
        """)
        
        updated_rows = cursor.rowcount
        print(f"✅ Updated {updated_rows} records with last_timestamp")
        
        conn.commit()
        conn.close()
        
        print("🎉 Database schema fix complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        return False

def main():
    """Main function"""
    print("🔧 DATABASE SCHEMA FIX")
    print("=" * 30)
    
    if fix_database_schema():
        print("\n✅ Database is now compatible with enhanced collector!")
        print("💡 You can now run: python multi_timeframe_collector.py --force")
    else:
        print("\n❌ Schema fix failed!")

if __name__ == "__main__":
    main()