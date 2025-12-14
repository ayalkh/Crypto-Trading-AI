"""
SQLite to Neon PostgreSQL Migration Script
Migrates your local crypto trading data to Neon cloud database
"""
import os
import sys
import sqlite3
import psycopg2
from psycopg2 import extras
from datetime import datetime
import pandas as pd

class DatabaseMigration:
    def __init__(self, sqlite_path='data/multi_timeframe_data.db'):
        """Initialize migration with SQLite path"""
        self.sqlite_path = sqlite_path
        self.pg_conn = None
        self.sqlite_conn = None
        
        # Connection details will be set later
        self.neon_config = {}
    
    def get_neon_credentials(self):
        """Get Neon database credentials from user"""
        print("\n🔐 NEON DATABASE CREDENTIALS")
        print("=" * 50)
        print("📝 You'll need your Neon connection details:")
        print("   1. Go to https://neon.tech")
        print("   2. Sign up (free, no credit card)")
        print("   3. Create a new project")
        print("   4. Get your connection string")
        print()
        print("Your connection string looks like:")
        print("postgresql://user:password@ep-xxx.region.aws.neon.tech/dbname")
        print("=" * 50)
        print()
        
        # Option 1: Full connection string
        print("OPTION 1: Enter full connection string")
        conn_string = input("Paste connection string (or press Enter for manual): ").strip()
        
        if conn_string:
            # Parse connection string
            try:
                # Format: postgresql://user:password@host:port/database
                import re
                pattern = r'postgresql://([^:]+):([^@]+)@([^:]+):?(\d+)?/(.+)'
                match = re.match(pattern, conn_string)
                
                if match:
                    user, password, host, port, database = match.groups()
                    self.neon_config = {
                        'host': host,
                        'port': port or '5432',
                        'database': database.split('?')[0],  # Remove query params
                        'user': user,
                        'password': password,
                        'sslmode': 'require'
                    }
                    print("✅ Connection string parsed successfully!")
                    return True
                else:
                    print("❌ Invalid connection string format")
                    return False
            except Exception as e:
                print(f"❌ Error parsing connection string: {e}")
                return False
        
        # Option 2: Manual entry
        print("\nOPTION 2: Enter details manually")
        self.neon_config = {
            'host': input("Host (ep-xxx.region.aws.neon.tech): ").strip(),
            'port': input("Port (default 5432): ").strip() or '5432',
            'database': input("Database name (neondb): ").strip() or 'neondb',
            'user': input("Username: ").strip(),
            'password': input("Password: ").strip(),
            'sslmode': 'require'
        }
        
        return True
    
    def test_neon_connection(self):
        """Test connection to Neon database"""
        print("\n🔌 Testing Neon connection...")
        
        try:
            conn = psycopg2.connect(**self.neon_config)
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"✅ Connected successfully!")
            print(f"   PostgreSQL version: {version.split(',')[0]}")
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("\n💡 Troubleshooting:")
            print("   - Check your credentials")
            print("   - Ensure your Neon project is active")
            print("   - Check your internet connection")
            return False
    
    def analyze_sqlite_data(self):
        """Analyze SQLite database"""
        print("\n📊 Analyzing SQLite database...")
        
        if not os.path.exists(self.sqlite_path):
            print(f"❌ SQLite database not found: {self.sqlite_path}")
            return False
        
        try:
            self.sqlite_conn = sqlite3.connect(self.sqlite_path)
            cursor = self.sqlite_conn.cursor()
            
            # Get tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            print(f"✅ Found {len(tables)} tables: {', '.join(tables)}")
            
            # Analyze each table
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   📋 {table}: {count:,} records")
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing SQLite: {e}")
            return False
    
    def create_neon_schema(self):
        """Create schema in Neon database"""
        print("\n🏗️ Creating schema in Neon...")
        
        try:
            self.pg_conn = psycopg2.connect(**self.neon_config)
            cursor = self.pg_conn.cursor()
            
            # Create price_data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open DECIMAL(20, 8) NOT NULL,
                    high DECIMAL(20, 8) NOT NULL,
                    low DECIMAL(20, 8) NOT NULL,
                    close DECIMAL(20, 8) NOT NULL,
                    volume DECIMAL(20, 8) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                );
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_timestamp 
                ON price_data(symbol, timeframe, timestamp DESC);
            """)
            
            # Create collection_status table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collection_status (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    last_update TIMESTAMP NOT NULL,
                    records_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    last_timestamp TIMESTAMP,
                    UNIQUE(symbol, timeframe)
                );
            """)
            
            self.pg_conn.commit()
            print("✅ Schema created successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error creating schema: {e}")
            if self.pg_conn:
                self.pg_conn.rollback()
            return False
    
    def migrate_data(self, batch_size=1000):
        """Migrate data from SQLite to Neon"""
        print("\n📦 Migrating data...")
        
        try:
            sqlite_cursor = self.sqlite_conn.cursor()
            pg_cursor = self.pg_conn.cursor()
            
            # Get tables to migrate
            sqlite_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in sqlite_cursor.fetchall()]
            
            for table in tables:
                print(f"\n📋 Migrating {table}...")
                
                # Get total count
                sqlite_cursor.execute(f"SELECT COUNT(*) FROM {table}")
                total_rows = sqlite_cursor.fetchone()[0]
                
                if total_rows == 0:
                    print(f"   ⚪ No data to migrate")
                    continue
                
                print(f"   Total records: {total_rows:,}")
                
                # Get column names
                sqlite_cursor.execute(f"PRAGMA table_info({table})")
                columns_info = sqlite_cursor.fetchall()
                columns = [col[1] for col in columns_info if col[1] != 'id']  # Exclude auto-increment id
                
                # Fetch and migrate in batches
                migrated = 0
                
                for offset in range(0, total_rows, batch_size):
                    # Fetch batch from SQLite
                    query = f"SELECT {', '.join(columns)} FROM {table} LIMIT {batch_size} OFFSET {offset}"
                    sqlite_cursor.execute(query)
                    batch = sqlite_cursor.fetchall()
                    
                    if not batch:
                        break
                    
                    # Insert batch into PostgreSQL
                    placeholders = ','.join(['%s'] * len(columns))
                    insert_query = f"""
                        INSERT INTO {table} ({', '.join(columns)})
                        VALUES ({placeholders})
                        ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                    """ if table == 'price_data' else f"""
                        INSERT INTO {table} ({', '.join(columns)})
                        VALUES ({placeholders})
                        ON CONFLICT (symbol, timeframe) DO NOTHING
                    """
                    
                    try:
                        extras.execute_batch(pg_cursor, insert_query, batch)
                        self.pg_conn.commit()
                        
                        migrated += len(batch)
                        progress = migrated / total_rows * 100
                        print(f"   ⏳ Progress: {migrated:,}/{total_rows:,} ({progress:.1f}%)", end='\r')
                        
                    except Exception as e:
                        print(f"\n   ⚠️ Batch error: {e}")
                        self.pg_conn.rollback()
                        continue
                
                print(f"\n   ✅ Migrated {migrated:,} records to {table}")
            
            return True
            
        except Exception as e:
            print(f"❌ Migration error: {e}")
            if self.pg_conn:
                self.pg_conn.rollback()
            return False
    
    def verify_migration(self):
        """Verify that migration was successful"""
        print("\n✅ Verifying migration...")
        
        try:
            sqlite_cursor = self.sqlite_conn.cursor()
            pg_cursor = self.pg_conn.cursor()
            
            # Compare record counts
            tables = ['price_data', 'collection_status']
            
            print(f"\n{'Table':<20} {'SQLite':<15} {'Neon':<15} {'Status'}")
            print("-" * 60)
            
            all_match = True
            
            for table in tables:
                # SQLite count
                sqlite_cursor.execute(f"SELECT COUNT(*) FROM {table}")
                sqlite_count = sqlite_cursor.fetchone()[0]
                
                # PostgreSQL count
                pg_cursor.execute(f"SELECT COUNT(*) FROM {table}")
                pg_count = pg_cursor.fetchone()[0]
                
                match = sqlite_count == pg_count
                status = "✅ Match" if match else "⚠️ Mismatch"
                
                print(f"{table:<20} {sqlite_count:<15,} {pg_count:<15,} {status}")
                
                if not match:
                    all_match = False
            
            if all_match:
                print("\n🎉 Migration verification successful! All data matched.")
            else:
                print("\n⚠️ Some counts don't match. This may be due to duplicates.")
            
            return all_match
            
        except Exception as e:
            print(f"❌ Verification error: {e}")
            return False
    
    def save_connection_config(self):
        """Save Neon connection config for future use"""
        print("\n💾 Saving connection configuration...")
        
        # Create config without password (for security)
        safe_config = {
            'host': self.neon_config['host'],
            'port': self.neon_config['port'],
            'database': self.neon_config['database'],
            'user': self.neon_config['user'],
            'sslmode': self.neon_config['sslmode']
        }
        
        # Save to file
        import json
        
        config_data = {
            'database_type': 'neon_postgresql',
            'connection': safe_config,
            'migration_date': datetime.now().isoformat(),
            'note': 'Password should be stored in environment variable NEON_PASSWORD'
        }
        
        with open('neon_config.json', 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print("✅ Configuration saved to neon_config.json")
        print("\n⚠️ IMPORTANT: Set your password as environment variable:")
        print("   Windows: set NEON_PASSWORD=your_password")
        print("   Linux/Mac: export NEON_PASSWORD=your_password")
    
    def cleanup(self):
        """Close database connections"""
        if self.sqlite_conn:
            self.sqlite_conn.close()
        if self.pg_conn:
            self.pg_conn.close()
    
    def run_migration(self):
        """Run complete migration process"""
        print("\n🚀 SQLITE TO NEON POSTGRESQL MIGRATION")
        print("=" * 60)
        
        # Step 1: Get credentials
        if not self.get_neon_credentials():
            print("❌ Failed to get credentials")
            return False
        
        # Step 2: Test connection
        if not self.test_neon_connection():
            print("❌ Cannot connect to Neon")
            return False
        
        # Step 3: Analyze SQLite
        if not self.analyze_sqlite_data():
            print("❌ Cannot analyze SQLite database")
            return False
        
        # Confirm migration
        print("\n⚠️ MIGRATION CONFIRMATION")
        print("=" * 60)
        print("This will:")
        print("  1. Create tables in your Neon database")
        print("  2. Copy all data from SQLite to Neon")
        print("  3. Verify the migration")
        print()
        confirm = input("Proceed with migration? (yes/no): ").strip().lower()
        
        if confirm != 'yes':
            print("❌ Migration cancelled")
            return False
        
        # Step 4: Create schema
        if not self.create_neon_schema():
            print("❌ Failed to create schema")
            return False
        
        # Step 5: Migrate data
        if not self.migrate_data():
            print("❌ Data migration failed")
            return False
        
        # Step 6: Verify
        self.verify_migration()
        
        # Step 7: Save config
        self.save_connection_config()
        
        # Cleanup
        self.cleanup()
        
        print("\n🎉 MIGRATION COMPLETE!")
        print("=" * 60)
        print("✅ Your data is now in Neon PostgreSQL cloud database")
        print("📝 Next steps:")
        print("   1. Set NEON_PASSWORD environment variable")
        print("   2. Update your scripts to use Neon connection")
        print("   3. Test your trading system with cloud database")
        print()
        
        return True


def main():
    """Main migration function"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        SQLite → Neon PostgreSQL Migration Tool          ║
    ║                                                          ║
    ║  Migrate your crypto trading data to the cloud!         ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check if SQLite database exists
    sqlite_path = 'data/multi_timeframe_data.db'
    if not os.path.exists(sqlite_path):
        print(f"❌ SQLite database not found: {sqlite_path}")
        print("💡 Make sure you have collected data first!")
        return
    
    # Run migration
    migrator = DatabaseMigration(sqlite_path)
    
    try:
        success = migrator.run_migration()
        
        if success:
            print("\n✅ Migration successful!")
        else:
            print("\n❌ Migration failed!")
            
    except KeyboardInterrupt:
        print("\n\n🛑 Migration cancelled by user")
        migrator.cleanup()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        migrator.cleanup()


if __name__ == "__main__":
    main()