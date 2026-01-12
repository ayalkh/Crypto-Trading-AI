"""
Database Viewer for Crypto Trading Data
Easy-to-use tool to view and explore your collected data
"""
import sqlite3
import pandas as pd
import os
from datetime import datetime
import sys

class DatabaseViewer:
    def __init__(self, db_path='data/ml_crypto_data.db'):
        """Initialize database viewer"""
        self.db_path = db_path
        
        # Check if database exists
        if not os.path.exists(db_path):
            print(f"❌ Database not found: {db_path}")
            print("💡 Please run data collection first!")
            sys.exit(1)
        
        print(f"✅ Connected to database: {db_path}")
    
    def list_tables(self):
        """List all tables in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return tables
    
    def get_table_info(self, table_name):
        """Get information about a table's structure"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get column information
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        conn.close()
        
        return columns, row_count
    
    def view_table_sample(self, table_name, limit=10):
        """View a sample of data from a table"""
        conn = sqlite3.connect(self.db_path)
        
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    
    def get_data_summary(self):
        """Get comprehensive summary of all data"""
        conn = sqlite3.connect(self.db_path)
        
        # Check which table has the data
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Look for price data table
        price_table = None
        for table in tables:
            if 'price' in table.lower():
                price_table = table
                break
        
        if not price_table:
            print("⚠️ No price data table found")
            conn.close()
            return None
        
        # Get summary statistics
        query = f"""
            SELECT 
                symbol,
                timeframe,
                COUNT(*) as records,
                MIN(timestamp) as earliest,
                MAX(timestamp) as latest,
                MIN(close) as min_price,
                MAX(close) as max_price,
                AVG(close) as avg_price,
                SUM(volume) as total_volume
            FROM {price_table}
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """
        
        summary_df = pd.read_sql_query(query, conn)
        conn.close()
        
        return summary_df
    
    def view_recent_data(self, symbol='BTC/USDT', timeframe='1h', limit=20):
        """View recent data for a specific symbol and timeframe"""
        conn = sqlite3.connect(self.db_path)
        
        # Find the price table
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        price_table = None
        for table in tables:
            if 'price' in table.lower():
                price_table = table
                break
        
        if not price_table:
            print("⚠️ No price data table found")
            conn.close()
            return None
        
        query = f"""
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM {price_table}
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
        conn.close()
        
        return df
    
    def get_symbols(self):
        """Get list of available symbols"""
        conn = sqlite3.connect(self.db_path)
        
        # Find price table
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        price_table = None
        for table in tables:
            if 'price' in table.lower():
                price_table = table
                break
        
        if not price_table:
            conn.close()
            return []
        
        cursor.execute(f"SELECT DISTINCT symbol FROM {price_table} ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return symbols
    
    def get_timeframes(self):
        """Get list of available timeframes"""
        conn = sqlite3.connect(self.db_path)
        
        # Find price table
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        price_table = None
        for table in tables:
            if 'price' in table.lower():
                price_table = table
                break
        
        if not price_table:
            conn.close()
            return []
        
        cursor.execute(f"SELECT DISTINCT timeframe FROM {price_table} ORDER BY timeframe")
        timeframes = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return timeframes
    
    def export_to_csv(self, symbol='BTC/USDT', timeframe='1h', output_file=None):
        """Export data to CSV file"""
        conn = sqlite3.connect(self.db_path)
        
        # Find price table
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        price_table = None
        for table in tables:
            if 'price' in table.lower():
                price_table = table
                break
        
        if not price_table:
            print("⚠️ No price data table found")
            conn.close()
            return False
        
        query = f"""
            SELECT * FROM {price_table}
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
        conn.close()
        
        if output_file is None:
            # Generate filename
            safe_symbol = symbol.replace('/', '_')
            output_file = f"export_{safe_symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df.to_csv(output_file, index=False)
        print(f"✅ Data exported to: {output_file}")
        
        return True
    
    def interactive_menu(self):
        """Interactive menu for exploring database"""
        while True:
            print("\n" + "="*60)
            print("📊 DATABASE VIEWER - INTERACTIVE MENU")
            print("="*60)
            
            print("\n1. 📋 List all tables")
            print("2. 📊 View data summary")
            print("3. 🔍 View recent data")
            print("4. 📈 View specific symbol/timeframe")
            print("5. 💾 Export data to CSV")
            print("6. 📑 View table structure")
            print("7. 🔢 Quick statistics")
            print("0. 🚪 Exit")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '1':
                self.show_tables()
            elif choice == '2':
                self.show_summary()
            elif choice == '3':
                self.show_recent_data()
            elif choice == '4':
                self.show_specific_data()
            elif choice == '5':
                self.export_data_interactive()
            elif choice == '6':
                self.show_table_structure()
            elif choice == '7':
                self.show_quick_stats()
            elif choice == '0':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice, please try again")
            
            input("\n⏸️  Press Enter to continue...")
    
    def show_tables(self):
        """Display all tables"""
        print("\n📋 DATABASE TABLES:")
        print("-" * 40)
        
        tables = self.list_tables()
        
        for i, table in enumerate(tables, 1):
            columns, row_count = self.get_table_info(table)
            print(f"{i}. {table}: {row_count:,} rows, {len(columns)} columns")
    
    def show_summary(self):
        """Display data summary"""
        print("\n📊 DATA SUMMARY:")
        print("-" * 80)
        
        summary = self.get_data_summary()
        
        if summary is not None and not summary.empty:
            # Format for display
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_rows', None)
            
            print(summary.to_string(index=False))
            
            # Overall statistics
            print("\n📈 OVERALL STATISTICS:")
            print(f"   Total records: {summary['records'].sum():,}")
            print(f"   Symbols: {summary['symbol'].nunique()}")
            print(f"   Timeframes: {summary['timeframe'].nunique()}")
        else:
            print("⚠️ No data available")
    
    def show_recent_data(self):
        """Show recent data interactively"""
        symbols = self.get_symbols()
        timeframes = self.get_timeframes()
        
        if not symbols or not timeframes:
            print("⚠️ No data available")
            return
        
        print(f"\n📈 Available symbols: {', '.join(symbols)}")
        print(f"⏱️  Available timeframes: {', '.join(timeframes)}")
        
        symbol = input(f"\nEnter symbol (default: {symbols[0]}): ").strip() or symbols[0]
        timeframe = input(f"Enter timeframe (default: 1h): ").strip() or '1h'
        limit = input("Number of records (default: 20): ").strip() or '20'
        
        try:
            limit = int(limit)
        except:
            limit = 20
        
        df = self.view_recent_data(symbol, timeframe, limit)
        
        if df is not None and not df.empty:
            print(f"\n📊 Recent {limit} records for {symbol} {timeframe}:")
            print("-" * 80)
            
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            
            print(df.to_string(index=False))
        else:
            print(f"⚠️ No data found for {symbol} {timeframe}")
    
    def show_specific_data(self):
        """Show specific symbol/timeframe data"""
        symbols = self.get_symbols()
        timeframes = self.get_timeframes()
        
        if not symbols or not timeframes:
            print("⚠️ No data available")
            return
        
        print(f"\n📈 Available symbols:")
        for i, symbol in enumerate(symbols, 1):
            print(f"   {i}. {symbol}")
        
        try:
            symbol_idx = int(input(f"\nSelect symbol (1-{len(symbols)}): ")) - 1
            symbol = symbols[symbol_idx]
        except:
            print("❌ Invalid selection")
            return
        
        print(f"\n⏱️  Available timeframes:")
        for i, tf in enumerate(timeframes, 1):
            print(f"   {i}. {tf}")
        
        try:
            tf_idx = int(input(f"\nSelect timeframe (1-{len(timeframes)}): ")) - 1
            timeframe = timeframes[tf_idx]
        except:
            print("❌ Invalid selection")
            return
        
        limit = input("\nNumber of records to display (default: 50): ").strip() or '50'
        
        try:
            limit = int(limit)
        except:
            limit = 50
        
        df = self.view_recent_data(symbol, timeframe, limit)
        
        if df is not None and not df.empty:
            print(f"\n📊 Last {len(df)} records for {symbol} {timeframe}:")
            print("-" * 80)
            
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            
            print(df.to_string(index=False))
            
            # Show statistics
            print(f"\n📈 STATISTICS:")
            print(f"   Latest Price: ${df['close'].iloc[0]:,.2f}")
            print(f"   Highest: ${df['high'].max():,.2f}")
            print(f"   Lowest: ${df['low'].min():,.2f}")
            print(f"   Average: ${df['close'].mean():,.2f}")
            print(f"   Total Volume: {df['volume'].sum():,.2f}")
        else:
            print(f"⚠️ No data found for {symbol} {timeframe}")
    
    def export_data_interactive(self):
        """Interactive data export"""
        symbols = self.get_symbols()
        timeframes = self.get_timeframes()
        
        if not symbols or not timeframes:
            print("⚠️ No data available")
            return
        
        print(f"\n💾 EXPORT DATA TO CSV")
        print("-" * 40)
        
        print(f"\n📈 Available symbols: {', '.join(symbols)}")
        symbol = input(f"Enter symbol (default: {symbols[0]}): ").strip() or symbols[0]
        
        print(f"\n⏱️  Available timeframes: {', '.join(timeframes)}")
        timeframe = input("Enter timeframe (default: 1h): ").strip() or '1h'
        
        output_file = input("Output filename (press Enter for auto-generated): ").strip() or None
        
        self.export_to_csv(symbol, timeframe, output_file)
    
    def show_table_structure(self):
        """Show table structure"""
        tables = self.list_tables()
        
        print(f"\n📑 Available tables:")
        for i, table in enumerate(tables, 1):
            print(f"   {i}. {table}")
        
        try:
            table_idx = int(input(f"\nSelect table (1-{len(tables)}): ")) - 1
            table_name = tables[table_idx]
        except:
            print("❌ Invalid selection")
            return
        
        columns, row_count = self.get_table_info(table_name)
        
        print(f"\n📊 TABLE: {table_name}")
        print(f"   Total rows: {row_count:,}")
        print(f"\n   Columns:")
        print(f"   {'Name':<20} {'Type':<15} {'Not Null':<10} {'Primary Key'}")
        print("   " + "-"*60)
        
        for col in columns:
            col_id, name, type_name, not_null, default, pk = col
            print(f"   {name:<20} {type_name:<15} {'Yes' if not_null else 'No':<10} {'Yes' if pk else 'No'}")
        
        # Show sample data
        print(f"\n   Sample data (first 5 rows):")
        sample_df = self.view_table_sample(table_name, 5)
        
        if not sample_df.empty:
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(sample_df.to_string(index=False))
    
    def show_quick_stats(self):
        """Show quick statistics"""
        print("\n🔢 QUICK STATISTICS:")
        print("-" * 60)
        
        # Database file size
        db_size = os.path.getsize(self.db_path) / (1024 * 1024)  # MB
        print(f"📁 Database size: {db_size:.2f} MB")
        
        # Table count
        tables = self.list_tables()
        print(f"📋 Number of tables: {len(tables)}")
        
        # Get summary
        summary = self.get_data_summary()
        
        if summary is not None and not summary.empty:
            print(f"\n📊 DATA OVERVIEW:")
            print(f"   Total records: {summary['records'].sum():,}")
            print(f"   Symbols tracked: {summary['symbol'].nunique()}")
            print(f"   Timeframes: {summary['timeframe'].nunique()}")
            
            # Data freshness
            summary['latest'] = pd.to_datetime(summary['latest'])
            latest_overall = summary['latest'].max()
            hours_old = (datetime.now() - latest_overall).total_seconds() / 3600
            
            print(f"\n📅 DATA FRESHNESS:")
            print(f"   Latest data: {latest_overall}")
            print(f"   Age: {hours_old:.1f} hours old")
            
            if hours_old < 2:
                status = "🟢 Very fresh"
            elif hours_old < 24:
                status = "🟡 Acceptable"
            else:
                status = "🔴 Needs update"
            
            print(f"   Status: {status}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Database Viewer for Crypto Trading Data')
    parser.add_argument('--db', default='data//ml_crypto_data.db', 
                       help='Path to database file')
    parser.add_argument('--summary', action='store_true', 
                       help='Show data summary and exit')
    parser.add_argument('--export', nargs=2, metavar=('SYMBOL', 'TIMEFRAME'),
                       help='Export data to CSV: --export BTC/USDT 1h')
    parser.add_argument('--list', action='store_true',
                       help='List tables and exit')
    
    args = parser.parse_args()
    
    # Initialize viewer
    viewer = DatabaseViewer(args.db)
    
    # Handle command line options
    if args.list:
        viewer.show_tables()
    elif args.summary:
        viewer.show_summary()
    elif args.export:
        symbol, timeframe = args.export
        viewer.export_to_csv(symbol, timeframe)
    else:
        # Run interactive menu
        viewer.interactive_menu()


if __name__ == "__main__":
    main()