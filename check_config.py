"""
Simple Configuration Checker
Helps diagnose .env file issues
"""
import os
import sys

print("🔍 CONFIGURATION CHECKER")
print("="*60)

# Check 1: Does .env file exist?
print("\n1️⃣ Checking if .env file exists...")
if os.path.exists('.env'):
    print("   ✅ .env file found")
    
    # Show file size
    size = os.path.getsize('.env')
    print(f"   📊 File size: {size} bytes")
    
    if size < 50:
        print("   ⚠️  WARNING: File is very small (might be empty)")
    
    # Try to read it
    print("\n   📄 Contents of .env file:")
    print("   " + "-"*50)
    try:
        with open('.env', 'r') as f:
            for line in f:
                # Hide password but show if it's set
                if 'PASSWORD' in line and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip()
                    if value and not value.startswith('#'):
                        print(f"   {key}={'*' * 20} (password hidden)")
                    else:
                        print(f"   {line.rstrip()}")
                else:
                    print(f"   {line.rstrip()}")
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
    print("   " + "-"*50)
    
else:
    print("   ❌ .env file NOT found!")
    print("\n   📝 Creating .env file for you...")
    
    template = """# PostgreSQL/Supabase Database Configuration
DB_HOST=db.your-project-id.supabase.co
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your-password-here

DB_MIN_CONN=2
DB_MAX_CONN=10

# INSTRUCTIONS:
# 1. Go to https://supabase.com/dashboard
# 2. Click Settings → Database
# 3. Copy your connection details
# 4. Replace the values above
"""
    
    with open('.env', 'w') as f:
        f.write(template)
    
    print("   ✅ Created .env file")
    print("   📝 Please edit it with your Supabase credentials")

# Check 2: Can we load dotenv?
print("\n2️⃣ Checking python-dotenv installation...")
try:
    from dotenv import load_dotenv
    print("   ✅ python-dotenv is installed")
except ImportError:
    print("   ❌ python-dotenv NOT installed")
    print("   📦 Install it with: pip install python-dotenv")
    sys.exit(1)

# Check 3: Load and verify environment variables
print("\n3️⃣ Loading environment variables...")
load_dotenv()

db_host = os.getenv('DB_HOST', 'NOT_SET')
db_port = os.getenv('DB_PORT', 'NOT_SET')
db_name = os.getenv('DB_NAME', 'NOT_SET')
db_user = os.getenv('DB_USER', 'NOT_SET')
db_password = os.getenv('DB_PASSWORD', 'NOT_SET')

print("\n   📊 Current Environment Variables:")
print(f"   DB_HOST = {db_host}")
print(f"   DB_PORT = {db_port}")
print(f"   DB_NAME = {db_name}")
print(f"   DB_USER = {db_user}")
print(f"   DB_PASSWORD = {'*' * 20 if db_password != 'NOT_SET' else 'NOT_SET'}")

# Check 4: Validate values
print("\n4️⃣ Validating configuration...")

issues = []

if db_host == 'NOT_SET' or db_host == 'localhost' or 'your-project' in db_host:
    issues.append("❌ DB_HOST is not configured correctly")
    print("   ❌ DB_HOST issue detected")
    print("      Current value: " + db_host)
    print("      Should look like: db.abcdefghijklmnop.supabase.co")
else:
    print("   ✅ DB_HOST looks good")

if db_password == 'NOT_SET' or 'your-password' in db_password:
    issues.append("❌ DB_PASSWORD is not set")
    print("   ❌ DB_PASSWORD not set")
else:
    print("   ✅ DB_PASSWORD is set")

if db_name != 'postgres':
    issues.append("⚠️ DB_NAME should be 'postgres' for Supabase")
    print("   ⚠️  DB_NAME should be 'postgres'")
else:
    print("   ✅ DB_NAME is correct")

if db_user != 'postgres':
    issues.append("⚠️ DB_USER should be 'postgres' for Supabase")
    print("   ⚠️  DB_USER should be 'postgres'")
else:
    print("   ✅ DB_USER is correct")

# Summary
print("\n" + "="*60)
if not issues:
    print("🎉 CONFIGURATION LOOKS GOOD!")
    print("="*60)
    print("\n✅ Your .env file is properly configured")
    print("\n📋 Next step:")
    print("   python test_database_connection.py")
else:
    print("⚠️  CONFIGURATION ISSUES FOUND")
    print("="*60)
    print("\n❌ Issues to fix:")
    for issue in issues:
        print(f"   {issue}")
    
    print("\n📝 HOW TO FIX:")
    print("\n1. Open .env file in a text editor (Notepad, VS Code, etc.)")
    print("2. Go to https://supabase.com/dashboard")
    print("3. Select your project")
    print("4. Click Settings → Database")
    print("5. Copy the connection details:")
    print("   - Host (looks like: db.xxxxx.supabase.co)")
    print("   - Password (click reveal to see it)")
    print("6. Replace the placeholder values in .env")
    print("7. Save the file")
    print("8. Run this script again")

print("\n" + "="*60)