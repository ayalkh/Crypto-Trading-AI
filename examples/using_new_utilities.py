"""
Example: Using the New Utility Modules
Demonstrates how to use config_loader and retry_handler
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader, get_config
from utils.retry_handler import retry_on_failure, RetryHandler, APIError, NetworkError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Example 1: Loading Configuration with Environment Variables
def example_config_loading():
    """Example of loading configuration with env variable support"""
    print("\n" + "="*60)
    print("Example 1: Configuration Loading")
    print("="*60)
    
    # Method 1: Using the convenience function
    config = get_config('automation_config.json')
    print(f"✅ Config loaded. Database path: {config.get('system', {}).get('database_path')}")
    
    # Method 2: Using ConfigLoader class
    loader = ConfigLoader(config_file='automation_config.json', env_file='.env')
    config = loader.load_config()
    
    # Access configuration
    email_enabled = config.get('alerts', {}).get('email', {}).get('enabled', False)
    print(f"Email alerts enabled: {email_enabled}")
    
    # Environment variables will override JSON config
    # If EMAIL_USERNAME is set in .env, it will be used
    email_username = config.get('alerts', {}).get('email', {}).get('username', '')
    if email_username:
        print(f"Email username: {email_username[:3]}***")  # Partially hide for security
    else:
        print("Email username: Not set (use EMAIL_USERNAME env var)")


# Example 2: Using Retry Handler
def example_retry_handler():
    """Example of using retry handler for API calls"""
    print("\n" + "="*60)
    print("Example 2: Retry Handler")
    print("="*60)
    
    # Simulate an API call that might fail
    call_count = [0]  # Use list to allow modification in nested function
    
    def unreliable_api_call():
        """Simulate an API call that fails 2 times then succeeds"""
        call_count[0] += 1
        if call_count[0] < 3:
            raise NetworkError(f"Network error on attempt {call_count[0]}")
        return f"Success on attempt {call_count[0]}"
    
    # Use retry handler
    handler = RetryHandler(max_retries=3, initial_delay=0.5)
    
    try:
        result = handler.retry(
            unreliable_api_call,
            exceptions=(NetworkError,),
            on_retry=lambda attempt, error: print(f"  Retry {attempt}: {error}")
        )
        print(f"✅ {result}")
    except Exception as e:
        print(f"❌ Failed after retries: {e}")


# Example 3: Using Retry Decorator
def example_retry_decorator():
    """Example of using retry decorator"""
    print("\n" + "="*60)
    print("Example 3: Retry Decorator")
    print("="*60)
    
    @retry_on_failure(
        max_retries=3,
        initial_delay=0.5,
        exceptions=(ValueError,),
        on_retry=lambda attempt, error: print(f"  Decorator retry {attempt}: {error}")
    )
    def process_data(value):
        """Process data that might fail"""
        if value < 5:
            raise ValueError(f"Value {value} is too small")
        return f"Processed: {value * 2}"
    
    # This will succeed on retry
    try:
        result = process_data(3)  # Will fail first, then retry
        print(f"✅ {result}")
    except Exception as e:
        print(f"❌ Failed: {e}")


# Example 4: Real-world API Call with Retry
def example_real_api_call():
    """Example of real API call with retry logic"""
    print("\n" + "="*60)
    print("Example 4: Real API Call Pattern")
    print("="*60)
    
    import ccxt
    
    @retry_on_failure(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        exceptions=(ccxt.NetworkError, ccxt.ExchangeError, Exception)
    )
    def fetch_ticker(symbol='BTC/USDT'):
        """Fetch ticker with retry logic"""
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker(symbol)
        return ticker
    
    try:
        ticker = fetch_ticker('BTC/USDT')
        print(f"✅ BTC/USDT Price: ${ticker['last']:,.2f}")
    except Exception as e:
        print(f"❌ Failed to fetch ticker: {e}")
        print("   (This is expected if you don't have internet or API access)")


# Example 5: Configuration with Environment Override
def example_env_override():
    """Example showing how environment variables override config"""
    print("\n" + "="*60)
    print("Example 5: Environment Variable Override")
    print("="*60)
    
    # Set a test environment variable
    os.environ['DATABASE_PATH'] = 'test_database.db'
    
    loader = ConfigLoader(config_file='automation_config.json')
    config = loader.load_config()
    
    db_path = config.get('system', {}).get('database_path')
    print(f"Database path: {db_path}")
    print("✅ Environment variable DATABASE_PATH overrides JSON config")
    
    # Clean up
    if 'DATABASE_PATH' in os.environ:
        del os.environ['DATABASE_PATH']


if __name__ == "__main__":
    print("\n" + "🚀 Utility Examples" + "\n")
    
    # Run examples
    example_config_loading()
    example_retry_handler()
    example_retry_decorator()
    example_env_override()
    
    # Skip real API call example (requires internet)
    # example_real_api_call()
    
    print("\n" + "="*60)
    print("✅ Examples complete!")
    print("="*60)
    print("\n💡 Tips:")
    print("  1. Use ConfigLoader for secure config management")
    print("  2. Use retry_handler for API calls and network operations")
    print("  3. Store sensitive data in .env file (not in JSON)")
    print("  4. Use decorators for cleaner code")
    print()




