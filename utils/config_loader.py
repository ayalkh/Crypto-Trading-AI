"""
Enhanced Configuration Loader with Environment Variable Support
Safely loads configuration from JSON files and environment variables
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    logging.warning("python-dotenv not installed. Install with: pip install python-dotenv")


class ConfigLoader:
    """Enhanced configuration loader with environment variable support"""
    
    def __init__(self, config_file: str = 'automation_config.json', env_file: str = '.env'):
        """
        Initialize config loader
        
        Args:
            config_file: Path to JSON configuration file
            env_file: Path to .env file
        """
        self.config_file = config_file
        self.env_file = env_file
        
        # Load environment variables from .env file
        if DOTENV_AVAILABLE:
            env_path = Path(env_file)
            if env_path.exists():
                load_dotenv(env_path)
                logging.info(f"✅ Loaded environment variables from {env_file}")
            else:
                logging.info(f"ℹ️ {env_file} not found, using system environment variables")
        else:
            logging.warning("⚠️ python-dotenv not available, only system environment variables will be used")
    
    def load_config(self, default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration from JSON file and merge with environment variables
        
        Args:
            default_config: Default configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        # Start with defaults
        if default_config is None:
            default_config = self._get_default_config()
        
        config = default_config.copy()
        
        # Load from JSON file if it exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                # Deep merge with defaults
                config = self._deep_merge(config, file_config)
                logging.info(f"✅ Configuration loaded from {self.config_file}")
            except Exception as e:
                logging.error(f"❌ Error loading config file: {e}, using defaults")
        else:
            logging.info(f"ℹ️ Config file {self.config_file} not found, using defaults")
        
        # Override with environment variables
        config = self._apply_env_overrides(config)
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "data_collection": {
                "enabled": True,
                "interval_minutes": 60,
                "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                "timeframes": ["5m", "15m", "1h", "4h", "1d"],
                "force_update_hours": 24
            },
            "signal_analysis": {
                "enabled": True,
                "interval_minutes": 15,
                "confidence_threshold": 75,
                "analyze_symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
            },
            "alerts": {
                "enabled": True,
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "to_email": ""
                },
                "desktop": {
                    "enabled": True
                },
                "log_file": {
                    "enabled": True
                }
            },
            "performance_tracking": {
                "enabled": True,
                "interval_hours": 6,
                "daily_report": True
            },
            "system": {
                "max_errors": 10,
                "error_cooldown_minutes": 30,
                "cleanup_interval_hours": 24,
                "database_path": "data/multi_timeframe_data.db"
            }
        }
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration"""
        # Email configuration
        if os.getenv('EMAIL_USERNAME'):
            config.setdefault('alerts', {}).setdefault('email', {})['username'] = os.getenv('EMAIL_USERNAME')
        if os.getenv('EMAIL_PASSWORD'):
            config.setdefault('alerts', {}).setdefault('email', {})['password'] = os.getenv('EMAIL_PASSWORD')
        if os.getenv('EMAIL_TO'):
            config.setdefault('alerts', {}).setdefault('email', {})['to_email'] = os.getenv('EMAIL_TO')
        if os.getenv('EMAIL_SMTP_SERVER'):
            config.setdefault('alerts', {}).setdefault('email', {})['smtp_server'] = os.getenv('EMAIL_SMTP_SERVER')
        if os.getenv('EMAIL_SMTP_PORT'):
            try:
                config.setdefault('alerts', {}).setdefault('email', {})['smtp_port'] = int(os.getenv('EMAIL_SMTP_PORT'))
            except ValueError:
                pass
        
        # Database path
        if os.getenv('DATABASE_PATH'):
            config.setdefault('system', {})['database_path'] = os.getenv('DATABASE_PATH')
        
        # Exchange API credentials (if needed)
        if os.getenv('EXCHANGE_API_KEY'):
            config.setdefault('exchange', {})['api_key'] = os.getenv('EXCHANGE_API_KEY')
        if os.getenv('EXCHANGE_API_SECRET'):
            config.setdefault('exchange', {})['api_secret'] = os.getenv('EXCHANGE_API_SECRET')
        
        # Webhook URLs
        if os.getenv('SLACK_WEBHOOK_URL'):
            config.setdefault('alerts', {}).setdefault('webhook', {})['url'] = os.getenv('SLACK_WEBHOOK_URL')
            config.setdefault('alerts', {}).setdefault('webhook', {})['enabled'] = True
        
        # System settings
        if os.getenv('MAX_ERRORS'):
            try:
                config.setdefault('system', {})['max_errors'] = int(os.getenv('MAX_ERRORS'))
            except ValueError:
                pass
        
        if os.getenv('ERROR_COOLDOWN_MINUTES'):
            try:
                config.setdefault('system', {})['error_cooldown_minutes'] = int(os.getenv('ERROR_COOLDOWN_MINUTES'))
            except ValueError:
                pass
        
        return config
    
    def save_config(self, config: Dict[str, Any], output_file: Optional[str] = None) -> bool:
        """
        Save configuration to JSON file (excluding sensitive data)
        
        Args:
            config: Configuration dictionary
            output_file: Output file path (defaults to self.config_file)
            
        Returns:
            True if successful, False otherwise
        """
        output_file = output_file or self.config_file
        
        # Create a safe copy without sensitive data
        safe_config = self._sanitize_config(config.copy())
        
        try:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(safe_config, f, indent=4, ensure_ascii=False)
            logging.info(f"✅ Configuration saved to {output_file}")
            return True
        except Exception as e:
            logging.error(f"❌ Error saving config: {e}")
            return False
    
    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from config before saving"""
        safe_config = config.copy()
        
        # Remove passwords and API keys
        if 'alerts' in safe_config and 'email' in safe_config['alerts']:
            if 'password' in safe_config['alerts']['email']:
                safe_config['alerts']['email']['password'] = '***REDACTED***'
        
        if 'exchange' in safe_config:
            if 'api_key' in safe_config['exchange']:
                safe_config['exchange']['api_key'] = '***REDACTED***'
            if 'api_secret' in safe_config['exchange']:
                safe_config['exchange']['api_secret'] = '***REDACTED***'
        
        return safe_config


def get_config(config_file: str = 'automation_config.json') -> Dict[str, Any]:
    """
    Convenience function to load configuration
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_file=config_file)
    return loader.load_config()

