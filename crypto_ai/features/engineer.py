import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

class FeatureEngineer:
    """
    Centralized Feature Engineering logic to ensure consistency between
    training (ML System) and serving (Analyzer).
    """

    def __init__(self):
        pass

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set (70+ features).
        This must be the SINGLE source of truth for feature generation.
        """
        if df.empty:
            return df
        
        # Working on a copy to avoid side effects
        df = df.copy()
        
        # 1. Basic price features
        df['price_change'] = df['close'].pct_change()
        df['high_low_pct'] = (df['high'] - df['low']) / df['low']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']
        
        # 2. Moving averages
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            df[f'ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
        
        # 3. Technical indicators
        self._add_rsi(df, windows=[14, 21, 28])
        self._add_macd(df)
        self._add_bollinger_bands(df)
        self._add_atr(df)
        
        # 4. Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        # Handle division by zero for initial rows
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan) 
        df['price_volume'] = df['close'] * df['volume']
        # VWAP
        vol_sum = df['volume'].rolling(20).sum()
        df['vwap'] = df['price_volume'].rolling(20).sum() / vol_sum.replace(0, np.nan)
        
        # 5. Volatility features
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['price_change'].rolling(window).std()
            # Ratio of current volatility to longer term average
            vol_long = df[f'volatility_{window}'].rolling(50).mean()
            df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / vol_long.replace(0, np.nan)
        
        # 6. Momentum features
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            df[f'roc_{window}'] = df['close'].pct_change(window)
        
        # 7. Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
        
        # 8. Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
        
        # 9. Time features
        # Ensure timestamp index or column is available. 
        # Assuming df has datetime index or 'timestamp' column.
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        elif 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
        else:
             # Fallback if no time data found, though this shouldn't happen in this system
            logging.warning("No timestamp information found for time features")
            dates = None

        if dates is not None:
            df['hour'] = dates.hour
            df['day_of_week'] = dates.dayofweek
            df['day_of_month'] = dates.day
            df['is_weekend'] = (dates.dayofweek >= 5).astype(int)
        
        # Clean up infinite values created by division by zero
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return df
    
    def _add_rsi(self, df: pd.DataFrame, windows: List[int] = [14, 21, 28]):
        """Add RSI indicators with multiple windows"""
        for window in windows:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    
    def _add_macd(self, df: pd.DataFrame):
        """Add MACD indicator"""
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    def _add_bollinger_bands(self, df: pd.DataFrame, window: int = 20):
        """Add Bollinger Bands"""
        rolling_mean = df['close'].rolling(window).mean()
        rolling_std = df['close'].rolling(window).std()
        df['bb_upper'] = rolling_mean + (rolling_std * 2)
        df['bb_lower'] = rolling_mean - (rolling_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        # Handle division by zero
        range_bb = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
        df['bb_position'] = (df['close'] - df['bb_lower']) / range_bb
    
    def _add_atr(self, df: pd.DataFrame, window: int = 14):
        """Add Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window).mean()
