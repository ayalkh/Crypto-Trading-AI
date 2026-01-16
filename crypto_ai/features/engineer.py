import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

class FeatureEngineer:
    """
    Centralized Feature Engineering logic to ensure consistency between
    training (ML System) and serving (Analyzer).
    
    Enhanced with advanced indicators:
    - Stochastic Oscillator (%K, %D)
    - Williams %R
    - On-Balance Volume (OBV)
    - Ichimoku Cloud (5 components)
    - ADX (Average Directional Index)
    - Market Regime Detection
    """

    def __init__(self):
        pass

    def create_features(self, df: pd.DataFrame, sentiment_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create comprehensive feature set (100+ features).
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
        
        # 4. NEW: Advanced indicators
        self._add_stochastic(df)
        self._add_williams_r(df)
        self._add_obv(df)
        self._add_ichimoku(df)
        self._add_adx(df)
        self._add_market_regime(df)
        
        # 5. NEW: Order book, funding, OI, arbitrage, and correlation proxies
        self._add_order_book_proxies(df)
        self._add_funding_rate_proxies(df)
        self._add_open_interest_proxies(df)
        self._add_arbitrage_features(df)
        self._add_correlation_features(df)
        
        # 6. Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan) 
        df['price_volume'] = df['close'] * df['volume']
        vol_sum = df['volume'].rolling(20).sum()
        df['vwap'] = df['price_volume'].rolling(20).sum() / vol_sum.replace(0, np.nan)
        
        # 7. Volatility features
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['price_change'].rolling(window).std()
            vol_long = df[f'volatility_{window}'].rolling(50).mean()
            df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / vol_long.replace(0, np.nan)
        
        # 8. Momentum features
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            df[f'roc_{window}'] = df['close'].pct_change(window)
        
        # 9. Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
        
        # 10. Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
        
        # 11. Time features
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        elif 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
            df['hour'] = dates.dt.hour
            df['day_of_week'] = dates.dt.dayofweek
            df['day_of_month'] = dates.dt.day
            df['is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
        else:
            logging.warning("No timestamp information found for time features")
        
        # 12. Sentiment Features (Optional)
        if sentiment_df is not None and not sentiment_df.empty:
            self._create_sentiment_features(df, sentiment_df)
        
        # Clean up infinite values
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
    
    # ========== NEW ADVANCED INDICATORS ==========
    
    def _add_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        """
        Add Stochastic Oscillator (%K and %D)
        - %K: (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        - %D: 3-period SMA of %K
        Great for identifying oversold/overbought conditions
        """
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        # Avoid division by zero
        range_hl = (highest_high - lowest_low).replace(0, np.nan)
        
        df['stoch_k'] = ((df['close'] - lowest_low) / range_hl) * 100
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        # Stochastic divergence (difference between %K and %D)
        df['stoch_divergence'] = df['stoch_k'] - df['stoch_d']
    
    def _add_williams_r(self, df: pd.DataFrame, period: int = 14):
        """
        Add Williams %R indicator
        Lead indicator for reversals, ranges from -100 to 0
        - Above -20: Overbought
        - Below -80: Oversold
        """
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        range_hl = (highest_high - lowest_low).replace(0, np.nan)
        
        df['williams_r'] = ((highest_high - df['close']) / range_hl) * -100
    
    def _add_obv(self, df: pd.DataFrame):
        """
        Add On-Balance Volume (OBV)
        Tracks cumulative buying/selling pressure based on volume
        Rising OBV = accumulation, Falling OBV = distribution
        """
        obv = np.where(df['close'] > df['close'].shift(1), df['volume'],
                       np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
        df['obv'] = pd.Series(obv, index=df.index).cumsum()
        
        # OBV momentum (rate of change)
        df['obv_sma'] = df['obv'].rolling(20).mean()
        df['obv_ratio'] = df['obv'] / df['obv_sma'].replace(0, np.nan)
        
        # OBV trend
        df['obv_change'] = df['obv'].pct_change(5)
    
    def _add_ichimoku(self, df: pd.DataFrame):
        """
        Add Ichimoku Cloud components
        Popular for crypto trend analysis
        
        Components:
        - Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        - Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        - Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
        - Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
        - Chikou Span: Close shifted back 26 periods
        """
        # Tenkan-sen (Conversion Line) - 9 periods
        period9_high = df['high'].rolling(window=9).max()
        period9_low = df['low'].rolling(window=9).min()
        df['ichimoku_tenkan'] = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line) - 26 periods
        period26_high = df['high'].rolling(window=26).max()
        period26_low = df['low'].rolling(window=26).min()
        df['ichimoku_kijun'] = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A)
        df['ichimoku_senkou_a'] = (df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2
        
        # Senkou Span B (Leading Span B) - 52 periods
        period52_high = df['high'].rolling(window=52).max()
        period52_low = df['low'].rolling(window=52).min()
        df['ichimoku_senkou_b'] = (period52_high + period52_low) / 2
        
        # Cloud thickness (Kumo)
        df['ichimoku_cloud_thickness'] = df['ichimoku_senkou_a'] - df['ichimoku_senkou_b']
        
        # Price position relative to cloud
        cloud_top = df[['ichimoku_senkou_a', 'ichimoku_senkou_b']].max(axis=1)
        cloud_bottom = df[['ichimoku_senkou_a', 'ichimoku_senkou_b']].min(axis=1)
        
        # 1 = above cloud (bullish), -1 = below cloud (bearish), 0 = in cloud
        df['ichimoku_cloud_position'] = np.where(
            df['close'] > cloud_top, 1,
            np.where(df['close'] < cloud_bottom, -1, 0)
        )
        
        # Tenkan-Kijun cross signal
        df['ichimoku_tk_diff'] = df['ichimoku_tenkan'] - df['ichimoku_kijun']
    
    def _add_adx(self, df: pd.DataFrame, period: int = 14):
        """
        Add Average Directional Index (ADX)
        Measures trend strength (not direction)
        - ADX > 25: Strong trend
        - ADX < 20: Weak trend / ranging market
        
        Also adds +DI and -DI for direction
        """
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate +DM and -DM
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff().abs() * -1
        
        plus_dm = np.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm.abs() > plus_dm) & (minus_dm < 0), minus_dm.abs(), 0)
        
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)
        
        # Smoothed TR, +DM, -DM
        atr = tr.rolling(window=period).mean()
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()
        
        # +DI and -DI
        df['plus_di'] = (plus_dm_smooth / atr.replace(0, np.nan)) * 100
        df['minus_di'] = (minus_dm_smooth / atr.replace(0, np.nan)) * 100
        
        # DX (Directional Index)
        di_diff = np.abs(df['plus_di'] - df['minus_di'])
        di_sum = df['plus_di'] + df['minus_di']
        dx = (di_diff / di_sum.replace(0, np.nan)) * 100
        
        # ADX (smoothed DX)
        df['adx'] = dx.rolling(window=period).mean()
        
        # Trend direction signal: +DI > -DI = bullish
        df['adx_trend_direction'] = np.where(df['plus_di'] > df['minus_di'], 1, -1)
    
    def _add_market_regime(self, df: pd.DataFrame):
        """
        Add Market Regime Detection
        Classifies market into different regimes based on volatility and trend
        
        Regimes:
        - 0: Low volatility, ranging
        - 1: Low volatility, trending
        - 2: High volatility, ranging  
        - 3: High volatility, trending
        """
        # Volatility regime (based on ATR percentile)
        if 'atr' not in df.columns:
            self._add_atr(df)
        
        atr_percentile = df['atr'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        df['volatility_regime'] = np.where(atr_percentile > 0.5, 1, 0)
        
        # Trend regime (based on ADX)
        if 'adx' not in df.columns:
            self._add_adx(df)
        
        df['trend_regime'] = np.where(df['adx'] > 25, 1, 0)
        
        # Combined regime (0-3)
        df['market_regime'] = df['volatility_regime'] + (df['trend_regime'] * 2)
        
        # Volatility cluster detection
        df['volatility_cluster'] = df['atr'].rolling(5).mean() / df['atr'].rolling(20).mean()
        
        # Trend strength score (normalized ADX)
        df['trend_strength'] = df['adx'] / 100

    def _add_order_book_proxies(self, df: pd.DataFrame):
        """
        Add Order Book proxy features using volume data.
        Since real order book data is not available, we proxy using volume patterns.
        
        Features:
        - ob_imbalance_ma12: 12-period MA of volume imbalance proxy
        - ob_imbalance_delta: Change in imbalance over 5 periods
        - ob_spread_zscore: Z-score normalized price spread
        """
        # Volume imbalance proxy based on price direction and volume
        price_direction = np.sign(df['close'] - df['open'])
        ob_imbalance = price_direction * df['volume']
        
        df['ob_imbalance_ma12'] = ob_imbalance.rolling(12).mean()
        df['ob_imbalance_delta'] = df['ob_imbalance_ma12'].diff(5)
        
        # Spread proxy using high-low range normalized
        spread = (df['high'] - df['low']) / df['close']
        spread_mean = spread.rolling(20).mean()
        spread_std = spread.rolling(20).std().replace(0, np.nan)
        df['ob_spread_zscore'] = (spread - spread_mean) / spread_std
    
    def _add_funding_rate_proxies(self, df: pd.DataFrame):
        """
        Add Funding Rate proxy features.
        Since real funding rate data is not available, we proxy using momentum patterns.
        
        Features:
        - funding_rate_zscore: Z-score of funding rate proxy (based on momentum)
        - funding_cum_3d: 3-day cumulative funding proxy
        """
        # Funding rate proxy based on short-term momentum imbalance
        # Positive momentum = long pressure = positive funding
        momentum = df['close'].pct_change(4)  # ~4h momentum for funding
        funding_proxy = momentum * 100  # Scale to typical funding range
        
        funding_mean = funding_proxy.rolling(24).mean()
        funding_std = funding_proxy.rolling(24).std().replace(0, np.nan)
        df['funding_rate_zscore'] = (funding_proxy - funding_mean) / funding_std
        
        # Cumulative funding over 3 days (~72 periods for hourly)
        df['funding_cum_3d'] = funding_proxy.rolling(72).sum()
    
    def _add_open_interest_proxies(self, df: pd.DataFrame):
        """
        Add Open Interest proxy features.
        Using volume as a proxy for OI changes.
        
        Features:
        - oi_pct_change: Volume-based OI change proxy
        - oi_sentiment: Sentiment based on OI proxy direction vs price direction
        """
        # OI proxy using volume changes
        volume_ma = df['volume'].rolling(20).mean()
        oi_proxy = df['volume'] / volume_ma.replace(0, np.nan)
        
        df['oi_pct_change'] = oi_proxy.pct_change(5)
        
        # OI sentiment: high volume with price up = bullish, high volume with price down = bearish
        price_direction = np.sign(df['close'] - df['close'].shift(1))
        volume_spike = (oi_proxy > 1.5).astype(int)
        df['oi_sentiment'] = price_direction * volume_spike
    
    def _add_arbitrage_features(self, df: pd.DataFrame):
        """
        Add Arbitrage-related features.
        Using price patterns to detect potential arbitrage conditions.
        
        Features:
        - arb_exch_delta_pct: Exchange delta percentage proxy
        - arb_delta_zscore: Z-score of arbitrage delta
        - ext_price_roc: External price rate of change (lag proxy for external markets)
        - arb_roc_divergence: ROC divergence between timeframes
        """
        # Arbitrage delta proxy using price deviation from moving average
        ma_20 = df['close'].rolling(20).mean()
        arb_delta = (df['close'] - ma_20) / ma_20.replace(0, np.nan) * 100
        
        df['arb_exch_delta_pct'] = arb_delta
        
        arb_mean = arb_delta.rolling(50).mean()
        arb_std = arb_delta.rolling(50).std().replace(0, np.nan)
        df['arb_delta_zscore'] = (arb_delta - arb_mean) / arb_std
        
        # External price ROC proxy (lagged ROC as if from another exchange)
        df['ext_price_roc'] = df['close'].pct_change(10) * 100
        
        # ROC divergence between short and long term
        roc_short = df['close'].pct_change(5)
        roc_long = df['close'].pct_change(20)
        df['arb_roc_divergence'] = roc_short - roc_long
    
    def _add_correlation_features(self, df: pd.DataFrame):
        """
        Add Correlation-based features.
        Using auto-correlation and price pattern correlations.
        
        Features:
        - corr_btc_eth: Correlation proxy (auto-correlation of returns)
        - rel_strength: Relative strength vs historical performance
        - corr_divergence: Correlation divergence signal
        """
        returns = df['close'].pct_change()
        
        # Auto-correlation as proxy for BTC/ETH correlation 
        # (in real implementation, would need multi-symbol data)
        df['corr_btc_eth'] = returns.rolling(20).apply(
            lambda x: x[:-1].corr(pd.Series(x[1:])) if len(x) > 1 else np.nan, 
            raw=False
        )
        
        # Relative strength: current return vs rolling average
        avg_return = returns.rolling(50).mean()
        std_return = returns.rolling(50).std().replace(0, np.nan)
        df['rel_strength'] = (returns - avg_return) / std_return
        
        # Correlation divergence: difference between short and long term correlations
        short_corr = returns.rolling(10).apply(
            lambda x: x[:-1].corr(pd.Series(x[1:])) if len(x) > 1 else np.nan,
            raw=False
        )
        long_corr = returns.rolling(30).apply(
            lambda x: x[:-1].corr(pd.Series(x[1:])) if len(x) > 1 else np.nan,
            raw=False
        )
        df['corr_divergence'] = short_corr - long_corr

    def _create_sentiment_features(self, df: pd.DataFrame, sentiment_df: pd.DataFrame):
        """
        Merge and create sentiment features.
        Uses merge_asof to align sentiment data with price candles.
        """
        try:
            # Ensure timestamps are set and sorted
            df_temp = df.copy()
            sent_temp = sentiment_df.copy()
            
            if 'timestamp' not in df_temp.columns and isinstance(df_temp.index, pd.DatetimeIndex):
                df_temp['timestamp'] = df_temp.index
            
            # Ensure both are datetime
            df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
            sent_temp['timestamp'] = pd.to_datetime(sent_temp['timestamp'])
            
            df_temp = df_temp.sort_values('timestamp')
            sent_temp = sent_temp.sort_values('timestamp')
            
            # Merge using merge_asof (backward direction: match with previous known sentiment)
            merged = pd.merge_asof(
                df_temp, 
                sent_temp[['timestamp', 'composite_score', 'twitter_volume', 'reddit_volume']], 
                on='timestamp', 
                direction='backward',
                tolerance=pd.Timedelta('4h') # Allow up to 4h staleness
            )
            
            # Generate features on the merged result
            # Fill missing with 0 (neutral) if no sentiment found
            merged['sentiment_score'] = merged['composite_score'].fillna(0)
            merged['social_volume'] = (merged['twitter_volume'] + merged['reddit_volume']).fillna(0)
            
            # Rolling averages
            for window in [1, 4, 12, 24]:
                merged[f'sentiment_ma_{window}'] = merged['sentiment_score'].rolling(window=window, min_periods=1).mean()
                merged[f'volume_ma_{window}'] = merged['social_volume'].rolling(window=window, min_periods=1).mean()
            
            # Sentiment Momentum
            merged['sentiment_momentum'] = merged['sentiment_score'] - merged['sentiment_ma_4']
            
            # Assign back to original dataframe
            # Note: This relies on the index being preserved or aligned.
            # Ideally create_features returns the new df.
            # Here we are modifying df in place in standard usage, but merge_asof created a new object.
            
            feature_cols = [c for c in merged.columns if 'sentiment' in c or 'social' in c]
            for col in feature_cols:
                # Align back to original df based on index if possible, or assume lengths match if sorted
                if len(merged) == len(df):
                    df[col] = merged[col].values
                else:
                    # Fallback if lengths differ (shouldn't happen with merge_asof on all rows)
                    logging.warning("Length mismatch in sentiment feature creation")
            
        except Exception as e:
            logging.error(f"Error creating sentiment features: {e}")
