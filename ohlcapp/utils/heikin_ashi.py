# utils/heikin_ashi.py

import pandas as pd
import numpy as np


def calculate_heikin_ashi(df):
    """
    Convert regular OHLC data to Heikin-Ashi candles.
    
    Heikin-Ashi formula:
    - HA_Close = (Open + High + Low + Close) / 4
    - HA_Open = (Previous HA_Open + Previous HA_Close) / 2
    - HA_High = Max(High, HA_Open, HA_Close)
    - HA_Low = Min(Low, HA_Open, HA_Close)
    
    Args:
        df: DataFrame with OHLC columns and datetime index
    
    Returns:
        DataFrame with Heikin-Ashi OHLC values
    """
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying original
    ha_df = df.copy()
    
    # Initialize arrays for Heikin-Ashi values
    ha_close = np.zeros(len(df))
    ha_open = np.zeros(len(df))
    ha_high = np.zeros(len(df))
    ha_low = np.zeros(len(df))
    
    # Calculate HA_Close for all rows
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # First row: HA_Open = (Open + Close) / 2
    ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    
    # Remaining rows: HA_Open = (Previous HA_Open + Previous HA_Close) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    
    # Calculate HA_High and HA_Low
    ha_high = np.maximum(df['high'].values, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(df['low'].values, np.minimum(ha_open, ha_close))
    
    # Update the dataframe
    ha_df['open'] = ha_open
    ha_df['high'] = ha_high
    ha_df['low'] = ha_low
    ha_df['close'] = ha_close
    
    return ha_df


def is_heikin_ashi_bullish(row):
    """
    Determine if a Heikin-Ashi candle is bullish.
    
    Args:
        row: DataFrame row with HA OHLC values
    
    Returns:
        bool: True if bullish (close >= open)
    """
    return row['close'] >= row['open']


def get_heikin_ashi_strength(df):
    """
    Calculate trend strength based on consecutive HA candles.
    
    Args:
        df: Heikin-Ashi DataFrame
    
    Returns:
        Series: Consecutive bullish/bearish count (positive=bullish, negative=bearish)
    """
    if df is None or df.empty:
        return pd.Series(dtype=int)
    
    # Determine if each candle is bullish
    is_bullish = df['close'] >= df['open']
    
    # Calculate consecutive count
    strength = pd.Series(0, index=df.index, dtype=int)
    current_count = 0
    
    for i in range(len(df)):
        if i == 0:
            current_count = 1 if is_bullish.iloc[i] else -1
        else:
            if is_bullish.iloc[i] == is_bullish.iloc[i-1]:
                # Same direction, increment
                current_count = current_count + 1 if is_bullish.iloc[i] else current_count - 1
            else:
                # Direction change, reset
                current_count = 1 if is_bullish.iloc[i] else -1
        
        strength.iloc[i] = current_count
    
    return strength