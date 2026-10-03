import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def detect_swings_and_golden_zone(ticker_symbol: str, period="6mo", distance=10, prominence=2):
    df = yf.download(ticker_symbol, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    highs = df['High'].values
    lows = df['Low'].values
    
    peaks_high, _ = find_peaks(highs, distance=distance, prominence=prominence)
    peaks_low, _ = find_peaks(-lows, distance=distance, prominence=prominence)
    
    last_high_idx = peaks_high[-1] if len(peaks_high) > 0 else np.argmax(highs)
    last_low_idx = peaks_low[-1] if len(peaks_low) > 0 else np.argmin(lows)
    
    swing_high = highs[last_high_idx]
    swing_low = lows[last_low_idx]
    current_price = df['Close'].iloc[-1]
    
    diff = swing_high - swing_low
    is_uptrend = last_high_idx > last_low_idx
    
    if is_uptrend:
        gz_top = swing_high - (0.500 * diff)
        gz_bottom = swing_high - (0.618 * diff)
    else:
        gz_bottom = swing_low + (0.500 * diff)
        gz_top = swing_low + (0.618 * diff)

    fib_levels = {
        'Swing High': swing_high,
        'Golden Zone Top': gz_top,
        'Golden Zone Bottom': gz_bottom,
        'Swing Low': swing_low
    }
    
    in_golden_zone = min(gz_top, gz_bottom) <= current_price <= max(gz_top, gz_bottom)
    
    return df, peaks_high, peaks_low, fib_levels, current_price, in_golden_zone, is_uptrend
