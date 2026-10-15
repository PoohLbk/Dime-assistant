from datetime import datetime
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import requests
from scipy.signal import find_peaks
import yfinance as yf

nltk.download('vader_lexicon', quiet=True)
app = Flask(__name__)


def calculate_rsi(series, period=14):
  delta = series.diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
  rs = gain / loss
  return 100 - (100 / (1 + rs))


@app.route('/')
def home():
  return render_template('index.html')


@app.route('/api/analyze')
def analyze():
  ticker_symbol = request.args.get('ticker', 'PLTR').upper()
  period = request.args.get('period', '6mo')
  interval = request.args.get('interval', '1d')

  ticker_map = {
      'XAUUSD': 'GC=F',
      'XAU/USD': 'GC=F',
      'GOLD': 'GC=F',
      'BTCUSD': 'BTC-USD',
  }
  clean_symbol = ticker_map.get(ticker_symbol, ticker_symbol)

  ticker_obj = yf.Ticker(clean_symbol)
  df = ticker_obj.history(period=period, interval=interval)

  if df.empty:
    return jsonify({'error': f'Data not found for {ticker_symbol}'}), 404

  df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
  df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
  df['RSI'] = calculate_rsi(df['Close'])

  highs, lows = df['High'].values, df['Low'].values
  peaks_high, _ = find_peaks(highs, distance=10, prominence=2)
  peaks_low, _ = find_peaks(-lows, distance=10, prominence=2)

  last_h_idx = peaks_high[-1] if len(peaks_high) > 0 else np.argmax(highs)
  last_l_idx = peaks_low[-1] if len(peaks_low) > 0 else np.argmin(lows)
  swing_high, swing_low = highs[last_h_idx], lows[last_l_idx]
  diff = swing_high - swing_low
  is_uptrend = last_h_idx > last_l_idx

  gz_top = (
      swing_high - (0.500 * diff)
      if is_uptrend
      else swing_low + (0.618 * diff)
  )
  gz_bottom = (
      swing_high - (0.618 * diff)
      if is_uptrend
      else swing_low + (0.500 * diff)
  )

  current_price = float(df['Close'].iloc[-1])
  current_rsi = float(df['RSI'].iloc[-1])
  in_gz = min(gz_top, gz_bottom) <= current_price <= max(gz_top, gz_bottom)

  # Chart data formatting
  chart_data = []
  for idx, row in df.iterrows():
    chart_data.append({
        'time': idx.strftime('%Y-%m-%d'),
        'open': float(row['Open']),
        'high': float(row['High']),
        'low': float(row['Low']),
        'close': float(row['Close']),
        'ema20': float(row['EMA_20']) if not pd.isna(row['EMA_20']) else None,
        'ema50': float(row['EMA_50']) if not pd.isna(row['EMA_50']) else None,
        'rsi': float(row['RSI']) if not pd.isna(row['RSI']) else None,
    })

  return jsonify({
      'ticker': ticker_symbol,
      'current_price': current_price,
      'rsi': current_rsi,
      'in_gz': in_gz,
      'gz_min': min(gz_top, gz_bottom),
      'gz_max': max(gz_top, gz_bottom),
      'is_uptrend': is_uptrend,
      'chart_data': chart_data,
  })


if __name__ == '__main__':
  app.run(debug=True, port=5000)
