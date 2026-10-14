from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from scipy.signal import find_peaks
import streamlit as st
import yfinance as yf

# --- 1. INITIALIZATION & CACHING ---


@st.cache_resource
def init_nltk():
  nltk.download("vader_lexicon", quiet=True)


init_nltk()

# --- 2. SET PAGE CONFIG & CUSTOM CSS ---
st.set_page_config(
    page_title="Emerald Ultimate Analytics",
    layout="wide",
)

custom_css = """
<style>
    .stApp { background-color: #0B0E11; color: #EAEAEA; }
    section[data-testid="stSidebar"] { background-color: #12161C; border-right: 1px solid #1E232A; }
    div[data-testid="stMetric"] {
        background-color: #161B22; border: 1px solid #008000;
        border-radius: 8px; padding: 12px; box-shadow: 0px 4px 12px rgba(0, 128, 0, 0.15);
    }
    div[data-testid="stMetricLabel"] { color: #8B949E !important; font-size: 0.8rem !important; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700 !important; font-size: 1.2rem !important; }
    div.stButton > button {
        background-color: #008000; color: #FFFFFF; border: none;
        border-radius: 6px; font-weight: 600; padding: 0.6rem 1.2rem; width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover { background-color: #00B300; box-shadow: 0px 0px 10px rgba(0, 255, 0, 0.4); color: #FFFFFF; }
    button[data-baseweb="tab"] { color: #8B949E !important; font-weight: 600; }
    button[aria-selected="true"] { color: #00FF00 !important; border-bottom-color: #00FF00 !important; }
    .emerald-accent { color: #00FF00; font-weight: bold; }
    .trend-card {
        padding: 12px; border-radius: 8px; font-weight: bold; text-align: center;
        margin-bottom: 15px; font-size: 1.1rem;
    }
    .trend-bullish { background-color: rgba(0, 230, 118, 0.15); border: 1px solid #00E676; color: #00E676; }
    .trend-bearish { background-color: rgba(255, 82, 82, 0.15); border: 1px solid #FF5252; color: #FF5252; }
    .trend-sideways { background-color: rgba(255, 215, 0, 0.15); border: 1px solid #FFD700; color: #FFD700; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 3. HELPER & TECHNICAL ANALYSIS FUNCTIONS ---


def calculate_rsi(series, period=14):
  delta = series.diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
  rs = gain / loss
  return 100 - (100 / (1 + rs))


def detect_divergence(df, peaks_high, peaks_low):
  divergence_signals = []
  if len(peaks_high) >= 2:
    h1, h2 = peaks_high[-2], peaks_high[-1]
    price_h1, price_h2 = df["High"].iloc[h1], df["High"].iloc[h2]
    rsi_h1, rsi_h2 = df["RSI"].iloc[h1], df["RSI"].iloc[h2]
    if price_h2 > price_h1 and rsi_h2 < rsi_h1:
      divergence_signals.append(
          "Regular Bearish Divergence (สัญญาณกลับตัวลง)"
      )
    elif price_h2 < price_h1 and rsi_h2 > rsi_h1:
      divergence_signals.append("Hidden Bullish Divergence (เทรนด์ขึ้นต่อ)")

  if len(peaks_low) >= 2:
    l1, l2 = peaks_low[-2], peaks_low[-1]
    price_l1, price_l2 = df["Low"].iloc[l1], df["Low"].iloc[l2]
    rsi_l1, rsi_l2 = df["RSI"].iloc[l1], df["RSI"].iloc[l2]
    if price_l2 < price_l1 and rsi_l2 > rsi_l1:
      divergence_signals.append(
          "Regular Bullish Divergence (สัญญาณกลับตัวขึ้น)"
      )
    elif price_l2 > price_l1 and rsi_l2 < rsi_l1:
      divergence_signals.append("Hidden Bearish Divergence (เทรนด์ลงต่อ)")

  return divergence_signals


def analyze_trend_state(
    df, current_price, ema_20, ema_50, peaks_high, peaks_low
):
  """ฟังก์ชันวิเคราะห์สภาวะแนวโน้มราคา (Bullish / Bearish / Sideways)"""
  # 1. ดูโครงสร้างการเรียงตัวของเส้น EMA
  ema_uptrend = (current_price > ema_20) and (ema_20 > ema_50)
  ema_downtrend = (current_price < ema_20) and (ema_20 < ema_50)

  # 2. ดูการทำ High / Low ล่าสุด (Structure)
  if len(peaks_high) >= 2 and len(peaks_low) >= 2:
    h1, h2 = df["High"].iloc[peaks_high[-2]], df["High"].iloc[peaks_high[-1]]
    l1, l2 = df["Low"].iloc[peaks_low[-2]], df["Low"].iloc[peaks_low[-1]]

    higher_high = h2 > h1
    higher_low = l2 > l1
    lower_high = h2 < h1
    lower_low = l2 < l1

    if ema_uptrend or (higher_high and higher_low):
      return (
          "BULLISH TREND (แนวโน้มขาขึ้น)",
          "trend-bullish",
          "🚀 ราคาเคลื่อนที่ทำจุดสูงใหม่ (Higher High) และยืนเหนือเส้น EMA",
      )
    elif ema_downtrend or (lower_high and lower_low):
      return (
          "BEARISH TREND (แนวโน้มขาลง)",
          "trend-bearish",
          "🔻 ราคาเคลื่อนที่ทำจุดต่ำใหม่ (Lower Low) และอยู่ใต้เส้น EMA",
      )

  # 3. หากกรอบราคากว้างผันผวนสลับไปมา หรือ EMA พันกัน
  if ema_uptrend:
    return (
        "BULLISH TREND (แนวโน้มขาขึ้น)",
        "trend-bullish",
        "🚀 ราคายังทรงตัวได้ดีเหนือเส้นค่าเฉลี่ย EMA",
    )
  elif ema_downtrend:
    return (
        "BEARISH TREND (แนวโน้มขาลง)",
        "trend-bearish",
        "🔻 ราคายังคงถูกกดดันใต้เส้นค่าเฉลี่ย EMA",
    )

  return (
      "SIDEWAYS / RANGING (กรอบสะสมกำลัง)",
      "trend-sideways",
      "↔️ ราคาเคลื่อนที่ออกด้านข้างในกรอบผันผวน ยังไม่มีทิศทางชัดเจน",
  )


@st.cache_data(ttl=60)
def fetch_and_analyze(
    ticker_symbol: str,
    period="6mo",
    interval="1d",
    distance=10,
    prominence=2,
):
  ticker_map = {
      "XAUUSD": "GC=F",
      "XAU/USD": "GC=F",
      "GOLD": "GC=F",
      "SILVER": "SI=F",
      "BTCUSD": "BTC-USD",
      "ETHUSD": "ETH-USD",
  }
  clean_symbol = ticker_map.get(
      ticker_symbol.upper().strip(), ticker_symbol.upper().strip()
  )

  ticker_obj = yf.Ticker(clean_symbol)
  df = ticker_obj.history(period=period, interval=interval)

  if df.empty:
    return None, {}

  # คำนวณ DIVIDEND YIELD
  dividend_yield_pct = 0.0
  try:
    info = ticker_obj.info
    div_yield = info.get("dividendYield")

    if div_yield is None:
      div_yield = info.get("yield")

    if div_yield is not None:
      dividend_yield_pct = (
          div_yield * 100 if div_yield < 1 else float(div_yield)
      )
    else:
      divs = ticker_obj.dividends
      if not divs.empty:
        one_year_ago = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(
            days=365
        )
        last_year_divs = divs[divs.index >= one_year_ago].sum()
        current_price = df["Close"].iloc[-1]
        dividend_yield_pct = (
            (last_year_divs / current_price) * 100
            if current_price > 0
            else 0.0
        )
  except Exception:
    dividend_yield_pct = 0.0

  df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
  df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
  df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
  df["RSI"] = calculate_rsi(df["Close"])

  highs, lows = df["High"].values, df["Low"].values
  peaks_high, _ = find_peaks(highs, distance=distance, prominence=prominence)
  peaks_low, _ = find_peaks(-lows, distance=distance, prominence=prominence)

  last_h_idx = peaks_high[-1] if len(peaks_high) > 0 else np.argmax(highs)
  last_l_idx = peaks_low[-1] if len(peaks_low) > 0 else np.argmin(lows)
  swing_high, swing_low = highs[last_h_idx], lows[last_l_idx]
  diff = swing_high - swing_low
  is_uptrend = last_h_idx > last_l_idx

  if is_uptrend:
    gz_top = swing_high - (0.500 * diff)
    gz_bottom = swing_high - (0.618 * diff)
  else:
    gz_bottom = swing_low + (0.500 * diff)
    gz_top = swing_low + (0.618 * diff)

  # --- คำนวณ DEMAND / SUPPLY ZONES ---
  supply_top = swing_high
  supply_bottom = max(
      df["Open"].iloc[last_h_idx], df["Close"].iloc[last_h_idx]
  )

  demand_bottom = swing_low
  demand_top = min(df["Open"].iloc[last_l_idx], df["Close"].iloc[last_l_idx])

  current_price = df["Close"].iloc[-1]
  current_rsi = df["RSI"].iloc[-1]
  in_gz = min(gz_top, gz_bottom) <= current_price <= max(gz_top, gz_bottom)

  ema_20_curr, ema_50_curr = df["EMA_20"].iloc[-1], df["EMA_50"].iloc[-1]
  ema_20_prev, ema_50_prev = df["EMA_20"].iloc[-2], df["EMA_50"].iloc[-2]

  if ema_20_prev < ema_50_prev and ema_20_curr >= ema_50_curr:
    ema_signal = "GOLDEN CROSS (EMA20 ตัดขึ้น EMA50)"
  elif ema_20_prev > ema_50_prev and ema_20_curr <= ema_50_curr:
    ema_signal = "DEATH CROSS (EMA20 ตัดลง EMA50)"
  elif ema_20_curr > ema_50_curr:
    ema_signal = "BULLISH ALIGNMENT (EMA20 > EMA50)"
  else:
    ema_signal = "BEARISH ALIGNMENT (EMA20 < EMA50)"

  # วิเคราะห์สภาวะแนวโน้มราคา (Trend Status)
  trend_label, trend_css, trend_desc = analyze_trend_state(
      df, current_price, ema_20_curr, ema_50_curr, peaks_high, peaks_low
  )

  divergence_signals = detect_divergence(df, peaks_high, peaks_low)

  analysis_results = {
      "current_price": current_price,
      "current_rsi": current_rsi,
      "dividend_yield": dividend_yield_pct,
      "is_uptrend": is_uptrend,
      "in_gz": in_gz,
      "gz_min": min(gz_top, gz_bottom),
      "gz_max": max(gz_top, gz_bottom),
      "supply_top": supply_top,
      "supply_bottom": supply_bottom,
      "demand_top": demand_top,
      "demand_bottom": demand_bottom,
      "trend_label": trend_label,
      "trend_css": trend_css,
      "trend_desc": trend_desc,
      "ema_signal": ema_signal,
      "divergences": divergence_signals,
      "peaks_high": peaks_high,
      "peaks_low": peaks_low,
  }

  return df, analysis_results


def scan_golden_zone_stocks(ticker_list, period="6mo", interval="1d"):
  results = []
  progress_bar = st.progress(0)
  status_text = st.empty()
  total = len(ticker_list)

  for idx, symbol in enumerate(ticker_list):
    status_text.markdown(
        f"Scanning: `<span class='emerald-accent'>{symbol}</span>`"
        f" ({idx+1}/{total})",
        unsafe_allow_html=True,
    )
    progress_bar.progress((idx + 1) / total)

    try:
      df = yf.download(symbol, period=period, interval=interval, progress=False)
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

      if df.empty or len(df) < 30:
        continue

      highs, lows, close_prices = (
          df["High"].values,
          df["Low"].values,
          df["Close"].values,
      )
      peaks_high, _ = find_peaks(highs, distance=10, prominence=2)
      peaks_low, _ = find_peaks(-lows, distance=10, prominence=2)

      last_h_idx = peaks_high[-1] if len(peaks_high) > 0 else np.argmax(highs)
      last_l_idx = peaks_low[-1] if len(peaks_low) > 0 else np.argmin(lows)

      swing_high, swing_low, current_price = (
          highs[last_h_idx],
          lows[last_l_idx],
          close_prices[-1],
      )
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

      gz_min, gz_max = min(gz_top, gz_bottom), max(gz_top, gz_bottom)
      in_gz = gz_min <= current_price <= gz_max

      df["RSI"] = calculate_rsi(df["Close"])
      rsi = df["RSI"].iloc[-1]

      if in_gz:
        results.append({
            "Symbol": symbol,
            "Price (\$/฿)": round(current_price, 2),
            "Golden Zone Min": round(gz_min, 2),
            "Golden Zone Max": round(gz_max, 2),
            "Trend": (
                "Uptrend (Retracement)" if is_uptrend else "Downtrend (Bounce)"
            ),
            "RSI (14)": round(rsi, 1),
            "RSI Status": (
                "Oversold"
                if rsi <= 35
                else ("Overbought" if rsi >= 65 else "Neutral")
            ),
        })
    except Exception:
      continue

  progress_bar.empty()
  status_text.empty()
  return pd.DataFrame(results)


# --- 4. GMAIL ALERT & REAL-TIME NEWS SCRAPER ---


def send_gmail_alert(
    sender_email,
    app_password,
    receiver_email,
    ticker,
    current_price,
    timeframe,
    signals,
    gz_min,
    gz_max,
):
  if not sender_email or not app_password or not receiver_email or not signals:
    return False, "กรุณากรอกข้อมูล Gmail ให้ครบถ้วน"

  subject = f"EMERALD BUY ALERT: {ticker} (\${current_price:.2f})"
  signals_html = "".join(
      [f"<li style='padding: 4px 0; color: #00FF00;'><b>{s}</b></li>" for s in signals]
  )

  body = f"""
    <html>
    <body style="background-color: #0B0E11; color: #EAEAEA; font-family: Arial, sans-serif; padding: 20px;">
        <div style="max-width: 600px; margin: auto; background-color: #12161C; border: 1px solid #008000; border-radius: 10px; padding: 25px;">
            <h2 style="color: #00FF00; margin-top: 0;">EMERALD BUY SIGNAL ALERT</h2>
            <p style="font-size: 16px; color: #FFFFFF;">พบสัญญาณเข้าซื้อสำหรับ <b>{ticker}</b></p>
            <hr style="border: 0.5px solid #1E232A;">
            <table style="width: 100%; margin: 15px 0; font-size: 14px;">
                <tr><td style="color: #8B949E;">Symbol:</td><td style="color: #FFFFFF; font-weight: bold;">{ticker}</td></tr>
                <tr><td style="color: #8B949E;">Last Price:</td><td style="color: #00FF00; font-weight: bold;">\${current_price:.2f}</td></tr>
                <tr><td style="color: #8B949E;">Timeframe:</td><td style="color: #FFFFFF;">{timeframe}</td></tr>
                <tr><td style="color: #8B949E;">Golden Zone:</td><td style="color: #00FF00;">${gz_min:.2f} -${gz_max:.2f}</td></tr>
            </table>
            <h4 style="color: #FFFFFF; margin-bottom: 5px;">Signals Detected:</h4>
            <ul style="background-color: #161B22; padding: 15px 25px; border-radius: 6px; list-style-type: square;">
                {signals_html}
            </ul>
        </div>
    </body>
    </html>
    """

  msg = MIMEMultipart()
  msg["From"] = sender_email
  msg["To"] = receiver_email
  msg["Subject"] = subject
  msg.attach(MIMEText(body, "html"))

  try:
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email.strip(), app_password.strip())
    server.send_message(msg)
    server.quit()
    return True, "ส่งอีเมลสำเร็จ!"
  except Exception as e:
    return False, f"เกิดข้อผิดพลาด: {str(e)}"


@st.cache_data(ttl=60)
def scrape_news_vader(ticker_symbol: str):
  ticker_map = {
      "GC=F": "Gold market",
      "XAUUSD": "Gold market",
      "XAU/USD": "Gold market",
      "SI=F": "Silver market",
      "BTC-USD": "Bitcoin",
      "ETH-USD": "Ethereum",
  }
  search_term = ticker_map.get(ticker_symbol.upper().strip(), ticker_symbol)

  sia = SentimentIntensityAnalyzer()
  parsed_news = []

  try:
    rss_url = f"https://news.google.com/rss/search?q={search_term}+stock+when:30d&hl=en-US&gl=US&ceid=US:en"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
    }
    res = requests.get(rss_url, headers=headers, timeout=5)

    if res.status_code == 200:
      root = ET.fromstring(res.content)
      for item in root.findall(".//item")[:30]:
        title = item.find("title").text
        clean_title = (
            title.rsplit(" - ", 1)[0] if " - " in title else title
        )

        raw_date = item.find("pubDate").text
        dt_obj = None
        try:
          dt_obj = datetime.strptime(raw_date[5:25], "%d %b %Y %H:%M:%S")
          formatted_time = dt_obj.strftime("%b-%d %I:%M%p")
        except Exception:
          formatted_time = raw_date[5:22]

        score = sia.polarity_scores(clean_title)["compound"]
        sentiment = (
            "BULLISH"
            if score >= 0.05
            else ("BEARISH" if score <= -0.05 else "NEUTRAL")
        )

        parsed_news.append({
            "dt": dt_obj if dt_obj else datetime.min,
            "Time": formatted_time,
            "Headline": clean_title,
            "Sentiment": sentiment,
            "Score": score,
        })

      if parsed_news:
        df_news = pd.DataFrame(parsed_news)
        df_news = df_news.sort_values(by="dt", ascending=False).drop(
            columns=["dt"]
        )
        return df_news
  except Exception:
    pass

  return pd.DataFrame()


# --- 5. SIDEBAR CONTROL ---
st.sidebar.markdown(
    "<h2 style='color: #00FF00;'>CONTROL PANEL</h2>", unsafe_allow_html=True
)

ticker = st.sidebar.text_input(
    "SYMBOL (e.g. NVDA, GC=F, PLTR, WDC, MU)", value="PLTR"
).upper()
interval = st.sidebar.selectbox(
    "TIMEFRAME", ["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"], index=4
)

if interval == "1m":
  period_options, default_p = ["1d", "5d", "7d"], 0
elif interval in ["5m", "15m", "1h"]:
  period_options, default_p = ["1d", "5d", "1mo", "60d"], 2
else:
  period_options, default_p = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], 3

period = st.sidebar.selectbox(
    "LOOKBACK PERIOD", period_options, index=default_p
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<h4 style='color: #8B949E;'>Display Settings</h4>",
    unsafe_allow_html=True,
)
show_demand_supply = st.sidebar.checkbox(
    "Show Demand / Supply Zones", value=True
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<h4 style='color: #8B949E;'>Gmail Alert Settings</h4>",
    unsafe_allow_html=True,
)

sender_email = st.sidebar.text_input(
    "Sender Gmail", value="อีเมลผู้ส่ง@gmail.com"
)
app_password = st.sidebar.text_input(
    "App Password (16-digits)", value="รหัสผ่านแอป16หลัก", type="password"
)
receiver_email = st.sidebar.text_input(
    "Receiver Email", value="อีเมลผู้รับ@gmail.com"
)
enable_email_alert = st.sidebar.checkbox("Enable Gmail Alert", value=True)

# --- 6. MAIN CONTENT ---
st.markdown(
    "<h1>EMERALD <span class='emerald-accent'>STOCK & COMMODITY"
    " ANALYTICS</span></h1>",
    unsafe_allow_html=True,
)

with st.spinner(f"Analyzing {ticker}..."):
  df, res = fetch_and_analyze(ticker, period=period, interval=interval)

  if df is None or df.empty:
    st.error(f"ไม่พบข้อมูลสำหรับสัญลักษณ์ {ticker}")
  else:
    # METRICS BAR
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PRICE", f"\${res['current_price']:.2f}")
    c2.metric("RSI (14)", f"{res['current_rsi']:.1f}")
    c3.metric("DIV YIELD", f"{res['dividend_yield']:.2f}%")
    c4.metric("GOLDEN ZONE", f"${res['gz_min']:.2f} -${res['gz_max']:.2f}")
    c5.metric("GZ STATUS", "IN ZONE" if res["in_gz"] else "OUTSIDE ZONE")

    st.markdown("<br>", unsafe_allow_html=True)

    # TREND STATUS BANNER
    st.markdown(
        f"<div class='trend-card {res['trend_css']}'>"
        f"{res['trend_label']}<br><span style='font-size: 0.9rem; font-weight:"
        f" normal;'>{res['trend_desc']}</span></div>",
        unsafe_allow_html=True,
    )

    # GMAIL ALERT TRIGGER
    bullish_signals = []
    if res["in_gz"]:
      bullish_signals.append(
          f"Price is in Golden Zone (\${res['gz_min']:.2f} -"
          f" \${res['gz_max']:.2f})"
      )
    for div in res["divergences"]:
      if "Bullish" in div:
        bullish_signals.append(div)
    if "GOLDEN CROSS" in res["ema_signal"]:
      bullish_signals.append("EMA 20 Crossed Above EMA 50 (Golden Cross)")
    if res["current_rsi"] <= 30:
      bullish_signals.append(f"RSI Oversold ({res['current_rsi']:.1f})")

    if res["in_gz"] and len(bullish_signals) > 1 and enable_email_alert:
      if sender_email and app_password and receiver_email:
        success, msg_text = send_gmail_alert(
            sender_email,
            app_password,
            receiver_email,
            ticker,
            res["current_price"],
            interval,
            bullish_signals,
            res["gz_min"],
            res["gz_max"],
        )
        if success:
          st.success("Gmail Alert Sent! ส่งแจ้งเตือนสำเร็จ")
        else:
          st.warning(f"Email Warning: {msg_text}")

    # TABS LAYOUT
    tab1, tab2, tab3 = st.tabs([
        "Technical Analysis & Golden Zone",
        "News Sentiment Analysis",
        "Auto Market Screener",
    ])

    # TAB 1: TECHNICAL ANALYSIS CHART
    with tab1:
      fig = make_subplots(
          rows=2,
          cols=1,
          shared_xaxes=True,
          vertical_spacing=0.03,
          row_heights=[0.75, 0.25],
      )

      # แท่งเทียน Candlestick
      fig.add_trace(
          go.Candlestick(
              x=df.index,
              open=df["Open"],
              high=df["High"],
              low=df["Low"],
              close=df["Close"],
              increasing_line_color="#00FF00",
              decreasing_line_color="#FF5252",
              name="Price",
          ),
          row=1,
          col=1,
      )

      # เส้น EMA
      fig.add_trace(
          go.Scatter(
              x=df.index,
              y=df["EMA_20"],
              line=dict(color="#00E5FF", width=1),
              name="EMA 20",
          ),
          row=1,
          col=1,
      )
      fig.add_trace(
          go.Scatter(
              x=df.index,
              y=df["EMA_50"],
              line=dict(color="#FFEA00", width=1),
              name="EMA 50",
          ),
          row=1,
          col=1,
      )

      # โซน Golden Zone (แถบสีเขียว)
      fig.add_hrect(
          y0=res["gz_min"],
          y1=res["gz_max"],
          fillcolor="#008000",
          opacity=0.25,
          line_color="#00FF00",
          line_width=1,
          line_dash="dash",
          row=1,
          col=1,
      )

      # --- วาด DEMAND & SUPPLY ZONES ---
      if show_demand_supply:
        # Supply Zone (กล่องสีแดงด้านบน)
        fig.add_hrect(
            y0=res["supply_bottom"],
            y1=res["supply_top"],
            fillcolor="#FF5252",
            opacity=0.20,
            line_color="#FF5252",
            line_width=1,
            annotation_text="Supply Zone (แรงขาย)",
            annotation_position="top left",
            annotation_font_color="#FF5252",
            row=1,
            col=1,
        )

        # Demand Zone (กล่องสีเขียวนีออนด้านล่าง)
        fig.add_hrect(
            y0=res["demand_bottom"],
            y1=res["demand_top"],
            fillcolor="#00E676",
            opacity=0.20,
            line_color="#00E676",
            line_width=1,
            annotation_text="Demand Zone (แรงซื้อ)",
            annotation_position="bottom left",
            annotation_font_color="#00E676",
            row=1,
            col=1,
        )

      # RSI
      fig.add_trace(
          go.Scatter(
              x=df.index,
              y=df["RSI"],
              line=dict(color="#00FF00", width=1.5),
              name="RSI",
          ),
          row=2,
          col=1,
      )
      fig.add_hline(y=70, line_dash="dash", line_color="#FF5252", row=2, col=1)
      fig.add_hline(y=30, line_dash="dash", line_color="#00FF00", row=2, col=1)

      fig.update_layout(
          template="plotly_dark",
          paper_bgcolor="#0B0E11",
          plot_bgcolor="#12161C",
          height=650,
          margin=dict(l=20, r=20, t=30, b=20),
          xaxis_rangeslider_visible=False,
      )
      st.plotly_chart(fig, use_container_width=True)

    # TAB 2: REAL-TIME MONTHLY NEWS SENTIMENT
    with tab2:
      st.markdown("### Real-Time Market News Sentiment (Past 30 Days)")
      news_df = scrape_news_vader(ticker)

      if news_df.empty:
        st.info("ไม่พบข่าวล่าสุดหรือเกิดข้อผิดพลาดในการดึงข้อมูลข่าวสาร")
      else:
        avg_score = news_df["Score"].mean()
        overall_str = (
            "OVERALL BULLISH"
            if avg_score >= 0.05
            else ("OVERALL BEARISH" if avg_score <= -0.05 else "NEUTRAL")
        )

        st.markdown(
            f"Overall Sentiment Score: <span"
            f" class='emerald-accent'>{avg_score:.3f}</span> →"
            f" **{overall_str}**",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(news_df, hide_index=True, use_container_width=True)

    # TAB 3: AUTO MARKET SCREENER (TOP 20 POPULAR STOCKS)
    with tab3:
      st.markdown("### Auto Market Golden Zone Screener")
      st.markdown(
          "<p style='color: #8B949E;'>สแกนหา Top 20 หุ้นยอดนิยมทั้งไทยและต่างประเทศ"
          " ที่ราคากำลังหลุดเข้าสู่ Golden Zone (50% - 61.8%) ณ ตอนนี้ทันที</p>",
          unsafe_allow_html=True,
      )

      col_s1, col_s2 = st.columns([2, 1])
      market_choice = col_s1.selectbox(
          "เลือกกลุ่มหุ้นที่จะสแกน",
          [
              "Top 20 Popular Stocks (ไทย & สหรัฐฯ)",
              "US Tech & Growth Giants (สหรัฐฯ)",
              "SET50 Big Caps (ไทย)",
          ],
      )

      top_20_popular = [
          "NVDA",
          "AAPL",
          "MSFT",
          "AMZN",
          "GOOGL",
          "META",
          "TSLA",
          "AMD",
          "PLTR",
          "WDC",
          "MU",
          "NFLX",
          "DELTA.BK",
          "ADVANC.BK",
          "GULF.BK",
          "PTT.BK",
          "BDMS.BK",
          "KBANK.BK",
          "CPALL.BK",
          "AOT.BK",
      ]

      us_tech_list = [
          "NVDA",
          "AAPL",
          "MSFT",
          "AMZN",
          "GOOGL",
          "META",
          "TSLA",
          "AMD",
          "PLTR",
          "WDC",
          "MU",
          "NFLX",
          "AVGO",
          "INTC",
          "ORCL",
      ]
      set50_list = [
          "DELTA.BK",
          "ADVANC.BK",
          "GULF.BK",
          "PTT.BK",
          "BDMS.BK",
          "KBANK.BK",
          "CPALL.BK",
          "AOT.BK",
          "BBL.BK",
          "SCB.BK",
          "KTB.BK",
          "CPN.BK",
          "PTTEP.BK",
          "SCC.BK",
          "TOP.BK",
      ]

      if market_choice == "Top 20 Popular Stocks (ไทย & สหรัฐฯ)":
        target_list = top_20_popular
      elif market_choice == "US Tech & Growth Giants (สหรัฐฯ)":
        target_list = us_tech_list
      else:
        target_list = set50_list

      if col_s2.button("START MARKET SCAN"):
        with st.spinner("กำลังสแกนหาหุ้นที่เข้า Golden Zone..."):
          screener_df = scan_golden_zone_stocks(
              target_list, period=period, interval=interval
          )
          if not screener_df.empty:
            st.success(
                f"สแกนเสร็จสิ้น! พบหุ้นที่อยู่ใน Golden Zone จำนวน"
                f" {len(screener_df)} ตัว"
            )
            st.dataframe(screener_df, hide_index=True, use_container_width=True)
          else:
            st.info(
                "ขณะนี้ยังไม่มีหุ้นในรายการ Top 20 หลุดเข้ามาใน Golden Zone"
                " (ราคายังไม่ย่อตัวถึงช่วง 50% - 61.8%)"
            )
