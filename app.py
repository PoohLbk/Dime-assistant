import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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


@st.cache_data(ttl=120)
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

  yf_obj = yf.Ticker(clean_symbol)
  df = yf_obj.history(period=period, interval=interval)

  if df.empty:
    return None, {}, "N/A"

  # ดึงข้อมูล Dividend Yield
  try:
    info = yf_obj.info
    div_yield_val = info.get("dividendYield", None)
    if div_yield_val is not None and div_yield_val > 0:
      dividend_yield_str = f"{div_yield_val * 100:.2f}%"
    else:
      dividend_yield_str = "N/A (0.00%)"
  except Exception:
    dividend_yield_str = "N/A"

  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

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

  divergence_signals = detect_divergence(df, peaks_high, peaks_low)

  analysis_results = {
      "current_price": current_price,
      "current_rsi": current_rsi,
      "is_uptrend": is_uptrend,
      "in_gz": in_gz,
      "gz_min": min(gz_top, gz_bottom),
      "gz_max": max(gz_top, gz_bottom),
      "ema_signal": ema_signal,
      "divergences": divergence_signals,
      "peaks_high": peaks_high,
      "peaks_low": peaks_low,
  }

  return df, analysis_results, dividend_yield_str


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
      yf_obj = yf.Ticker(symbol)
      df = yf_obj.history(period=period, interval=interval)
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


@st.cache_data(ttl=60)  # ดึงข่าวสดใหม่ทุก 1 นาที
def fetch_realtime_news(ticker_symbol: str):
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

  try:
    yf_obj = yf.Ticker(clean_symbol)
    news_items = yf_obj.news

    if not news_items:
      return pd.DataFrame()

    sia = SentimentIntensityAnalyzer()
    parsed_news = []

    for item in news_items[:10]:
      title = item.get("title", "")
      publisher = item.get("publisher", "")

      # แปลง Timestamp เป็นเวลา
      pub_time = item.get("providerPublishTime", None)
      if pub_time:
        time_str = pd.to_datetime(pub_time, unit="s").strftime(
            "%Y-%m-%d %H:%M"
        )
      else:
        time_str = "Recently"

      score = sia.polarity_scores(title)["compound"]
      sentiment = (
          "BULLISH"
          if score >= 0.05
          else ("BEARISH" if score <= -0.05 else "NEUTRAL")
      )

      parsed_news.append({
          "Time (UTC)": time_str,
          "Source": publisher,
          "Headline": title,
          "Sentiment": sentiment,
          "Score": round(score, 4),
      })

    return pd.DataFrame(parsed_news)
  except Exception:
    return pd.DataFrame()


# --- 5. SIDEBAR CONTROL ---
st.sidebar.markdown(
    "<h2 style='color: #00FF00;'>CONTROL PANEL</h2>", unsafe_allow_html=True
)

ticker = st.sidebar.text_input(
    "SYMBOL (e.g. PLTR, NVDA, GC=F)", value="PLTR"
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
  df, res, div_yield = fetch_and_analyze(ticker, period=period, interval=interval)

  if df is None or df.empty:
    st.error(f"ไม่พบข้อมูลสำหรับสัญลักษณ์ {ticker}")
  else:
    # METRICS BAR (เพิ่ม DIVIDEND YIELD)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PRICE", f"\${res['current_price']:.2f}")
    c2.metric("RSI (14)", f"{res['current_rsi']:.1f}")
    c3.metric("DIV YIELD", div_yield)
    c4.metric("GOLDEN ZONE", f"${res['gz_min']:.2f} -${res['gz_max']:.2f}")
    c5.metric("GZ STATUS", "IN ZONE" if res["in_gz"] else "OUTSIDE ZONE")

    st.markdown("<br>", unsafe_allow_html=True)

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
        "Real-Time News & Sentiment",
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
          height=600,
          margin=dict(l=20, r=20, t=30, b=20),
          xaxis_rangeslider_visible=False,
      )
      st.plotly_chart(fig, use_container_width=True)

    # TAB 2: REAL-TIME NEWS & SENTIMENT
    with tab2:
      st.markdown("### Real-Time Live News & Sentiment Analysis")
      news_df = fetch_realtime_news(ticker)

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
            f"**Overall Sentiment Score:** <span"
            f" class='emerald-accent'>{avg_score:.3f}</span> →"
            f" **{overall_str}**",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(news_df, hide_index=True, use_container_width=True)

    # TAB 3: AUTO MARKET SCREENER
    with tab3:
      st.markdown("### Auto Market Golden Zone Screener")
      st.markdown(
          "<p style='color: #8B949E;'>สแกนหาหุ้นในตลาดที่ราคากำลังหลุดเข้าสู่"
          " Golden Zone (50% - 61.8%) ณ ตอนนี้ทันที</p>",
          unsafe_allow_html=True,
      )

      col_s1, col_s2 = st.columns([2, 1])
      market_choice = col_s1.selectbox(
          "เลือกกลุ่มหุ้นที่จะสแกน",
          ["SET50 (หุ้นไทย)", "S&P 500 Top Tech (หุ้นอเมริกา)"],
      )

      set50_list = [
          "ADVANC.BK",
          "AOT.BK",
          "BDMS.BK",
          "BBL.BK",
          "CPALL.BK",
          "CPN.BK",
          "DELTA.BK",
          "GULF.BK",
          "KBANK.BK",
          "KTB.BK",
          "PTT.BK",
          "PTTEP.BK",
          "SCB.BK",
          "SCC.BK",
          "TOP.BK",
      ]
      sp500_list = [
          "NVDA",
          "AAPL",
          "MSFT",
          "AMZN",
          "GOOGL",
          "META",
          "TSLA",
          "AVGO",
          "AMD",
          "NFLX",
      ]

      if col_s2.button("START MARKET SCAN"):
        target_list = set50_list if "SET50" in market_choice else sp500_list
        with st.spinner("กำลังสแกนราคาหุ้นในตลาด..."):
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
            st.warning("ไม่พบหุ้นที่อยู่ใน Golden Zone ณ ขณะนี้")
