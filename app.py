import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from bs4 import BeautifulSoup
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from scipy.signal import find_peaks
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import yfinance as yf

# --- 1. INITIALIZATION & CACHING ---


@st.cache_resource
def init_nltk():
  nltk.download("vader_lexicon", quiet=True)


@st.cache_resource
def load_finbert_pipeline():
  """โหลดโมเดล FinBERT จาก Hugging Face (แคชไว้โหลดครั้งเดียว)"""
  model_name = "ProsusAI/finbert"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name)
  return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


init_nltk()

# --- 2. SET PAGE CONFIG & CUSTOM CSS ---
st.set_page_config(
    page_title="Emerald Ultimate Financial Analytics",
    page_icon="📈",
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

# --- 3. HELPER FUNCTIONS ---


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

  subject = f"🚨 EMERALD BUY ALERT: {ticker} (${current_price:.2f})"
  signals_html = "".join(
      [f"<li style='padding: 4px 0; color: #00FF00;'><b>{s}</b></li>" for s in signals]
  )

  body = f"""
    <html>
    <body style="background-color: #0B0E11; color: #EAEAEA; font-family: Arial, sans-serif; padding: 20px;">
        <div style="max-width: 600px; margin: auto; background-color: #12161C; border: 1px solid #008000; border-radius: 10px; padding: 25px;">
            <h2 style="color: #00FF00; margin-top: 0;">🚨 EMERALD BUY SIGNAL ALERT</h2>
            <p style="font-size: 16px; color: #FFFFFF;">พบสัญญาณเข้าซื้อสำหรับหุ้น <b>{ticker}</b></p>
            <hr style="border: 0.5px solid #1E232A;">
            <table style="width: 100%; margin: 15px 0; font-size: 14px;">
                <tr><td style="color: #8B949E;">Ticker Symbol:</td><td style="color: #FFFFFF; font-weight: bold;">{ticker}</td></tr>
                <tr><td style="color: #8B949E;">Last Price:</td><td style="color: #00FF00; font-weight: bold;">${current_price:.2f}</td></tr>
                <tr><td style="color: #8B949E;">Timeframe:</td><td style="color: #FFFFFF;">{timeframe}</td></tr>
                <tr><td style="color: #8B949E;">Golden Zone:</td><td style="color: #00FF00;">${gz_min:.2f} - ${gz_max:.2f}</td></tr>
            </table>
            <h4 style="color: #FFFFFF; margin-bottom: 5px;">🔍 Confluence Signals Detected:</h4>
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
    return True, "ส่งอีเมลแจ้งเตือนสำเร็จ!"
  except Exception as e:
    return False, f"เกิดข้อผิดพลาดในการส่งอีเมล: {str(e)}"


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


@st.cache_data(ttl=600)
def fetch_and_analyze(
    ticker_symbol: str,
    period="6mo",
    interval="1d",
    distance=10,
    prominence=2,
):
  df = yf.download(ticker_symbol, period=period, interval=interval)
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  if df.empty:
    return None, {}

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
    ema_signal = "GOLDEN CROSS 🚀 (EMA20 ตัดขึ้น EMA50)"
  elif ema_20_prev > ema_50_prev and ema_20_curr <= ema_50_curr:
    ema_signal = "DEATH CROSS 💀 (EMA20 ตัดลง EMA50)"
  elif ema_20_curr > ema_50_curr:
    ema_signal = "BULLISH ALIGNMENT 🟢 (EMA20 > EMA50)"
  else:
    ema_signal = "BEARISH ALIGNMENT 🔴 (EMA20 < EMA50)"

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

  return df, analysis_results


@st.cache_data(ttl=1800)
def scrape_news_finbert(ticker_symbol: str):
  """Scrape ข่าวและวิเคราะห์ด้วย FinBERT NLP"""
  url = f"https://finviz.com/quote.ashx?t={ticker_symbol}"
  headers = {"User-Agent": "Mozilla/5.0"}

  try:
    response = requests.get(url, headers=headers, timeout=5)
    soup = BeautifulSoup(response.text, "html.parser")
    news_table = soup.find(id="news-table")

    if not news_table:
      return pd.DataFrame()

    headlines, times = [], []
    for row in news_table.find_all("tr")[:10]:
      if row.a:
        headlines.append(row.a.text)
        times.append(row.td.text.strip())

    if not headlines:
      return pd.DataFrame()

    finbert = load_finbert_pipeline()
    results = finbert(headlines)

    parsed_news = []
    for time_str, title, res in zip(times, headlines, results):
      label = res["label"].upper()
      score = res["score"]

      if label == "POSITIVE":
        sentiment = "BULLISH 🟢"
        numeric_score = score
      elif label == "NEGATIVE":
        sentiment = "BEARISH 🔴"
        numeric_score = -score
      else:
        sentiment = "NEUTRAL ⚪"
        numeric_score = 0.0

      parsed_news.append({
          "Time": time_str,
          "Headline": title,
          "FinBERT Label": label,
          "Confidence": round(score, 4),
          "Sentiment": sentiment,
          "Score": numeric_score,
      })

    return pd.DataFrame(parsed_news)
  except Exception:
    return pd.DataFrame()


def backtest_golden_zone_strategy(
    df, take_profit_rsi=70, stop_loss_pct=0.03, initial_capital=10000
):
  trades = []
  in_position = False
  entry_price = 0
  entry_date = None
  capital = initial_capital
  portfolio_values = []

  highs = df["High"].values
  lows = df["Low"].values

  for i in range(30, len(df)):
    current_date = df.index[i]
    close_price = df["Close"].iloc[i]
    rsi_val = df["RSI"].iloc[i]

    window_high = max(highs[i - 30 : i])
    window_low = min(lows[i - 30 : i])
    diff = window_high - window_low
    gz_top = window_high - (0.500 * diff)
    gz_bottom = window_high - (0.618 * diff)
    in_gz = min(gz_bottom, gz_top) <= close_price <= max(gz_bottom, gz_top)

    prev_min_price_idx = np.argmin(lows[i - 15 : i]) + (i - 15)
    prev_min_rsi_idx = np.argmin(df["RSI"].iloc[i - 15 : i].values) + (i - 15)
    bullish_div = (
        lows[i] < lows[prev_min_price_idx]
        and df["RSI"].iloc[i] > df["RSI"].iloc[prev_min_rsi_idx]
        and df["RSI"].iloc[i] < 45
    )

    if not in_position and in_gz and bullish_div:
      in_position = True
      entry_price = close_price
      entry_date = current_date
      position_size = capital / entry_price

    elif in_position:
      hit_tp = rsi_val >= take_profit_rsi
      hit_sl = close_price <= entry_price * (1 - stop_loss_pct)

      if hit_tp or hit_sl or i == len(df) - 1:
        exit_price = close_price
        exit_date = current_date
        pnl = (exit_price - entry_price) * position_size
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100

        capital += pnl
        in_position = False

        trades.append({
            "Entry Date": entry_date,
            "Exit Date": exit_date,
            "Entry Price": entry_price,
            "Exit Price": exit_price,
            "PnL ($)": pnl,
            "Return (%)": pnl_pct,
            "Reason": (
                "Take Profit (RSI)"
                if hit_tp
                else ("Stop Loss" if hit_sl else "End of Data")
            ),
            "Capital After": capital,
        })

    portfolio_values.append(
        capital if not in_position else position_size * close_price
    )

  trades_df = pd.DataFrame(trades)

  if not trades_df.empty:
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df["PnL ($)"] > 0])
    losing_trades = len(trades_df[trades_df["PnL ($)"] <= 0])
    win_rate = (winning_trades / total_trades) * 100
    total_return_pct = ((capital - initial_capital) / initial_capital) * 100

    gross_profit = trades_df[trades_df["PnL ($)"] > 0]["PnL ($)"].sum()
    gross_loss = abs(trades_df[trades_df["PnL ($)"] < 0]["PnL ($)"].sum())
    profit_factor = (
        (gross_profit / gross_loss)
        if gross_loss > 0
        else (gross_profit if gross_profit > 0 else 1.0)
    )

    equity_series = pd.Series(portfolio_values)
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_drawdown_pct = abs(drawdown.min()) * 100
  else:
    total_trades, winning_trades, losing_trades = 0, 0, 0
    win_rate, total_return_pct, profit_factor, max_drawdown_pct = (
        0.0,
        0.0,
        0.0,
        0.0,
    )

  metrics = {
      "Initial Capital": initial_capital,
      "Final Capital": capital,
      "Total Return (%)": total_return_pct,
      "Win Rate (%)": win_rate,
      "Total Trades": total_trades,
      "Winning Trades": winning_trades,
      "Losing Trades": losing_trades,
      "Profit Factor": profit_factor,
      "Max Drawdown (%)": max_drawdown_pct,
  }

  return trades_df, metrics, portfolio_values


def scan_golden_zone_stocks(ticker_list, period="6mo", interval="1d"):
  results = []
  progress_bar = st.progress(0)
  status_text = st.empty()
  total = len(ticker_list)

  for idx, symbol in enumerate(ticker_list):
    status_text.markdown(
        f"🔍 *Scanning:* `<span class='emerald-accent'>{symbol}</span>`"
        f" ({idx+1}/{total})",
        unsafe_allow_html=True,
    )
    progress_bar.progress((idx + 1) / total)

    try:
      df = yf.download(
          symbol, period=period, interval=interval, progress=False
      )
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

      if df.empty or len(df) < 30:
        continue

      highs = df["High"].values
      lows = df["Low"].values
      close_prices = df["Close"].values

      peaks_high, _ = find_peaks(highs, distance=10, prominence=2)
      peaks_low, _ = find_peaks(-lows, distance=10, prominence=2)

      last_h_idx = peaks_high[-1] if len(peaks_high) > 0 else np.argmax(highs)
      last_l_idx = peaks_low[-1] if len(peaks_low) > 0 else np.argmin(lows)

      swing_high = highs[last_h_idx]
      swing_low = lows[last_l_idx]
      current_price = close_prices[-1]

      diff = swing_high - swing_low
      is_uptrend = last_h_idx > last_l_idx

      if is_uptrend:
        gz_top = swing_high - (0.500 * diff)
        gz_bottom = swing_high - (0.618 * diff)
      else:
        gz_bottom = swing_low + (0.500 * diff)
        gz_top = swing_low + (0.618 * diff)

      gz_min = min(gz_top, gz_bottom)
      gz_max = max(gz_top, gz_bottom)
      in_gz = gz_min <= current_price <= gz_max

      delta = df["Close"].diff()
      gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
      rs = gain / loss
      rsi = (100 - (100 / (1 + rs))).iloc[-1]

      if in_gz:
        results.append({
            "Symbol": symbol,
            "Price ($/฿)": round(current_price, 2),
            "Golden Zone Min": round(gz_min, 2),
            "Golden Zone Max": round(gz_max, 2),
            "Trend": (
                "Uptrend (Retracement)" if is_uptrend else "Downtrend (Bounce)"
            ),
            "RSI (14)": round(rsi, 1),
            "RSI Status": (
                "Oversold 🎯"
                if rsi <= 35
                else ("Overbought ⚠️" if rsi >= 65 else "Neutral")
            ),
        })
    except Exception:
      continue

  progress_bar.empty()
  status_text.empty()
  return pd.DataFrame(results)


# --- 4. SIDEBAR CONTROL ---
st.sidebar.markdown(
    "<h2 style='color: #00FF00;'>⚡ CONTROL PANEL</h2>", unsafe_allow_html=True
)
ticker = st.sidebar.text_input("SYMBOL", value="NVDA").upper()
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
    "<h4 style='color: #8B949E;'>📧 Gmail Alert Settings</h4>",
    unsafe_allow_html=True,
)
sender_email = st.sidebar.text_input(
    "Sender Gmail", placeholder="your_email@gmail.com"
)
app_password = st.sidebar.text_input(
    "App Password (16-digits)", type="password"
)
receiver_email = st.sidebar.text_input(
    "Receiver Email", placeholder="receive_email@gmail.com"
)
enable_email_alert = st.sidebar.checkbox("Enable Gmail Alert", value=True)

st.sidebar.markdown("---")
analyze_btn = st.sidebar.button("RUN ANALYSIS")


# --- 5. MAIN CONTENT ---
st.markdown(
    "<h1>📈 EMERALD <span class='emerald-accent'>ULTIMATE ANALYTICS</span></h1>",
    unsafe_allow_html=True,
)

if ticker:
  with st.spinner(f"Analyzing {ticker}..."):
    df, res = fetch_and_analyze(
        ticker, period=period, interval=interval
    )

    if df is None or df.empty:
      st.error(f"ไม่พบข้อมูลสำหรับหุ้น {ticker}")
    else:
      # METRICS BAR
      c1, c2, c3, c4 = st.columns(4)
      c1.metric("PRICE", f"${res['current_price']:.2f}")
      c2.metric("RSI (14)", f"{res['current_rsi']:.1f}")
      c3.metric(
          "GOLDEN ZONE", f"${res['gz_min']:.2f} -${res['gz_max']:.2f}"
      )
      c4.metric("GZ STATUS", "IN ZONE 🎯" if res["in_gz"] else "OUTSIDE ZONE")

      st.markdown("<br>", unsafe_allow_html=True)

      # GMAIL ALERT TRIGGER
      bullish_signals = []
      if res["in_gz"]:
        bullish_signals.append(
            f"Price is in Golden Zone (${res['gz_min']:.2f} -"
            f" ${res['gz_max']:.2f})"
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
            st.success(
                f"📧 **Gmail Alert Sent!** สัญญาณถูกส่งไปยัง {receiver_email}"
                " เรียบร้อยแล้ว"
            )
          else:
            st.warning(f"⚠️️ **Email Warning:** {msg_text}")

      # TABS LAYOUT
      tab1, tab2, tab3, tab4 = st.tabs([
          "📊 Technical & Golden Zone",
          "📰 News FinBERT Sentiment",
          "🧪 Strategy Backtest Engine",
          "🔎 Auto Market Screener",
      ])

      # TAB 1: TECHNICAL ANALYSIS
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
        fig.add_hline(
            y=70, line_dash="dash", line_color="#FF5252", row=2, col=1
        )
        fig.add_hline(
            y=30, line_dash="dash", line_color="#00FF00", row=2, col=1
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B0E11",
            plot_bgcolor="#12161C",
            height=600,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)

      # TAB 2: NEWS FINBERT SENTIMENT
      with tab2:
        st.markdown("### 📰 Latest Market News & FinBERT NLP Sentiment")
        news_df = scrape_news_finbert(ticker)

        if news_df.empty:
          st.info("ไม่พบข่าวล่าสุดหรือเกิดข้อผิดพลาดในการดึงข้อมูลข่าวสาร")
        else:
          avg_score = news_df["Score"].mean()
          overall_str = (
              "OVERALL BULLISH 🟢"
              if avg_score >= 0.15
              else ("OVERALL BEARISH 🔴" if avg_score <= -0.15 else "NEUTRAL ⚪")
          )

          st.markdown(
              f"**Overall FinBERT Score:** `<span"
              f" class='emerald-accent'>{avg_score:.3f}</span>` →"
              f" **{overall_str}**",
              unsafe_allow_html=True,
          )
          st.markdown("<br>", unsafe_allow_html=True)

          st.dataframe(
              news_df[[
                  "Time",
                  "Headline",
                  "FinBERT Label",
                  "Confidence",
                  "Sentiment",
              ]],
              column_config={
                  "Time": st.column_config.TextColumn(
                      "Time/Date", width="small"
                  ),
                  "Headline": st.column_config.TextColumn(
                      "News Headline", width="large"
                  ),
                  "Confidence": st.column_config.NumberColumn(
                      "Confidence", format="%.2f"
                  ),
              },
              hide_index=True,
              use_container_width=True,
          )

      # TAB 3: BACKTEST ENGINE
      with tab3:
        st.markdown(
            "### 🧪 Strategy Backtesting: Golden Zone + RSI Divergence"
        )
        bc1, bc2, bc3 = st.columns(3)
        init_cap = bc1.number_input(
            "เงินทุนเริ่มต้น ($)", value=10000, step=1000
        )
        tp_rsi = bc2.slider("Take Profit RSI Level", 60, 80, 70)
        sl_pct = bc3.slider("Stop Loss (%)", 1.0, 10.0, 3.0) / 100.0

        if st.button("RUN BACKTEST"):
          with st.spinner("Calculating backtest..."):
            trades_df, metrics, portfolio_values = (
                backtest_golden_zone_strategy(
                    df,
                    take_profit_rsi=tp_rsi,
                    stop_loss_pct=sl_pct,
                    initial_capital=init_cap,
                )
            )

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("TOTAL RETURN", f"{metrics['Total Return (%)']:.2f}%")
            m2.metric("WIN RATE", f"{metrics['Win Rate (%)']:.1f}%")
            m3.metric(
                "TRADES",
                f"{metrics['Total Trades']} (W:{metrics['Winning Trades']} /"
                f" L:{metrics['Losing Trades']})",
            )
            m4.metric("PROFIT FACTOR", f"{metrics['Profit Factor']:.2f}")
            m5.metric("MAX DRAWDOWN", f"-{metrics['Max Drawdown (%)']:.2f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            fig_eq = go.Figure()
            fig_eq.add_trace(
                go.Scatter(
                    x=df.index[30:],
                    y=portfolio_values,
                    mode="lines",
                    line=dict(color="#00FF00", width=2),
                    name="Portfolio ($)",
                )
            )
            fig_eq.update_layout(
                title="Portfolio Equity Curve ($)",
                template="plotly_dark",
                paper_bgcolor="#0B0E11",
                plot_bgcolor="#12161C",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            if not trades_df.empty:
              st.dataframe(
                  trades_df,
                  column_config={
                      "Entry Price": st.column_config.NumberColumn(
                          format="$%.2f"
                      ),
                      "Exit Price": st.column_config.NumberColumn(
                          format="$%.2f"
                      ),
                      "PnL ($)": st.column_config.NumberColumn(format="$%.2f"),
                      "Return (%)": st.column_config.NumberColumn(
                          format="%.2f%%"
                      ),
                  },
                  hide_index=True,
                  use_container_width=True,
              )

      # TAB 4: AUTO MARKET SCREENER
      with tab4:
        st.markdown("### 🔎 Auto Market Golden Zone Screener")
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
            "MSFT", "AMZN",
            "GOOGL",
            "META",
            "TSLA",
            "AVGO",
            "AMD",
            "NFLX",
            "PLTR",
            "COST",
        ]

        if col_s2.button("🚀 START MARKET SCAN"):
          target_list = set50_list if "SET50" in market_choice else sp500_list
          with st.spinner("สแกนราคาหุ้นในตลาด..."):
            screener_df = scan_golden_zone_stocks(
                target_list, period=period, interval=interval
            )
            if not screener_df.empty:
              st.success(f"🎯 พบหุ้นใน Golden Zone จำนวน {len(screener_df)} ตัว")
              st.dataframe(screener_df, hide_index=True, use_container_width=True)
            else:
              st.warning("⚠️ ไม่พบหุ้นที่อยู่ใน Golden Zone ณ ขณะนี้")
