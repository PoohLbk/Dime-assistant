import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# ดาวน์โหลด Lexicon สำหรับ Sentiment Analysis (โหลดครั้งเดียว)
@st.cache_resource
def init_nltk():
    nltk.download('vader_lexicon', quiet=True)

init_nltk()

# --- 1. SET PAGE CONFIG ---
st.set_page_config(
    page_title="Emerald Stock & Sentiment Analyzer",
    page_icon="📈",
    layout="wide"
)

# --- 2. CUSTOM CSS (Dark & Emerald Luxury Theme) ---
custom_css = """
<style>
    /* พื้นหลังหลักของแอป */
    .stApp {
        background-color: #0B0E11;
        color: #EAEAEA;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #12161C;
        border-right: 1px solid #1E232A;
    }

    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #008000;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0px 4px 12px rgba(0, 128, 0, 0.15);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #8B949E !important;
        font-size: 0.85rem !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Button Style (Emerald Green) */
    div.stButton > button {
        background-color: #008000;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    div.stButton > button:hover {
        background-color: #00B300;
        box-shadow: 0px 0px 10px rgba(0, 255, 0, 0.4);
        color: #FFFFFF;
    }

    /* Tab Styling */
    button[data-baseweb="tab"] {
        color: #8B949E !important;
        font-weight: 600;
    }
    button[aria-selected="true"] {
        color: #00FF00 !important;
        border-bottom-color: #00FF00 !important;
    }

    /* Custom Accent Text */
    .emerald-accent {
        color: #00FF00;
        font-weight: bold;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---

@st.cache_data(ttl=3600)
def fetch_stock_data_and_analyze(ticker_symbol: str, period="6mo", distance=10, prominence=2):
    """ ดึงราคาหุ้น และคำนวณ Swing High/Low + Golden Zone ด้วย find_peaks """
    df = yf.download(ticker_symbol, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if df.empty:
        return None, None, None, None, None, None, None

    highs = df['High'].values
    lows = df['Low'].values
    
    # ตรวจหาจุด Swing High/Low
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


@st.cache_data(ttl=1800)
def scrape_news_sentiment(ticker_symbol: str):
    """ Scrape ข่าวหุ้นจาก Finviz และทำ Sentiment Analysis """
    url = f"https://finviz.com/quote.ashx?t={ticker_symbol}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        news_table = soup.find(id='news-table')
        
        if not news_table:
            return pd.DataFrame()

        parsed_news = []
        sia = SentimentIntensityAnalyzer()
        rows = news_table.find_all('tr')
        
        for row in rows[:10]:
            if row.a:
                title = row.a.text
                time_str = row.td.text.strip()
                score = sia.polarity_scores(title)['compound']
                
                if score >= 0.05:
                    sentiment = "BULLISH 🟢"
                elif score <= -0.05:
                    sentiment = "BEARISH 🔴"
                else:
                    sentiment = "NEUTRAL ⚪"
                    
                parsed_news.append({
                    'Time': time_str,
                    'Headline': title,
                    'Sentiment Score': score,
                    'Sentiment': sentiment
                })
                
        return pd.DataFrame(parsed_news)
    except Exception as e:
        return pd.DataFrame()


# --- 4. SIDEBAR CONTROL ---
st.sidebar.markdown("<h2 style='color: #00FF00;'>⚡ CONTROL PANEL</h2>", unsafe_allow_html=True)
ticker = st.sidebar.text_input("SYMBOL (e.g. NVDA, AAPL, TSLA)", value="NVDA").upper()
period = st.sidebar.selectbox("TIMEFRAME", ["3mo", "6mo", "1y", "2y"], index=1)
st.sidebar.markdown("---")
st.sidebar.markdown("<h4 style='color: #8B949E;'>Peak Detection Settings</h4>", unsafe_allow_html=True)
peak_distance = st.sidebar.slider("Min Peak Distance (Bars)", min_value=5, max_value=30, value=10)
peak_prominence = st.sidebar.slider("Peak Prominence", min_value=1, max_value=10, value=2)

analyze_btn = st.sidebar.button("RUN ANALYSIS")

# --- 5. MAIN HEADER ---
st.markdown("<h1>📈 EMERALD <span class='emerald-accent'>STOCK ANALYZER</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #8B949E;'>Automated Fibonacci Golden Zone, Support/Resistance & News Sentiment</p>", unsafe_allow_html=True)

if ticker:
    with st.spinner(f"Analyzing {ticker}..."):
        df, peaks_high, peaks_low, fib_levels, current_price, in_gz, is_uptrend = fetch_stock_data_and_analyze(
            ticker, period=period, distance=peak_distance, prominence=peak_prominence
        )
        
        if df is None or df.empty:
            st.error(f"ไม่พบข้อมูลสำหรับหุ้นสัญลักษณ์: {ticker}")
        else:
            # --- TOP METRIC CARDS ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("CURRENT PRICE", f"${current_price:.2f}")
            col2.metric("MARKET TREND", "UPTREND (Retracement)" if is_uptrend else "DOWNTREND (Bounce)")
            col3.metric("GOLDEN ZONE", f"${min(fib_levels['Golden Zone Bottom'], fib_levels['Golden Zone Top']):.2f} -${max(fib_levels['Golden Zone Bottom'], fib_levels['Golden Zone Top']):.2f}")
            col4.metric("STATUS", "IN ZONE 🎯" if in_gz else "OUTSIDE ZONE")

            st.markdown("<br>", unsafe_allow_html=True)

            # --- TABS LAYOUT ---
            tab1, tab2 = st.tabs(["📊 Technical & Golden Zone", "📰 News & Sentiment Analysis"])

            # TAB 1: TECHNICAL ANALYSIS
            with tab1:
                fig = go.Figure()

                # Candlestick Chart
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                    name="Price",
                    increasing_line_color='#00FF00', increasing_fillcolor='#008000',
                    decreasing_line_color='#FF5252', decreasing_fillcolor='#8B0000'
                ))

                # Swing High Markers
                fig.add_trace(go.Scatter(
                    x=df.index[peaks_high], y=df['High'].iloc[peaks_high],
                    mode='markers', marker=dict(color='#FF5252', size=9, symbol='triangle-down'), name='Swing High'
                ))

                # Swing Low Markers
                fig.add_trace(go.Scatter(
                    x=df.index[peaks_low], y=df['Low'].iloc[peaks_low],
                    mode='markers', marker=dict(color='#00FF00', size=9, symbol='triangle-up'), name='Swing Low'
                ))

                # Golden Zone Highlight Fill
                gz_min = min(fib_levels['Golden Zone Bottom'], fib_levels['Golden Zone Top'])
                gz_max = max(fib_levels['Golden Zone Bottom'], fib_levels['Golden Zone Top'])
                
                fig.add_hrect(
                    y0=gz_min, y1=gz_max,
                    fillcolor="#008000", opacity=0.25,
                    line_color="#00FF00", line_width=1, line_dash="dash",
                    annotation_text="GOLDEN ZONE (50.0% - 61.8%)",
                    annotation_position="top left",
                    annotation_font_color="#00FF00"
                )

                # Chart Layout Styling
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='#0B0E11',
                    plot_bgcolor='#12161C',
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=dict(showgrid=True, gridcolor='#1E232A', rangeslider=dict(visible=False)),
                    yaxis=dict(showgrid=True, gridcolor='#1E232A'),
                    font=dict(family="Arial", size=12, color="#EAEAEA"),
                    height=580
                )

                st.plotly_chart(fig, use_container_width=True)

            # TAB 2: NEWS SENTIMENT
            with tab2:
                st.markdown("### 📰 Latest Market News & NLP Sentiment")
                news_df = scrape_news_sentiment(ticker)
                
                if news_df.empty:
                    st.info("ไม่พบข่าวล่าสุดหรือเกิดข้อผิดพลาดในการดึงข้อมูลข่าว")
                else:
                    avg_sentiment = news_df['Sentiment Score'].mean()
                    
                    # Sentiment Indicator
                    if avg_sentiment >= 0.05:
                        overall_str = "OVERALL BULLISH 🟢"
                    elif avg_sentiment <= -0.05:
                        overall_str = "OVERALL BEARISH 🔴"
                    else:
                        overall_str = "NEUTRAL ⚪"

                    st.markdown(f"**Overall News Sentiment Score:** `<span class='emerald-accent'>{avg_sentiment:.3f}</span>` → **{overall_str}**", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # แสดงตารางข่าว
                    st.dataframe(
                        news_df,
                        column_config={
                            "Time": st.column_config.TextColumn("Time/Date", width="small"),
                            "Headline": st.column_config.TextColumn("News Headline", width="large"),
                            "Sentiment Score": st.column_config.NumberColumn("Score", format="%.3f"),
                            "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
