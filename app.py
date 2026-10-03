import streamlit as st
import plotly.graph_objects as go
from modules.technical_analysis import detect_swings_and_golden_zone

st.set_page_config(page_title="Stock Golden Zone Analyzer", layout="wide")

st.title("📈 Stock Golden Zone & Technical Analyzer")

ticker = st.sidebar.text_input("Ticker Symbol", value="NVDA").upper()
period = st.sidebar.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1)

if st.button("Analyze"):
    with st.spinner("Fetching data..."):
        df, peaks_high, peaks_low, fib_levels, current_price, in_gz, is_uptrend = detect_swings_and_golden_zone(ticker, period=period)
        
        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:.2f}")
        col2.metric("Trend Status", "Uptrend (Retracement)" if is_uptrend else "Downtrend (Bounce)")
        col3.metric("In Golden Zone?", "YES 🎯" if in_gz else "NO")
        
        # Plot Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        
        fig.add_hrect(
            y0=min(fib_levels['Golden Zone Bottom'], fib_levels['Golden Zone Top']),
            y1=max(fib_levels['Golden Zone Bottom'], fib_levels['Golden Zone Top']),
            fillcolor="gold", opacity=0.3, line_width=0,
            annotation_text="Golden Zone (50% - 61.8%)"
        )
        
        fig.update_layout(title=f"{ticker} Price Chart with Golden Zone", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
