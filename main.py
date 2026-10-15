import os
from datetime import datetime
import xml.etree.ElementTree as ET

from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from scipy.signal import find_peaks
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# โหลด NLTK Lexicon
nltk.download("vader_lexicon", quiet=True)

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# --- SCRAPE NEWS & SORT BY DATE (DESCENDING) ---
def scrape_news_vader(ticker_symbol):
    ticker_map = {
        "GC=F": "Gold market",
        "XAUUSD": "Gold market",
        "SI=F": "Silver market",
        "BTC-USD": "Bitcoin",
    }
    search_term = ticker_map.get(ticker_symbol.upper().strip(), ticker_symbol)
    parsed_news = []

    try:
        rss_url = f"https://news.google.com/rss/search?q={search_term}+stock+when:30d&hl=en-US&gl=US&ceid=US:en"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        res = requests.get(rss_url, headers=headers, timeout=5)

        if res.status_code == 200:
            root = ET.fromstring(res.content)
            for item in root.findall(".//item"):
                title = item.find("title").text
                clean_title = (
                    title.rsplit(" - ", 1)[0] if " - " in title else title
                )
                raw_date = item.find("pubDate").text

                # แปลงวันที่สำหรับใช้ Sort และแสดงผล
                dt_obj = datetime.min
                try:
                    dt_obj = datetime.strptime(raw_date[5:25], "%d %b %Y %H:%M:%S")
                    formatted_time = dt_obj.strftime("%b-%d %I:%M%p")
                except Exception:
                    formatted_time = raw_date[5:22]

                score = float(sia.polarity_scores(clean_title)["compound"])
                sentiment = (
                    "BULLISH"
                    if score >= 0.05
                    else ("BEARISH" if score <= -0.05 else "NEUTRAL")
                )

                parsed_news.append({
                    "dt_obj": dt_obj,  # เก็บไว้ใช้เรียงลำดับวันที่
                    "time": formatted_time,
                    "headline": clean_title,
                    "sentiment": sentiment,
                    "score": round(score, 2),
                })

            # 📌 จัดเรียงข่าวจากวันที่ล่าสุดลงไป (Most Recent First)
            parsed_news = sorted(parsed_news, key=lambda x: x["dt_obj"], reverse=True)[:15]

            # ลบ dt_obj ออกก่อนส่ง JSON
            for n in parsed_news:
                del n["dt_obj"]

    except Exception as e:
        print("News Error:", e)

    return parsed_news


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/analyze")
def analyze():
    ticker_symbol = request.args.get("ticker", "PLTR").upper().strip()
    period = request.args.get("period", "6mo")
    interval = request.args.get("interval", "1d")

    ticker_map = {
        "XAUUSD": "GC=F",
        "XAU/USD": "GC=F",
        "GOLD": "GC=F",
        "SILVER": "SI=F",
        "BTCUSD": "BTC-USD",
        "ETHUSD": "ETH-USD",
    }
    clean_symbol = ticker_map.get(ticker_symbol, ticker_symbol)

    try:
        ticker_obj = yf.Ticker(clean_symbol)
        df = ticker_obj.history(period=period, interval=interval)

        if df.empty:
            return jsonify({"error": f"ไม่พบข้อมูลสำหรับสัญลักษณ์ {ticker_symbol}"}), 404

        # คำนวณ Dividend Yield
        dividend_yield_pct = 0.0
        try:
            info = ticker_obj.info
            div_yield = info.get("dividendYield") or info.get("yield")
            if div_yield is not None:
                dividend_yield_pct = float(div_yield * 100 if div_yield < 1 else div_yield)
        except Exception:
            dividend_yield_pct = 0.0

        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["RSI"] = calculate_rsi(df["Close"])

        highs, lows = df["High"].values, df["Low"].values
        peaks_high, _ = find_peaks(highs, distance=10, prominence=2)
        peaks_low, _ = find_peaks(-lows, distance=10, prominence=2)

        last_h_idx = peaks_high[-1] if len(peaks_high) > 0 else np.argmax(highs)
        last_l_idx = peaks_low[-1] if len(peaks_low) > 0 else np.argmin(lows)
        swing_high, swing_low = highs[last_h_idx], lows[last_l_idx]
        diff = swing_high - swing_low

        is_uptrend = bool(last_h_idx > last_l_idx)

        gz_top = (swing_high - (0.500 * diff)) if is_uptrend else (swing_low + (0.618 * diff))
        gz_bottom = (swing_high - (0.618 * diff)) if is_uptrend else (swing_low + (0.500 * diff))

        current_price = float(df["Close"].iloc[-1])
        current_rsi = float(df["RSI"].iloc[-1])
        gz_min = float(min(gz_top, gz_bottom))
        gz_max = float(max(gz_top, gz_bottom))
        in_gz = bool(gz_min <= current_price <= gz_max)

        # Demand & Supply Zones
        supply_top = float(swing_high)
        supply_bottom = float(max(df["Open"].iloc[last_h_idx], df["Close"].iloc[last_h_idx]))
        demand_bottom = float(swing_low)
        demand_top = float(min(df["Open"].iloc[last_l_idx], df["Close"].iloc[last_l_idx]))

        chart_data = []
        for idx, row in df.iterrows():
            chart_data.append({
                "time": idx.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "ema20": float(row["EMA_20"]) if not pd.isna(row["EMA_20"]) else None,
                "ema50": float(row["EMA_50"]) if not pd.isna(row["EMA_50"]) else None,
                "rsi": float(row["RSI"]) if not pd.isna(row["RSI"]) else None,
            })

        news_data = scrape_news_vader(ticker_symbol)
        avg_score = float(np.mean([n["score"] for n in news_data])) if news_data else 0.0
        overall_sentiment = (
            "OVERALL BULLISH"
            if avg_score >= 0.05
            else ("OVERALL BEARISH" if avg_score <= -0.05 else "NEUTRAL")
        )

        return jsonify({
            "ticker": ticker_symbol,
            "current_price": current_price,
            "rsi": current_rsi,
            "div_yield": dividend_yield_pct,
            "in_gz": in_gz,
            "gz_min": gz_min,
            "gz_max": gz_max,
            "is_uptrend": is_uptrend,
            "supply_top": supply_top,
            "supply_bottom": supply_bottom,
            "demand_top": demand_top,
            "demand_bottom": demand_bottom,
            "chart_data": chart_data,
            "news": news_data,
            "news_avg_score": round(avg_score, 3),
            "news_overall": overall_sentiment,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/screener")
def screener():
    market = request.args.get("market", "popular")

    lists = {
        "popular": [
            "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD",
            "PLTR", "WDC", "MU", "NFLX", "DELTA.BK", "ADVANC.BK", "GULF.BK",
            "PTT.BK", "BDMS.BK", "KBANK.BK", "CPALL.BK", "AOT.BK"
        ],
        "us": ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "PLTR", "AVGO", "INTC", "ORCL"],
        "thai": ["DELTA.BK", "ADVANC.BK", "GULF.BK", "PTT.BK", "BDMS.BK", "KBANK.BK", "CPALL.BK", "AOT.BK", "BBL.BK", "SCB.BK"]
    }

    target_list = lists.get(market, lists["popular"])
    matched = []

    for symbol in target_list:
        try:
            df = yf.download(symbol, period="3mo", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 20:
                continue

            highs, lows, closes = df["High"].values, df["Low"].values, df["Close"].values
            p_high, _ = find_peaks(highs, distance=10, prominence=2)
            p_low, _ = find_peaks(-lows, distance=10, prominence=2)

            lh = p_high[-1] if len(p_high) > 0 else np.argmax(highs)
            ll = p_low[-1] if len(p_low) > 0 else np.argmin(lows)

            sh, sl, cp = highs[lh], lows[ll], closes[-1]
            diff = sh - sl
            uptrend = bool(lh > ll)

            gz_top = (sh - (0.500 * diff)) if uptrend else (sl + (0.618 * diff))
            gz_bottom = (sh - (0.618 * diff)) if uptrend else (sl + (0.500 * diff))

            gz_min, gz_max = min(gz_top, gz_bottom), max(gz_top, gz_bottom)

            if gz_min <= cp <= gz_max:
                rsi = float(calculate_rsi(df["Close"]).iloc[-1])
                matched.append({
                    "symbol": symbol,
                    "price": float(cp),
                    "gz_min": float(gz_min),
                    "gz_max": float(gz_max),
                    "rsi": round(rsi, 1),
                    "trend": "Uptrend" if uptrend else "Downtrend",
                })
        except Exception:
            continue

    return jsonify({"matched": matched})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
